#!/usr/bin/env python3
# Copyright 2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare AutoDeploy vs PyTorch backend performance using sweep benchmarks.

Usage:
    python bench_compare.py --config workloads.yaml [options]

Input YAML format (workloads.yaml):
    workloads:
      - model: meta-llama/Llama-3.1-70B-Instruct
        world_size: 4
        isl: 1024
        osl: 1024
        concurrencies: [32, 64]
      - model: deepseek-ai/DeepSeek-R1
        world_size: 8
        isl: 1024
        osl: 2048
        concurrencies: [32, 64]

    metrics:
      - output_token_throughput     # or alias: "output tps", "otps"
      - output_token_throughput_per_user  # or alias: "user tps", "utps"
      - time_to_first_token         # or alias: "ttft"
      - inter_token_latency         # or alias: "itl"
      - request_latency
      - request_throughput          # or alias: "rps"
      - total_token_throughput

Outputs (under --output-dir/<timestamp>/):
  comparison.csv       - CSV table: one row per (model, world_size, ISL/OSL, concurrency)
  benchmark_log.txt    - Commands issued, success/fail/skip per run, warnings
  artifacts/           - Sweep artifact directories per (model, backend)
"""

import argparse
import csv
import json
import os

import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


_DEFAULT_PORT = 8100


# ── Metric name aliases → canonical JSON field ─────────────────────────────────
METRIC_ALIASES: Dict[str, str] = {
    "output_token_throughput": "output_token_throughput",
    "output tps": "output_token_throughput",
    "otps": "output_token_throughput",
    "output_tps": "output_token_throughput",
    "output_token_throughput_per_user": "output_token_throughput_per_user",
    "user tps": "output_token_throughput_per_user",
    "utps": "output_token_throughput_per_user",
    "user_tps": "output_token_throughput_per_user",
    "time_to_first_token": "time_to_first_token",
    "ttft": "time_to_first_token",
    "inter_token_latency": "inter_token_latency",
    "itl": "inter_token_latency",
    "inter_chunk_latency": "inter_chunk_latency",
    "request_latency": "request_latency",
    "e2e_latency": "request_latency",
    "request_throughput": "request_throughput",
    "rps": "request_throughput",
    "total_token_throughput": "total_token_throughput",
    "total_tps": "total_token_throughput",
    "prefill_throughput_per_user": "prefill_throughput_per_user",
    "time_to_second_token": "time_to_second_token",
    "t2st": "time_to_second_token",
}

# Default human-readable display names for CSV headers (fallback: use canonical key)
METRIC_DISPLAY: Dict[str, str] = {
    "output_token_throughput": "output TPS (avg)",
    "output_token_throughput_per_user": "user TPS (avg)",
    "time_to_first_token": "TTFT ms (avg)",
    "inter_token_latency": "ITL ms (avg)",
    "inter_chunk_latency": "inter_chunk_latency ms (avg)",
    "request_latency": "request latency ms (avg)",
    "request_throughput": "req/s (avg)",
    "total_token_throughput": "total TPS (avg)",
    "prefill_throughput_per_user": "prefill TPS/user (avg)",
    "time_to_second_token": "T2ST ms (avg)",
}


def resolve_metric(name: str) -> str:
    """Resolve metric name or alias to the canonical JSON key."""
    key = METRIC_ALIASES.get(name.lower())
    if key is None:
        key = name  # assume it's already a valid JSON field name
    return key


def metric_display_name(canonical: str) -> str:
    return METRIC_DISPLAY.get(canonical, f"{canonical} (avg)")


# ── Config helpers ──────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Nested dict merge: override updates base recursively."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def lookup_ad_model(model: str, trtllm_root: Path) -> Optional[List[str]]:
    """Return yaml_extra list for model from AD registry, or None if not found."""
    models_yaml = trtllm_root / "examples/auto_deploy/model_registry/models.yaml"
    with open(models_yaml) as f:
        registry = yaml.safe_load(f)
    for entry in registry.get("models", []):
        if entry["name"] == model:
            return entry.get("yaml_extra", [])
    return None


def build_ad_config(yaml_extra: List[str], trtllm_root: Path) -> dict:
    """Merge yaml_extra config files left to right (each overrides the previous)."""
    configs_dir = trtllm_root / "examples/auto_deploy/model_registry/configs"
    merged: dict = {}
    for fname in yaml_extra:
        path = configs_dir / fname
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        merged = deep_merge(merged, cfg)
    return merged


def resolve_local_model_path(model: str) -> Optional[str]:
    """Return local HF snapshot path for model, or None if not cached."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model, local_files_only=True)
    except Exception:
        return None


def lookup_pt_config(
    model: str, trtllm_root: Path, prefer_scenario: str = "Max Throughput"
) -> Optional[str]:
    """Return config_path for model from pytorch curated lookup.yaml.

    Prefers `prefer_scenario`; falls back to first non-disaggregated match.
    Returns None if model is not found.
    """
    lookup_yaml = trtllm_root / "examples/configs/curated/lookup.yaml"
    with open(lookup_yaml) as f:
        entries = yaml.safe_load(f)

    matches = [e for e in entries if e.get("model") == model and not e.get("disagg", False)]
    if not matches:
        return None

    for entry in matches:
        if entry.get("scenario", "") == prefer_scenario:
            return entry["config_path"]

    return matches[0]["config_path"]


# ── Result parsing ──────────────────────────────────────────────────────────────

def find_latest_run(result_base: Path) -> Optional[Path]:
    """Resolve the latest_run symlink in result_base to get the actual run dir."""
    link = result_base / "latest_run"
    if link.is_symlink():
        target = link.resolve()
        if target.exists():
            return target
    return None


def parse_sweep_results(
    run_dir: Path,
    isl: int,
    osl: int,
    concurrencies: List[int],
    metrics: List[str],
) -> Dict[int, Dict[str, Optional[float]]]:
    """Parse profile_export_aiperf.json for each concurrency; returns avg values."""
    results: Dict[int, Dict[str, Optional[float]]] = {}
    for conc in concurrencies:
        conc_dir = run_dir / f"isl_{isl}_osl_{osl}_conc_{conc}"
        json_path = conc_dir / "profile_export_aiperf.json"
        row: Dict[str, Optional[float]] = {}
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for metric in metrics:
                m_data = data.get(metric)
                row[metric] = m_data.get("avg") if isinstance(m_data, dict) else None
        else:
            for metric in metrics:
                row[metric] = None
        results[conc] = row
    return results


# ── Logger ──────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lines: List[str] = []

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._lines.append(line)

    def save(self):
        with open(self.log_path, "w") as f:
            f.write("\n".join(self._lines) + "\n")


# ── Sweep runner ────────────────────────────────────────────────────────────────

def run_sweep(
    *,
    model: str,
    config_path: str,
    server_type: str,
    world_size: Optional[int],
    isl: int,
    osl: int,
    concurrencies: List[int],
    result_base: Path,
    tag: str,
    startup_timeout: int,
    warmup_requests: str,
    rounds_per_concurrency: int,
    extra_sweep_args: List[str],
    logger: Logger,
    dry_run: bool,
    port: int = _DEFAULT_PORT,
) -> Tuple[Optional[Path], bool, int]:
    """Invoke sweep for one (model, backend) pair.

    Returns (run_dir, success, next_port). run_dir is None on failure or dry_run.
    next_port is the port to pass to the next run_sweep call: same as port if freed
    within timeout, or port+1 if the port was still occupied (escalation fallback).
    """
    conc_str = " ".join(str(c) for c in concurrencies)
    cmd = [
        "sweep",
        "--model", model,
        "--config-path", config_path,
        "--server-type", server_type,
        "--server-startup-timeout", str(startup_timeout),
        "--isl", str(isl),
        "--osl", str(osl),
        "--concurrencies", conc_str,
        "--rounds-per-concurrency", str(rounds_per_concurrency),
        "--tag", tag,
        "--result-base-dir", str(result_base),
        "--warmup-requests", warmup_requests,
        "--port", str(port),
    ]
    if world_size is not None and server_type == "trtllm-autodeploy":
        cmd += ["--world-size", str(world_size)]
    cmd += extra_sweep_args

    logger.log(f"CMD: {' '.join(cmd)}")

    if dry_run:
        logger.log("DRY RUN: skipping execution")
        return None, True, port

    result_base.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.log(f"FAILED: sweep exited with code {e.returncode}")
        return None, False, port + 1
    except FileNotFoundError:
        logger.log("FAILED: 'sweep' command not found. Is ad-perf-utils installed?")
        return None, False, port + 1

    run_dir = find_latest_run(result_base)
    if run_dir is None:
        logger.log(f"WARNING: latest_run symlink not found in {result_base}")
        return None, False, next_port

    logger.log(f"SUCCESS: results at {run_dir}")
    return run_dir, True, port + 1


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AutoDeploy vs PyTorch backend and produce a comparison CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="YAML file with workloads and metrics")
    parser.add_argument(
        "--output-dir",
        default="./bench_compare_results",
        help="Base output directory; a timestamped sub-dir is created per run (default: %(default)s)",
    )
    parser.add_argument(
        "--trtllm-root",
        default=None,
        help="TensorRT-LLM repo root (auto-detected from script location or cwd if omitted)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["autodeploy", "pytorch"],
        default=["autodeploy", "pytorch"],
        help="Which backends to benchmark (default: both)",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=3600,
        help="Seconds to wait for server startup (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-requests",
        default="match_concurrency",
        help="Warmup request count; integer or 'match_concurrency' (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds-per-concurrency",
        type=int,
        default=5,
        help="Number of AIPerf rounds per concurrency level (default: %(default)s)",
    )
    parser.add_argument(
        "--pt-scenario",
        default="Max Throughput",
        help="PyTorch config scenario preference from lookup.yaml (default: '%(default)s')",
    )
    parser.add_argument(
        "--extra-sweep-args",
        default="",
        help="Extra arguments forwarded verbatim to sweep (space-separated string)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    # ── Resolve TensorRT-LLM root ────────────────────────────────────────────
    if args.trtllm_root:
        trtllm_root = Path(args.trtllm_root).resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        if (script_dir / "examples/auto_deploy").exists():
            trtllm_root = script_dir
        elif (Path.cwd() / "examples/auto_deploy").exists():
            trtllm_root = Path.cwd()
        else:
            sys.exit(
                "ERROR: Cannot find TensorRT-LLM root. "
                "Run from the repo root or pass --trtllm-root."
            )

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_metrics: List[str] = config.get(
        "metrics", ["output_token_throughput", "output_token_throughput_per_user"]
    )
    # Resolve aliases and deduplicate while preserving order
    seen_metrics = set()
    metrics: List[str] = []  # canonical JSON field names
    metric_labels: List[str] = []  # display label (raw user name, title-cased)
    for raw in raw_metrics:
        canonical = resolve_metric(raw)
        if canonical not in seen_metrics:
            seen_metrics.add(canonical)
            metrics.append(canonical)
            # Use user-supplied name as display label if it differs from canonical
            metric_labels.append(raw if raw != canonical else metric_display_name(canonical))

    workloads: List[dict] = config.get("workloads", [])
    if not workloads:
        sys.exit("ERROR: No workloads specified in config YAML.")

    extra_sweep_args = args.extra_sweep_args.split() if args.extra_sweep_args else []

    # ── Set up output directory ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "benchmark_log.txt"
    csv_path = output_dir / "comparison.csv"
    logger = Logger(log_path)

    logger.log(f"bench_compare.py started")
    logger.log(f"TensorRT-LLM root: {trtllm_root}")
    logger.log(f"Output directory:  {output_dir}")
    logger.log(f"Backends:          {args.backends}")
    logger.log(f"Metrics:           {metrics}")
    logger.log(f"Workloads:         {len(workloads)}")
    logger.log(f"Dry run:           {args.dry_run}")

    # ── Results storage: (model, world_size, isl, osl, conc) → {backend: {metric: val}} ──
    ResultKey = Tuple[str, int, int, int, int]
    all_results: Dict[ResultKey, Dict[str, Dict[str, Optional[float]]]] = {}

    current_port: int = _DEFAULT_PORT  # threaded across sweeps; escalates on timeout

    for wl_idx, wl in enumerate(workloads):
        model: str = wl["model"]
        world_size: int = int(wl.get("world_size", 1))
        isl: int = int(wl["isl"])
        osl: int = int(wl["osl"])
        concurrencies: List[int] = [int(c) for c in wl.get("concurrencies", [])]

        if not concurrencies:
            logger.log(f"\nWARNING: Workload #{wl_idx + 1} ({model}) has no concurrencies — skipping.")
            continue

        model_slug = model.replace("/", "_").replace(".", "-")

        logger.log(f"\n{'=' * 70}")
        logger.log(
            f"Workload #{wl_idx + 1}: {model} | WS={world_size} | "
            f"ISL={isl} | OSL={osl} | Conc={concurrencies}"
        )

        null_metrics = {m: None for m in metrics}

        # ── AutoDeploy ──────────────────────────────────────────────────────
        ad_results: Dict[int, Dict[str, Optional[float]]] = {
            c: dict(null_metrics) for c in concurrencies
        }
        if "autodeploy" in args.backends:
            logger.log("\n--- AutoDeploy ---")
            yaml_extra = lookup_ad_model(model, trtllm_root)

            if yaml_extra is None:
                logger.log(
                    f"SKIP: {model} not found in "
                    "examples/auto_deploy/model_registry/models.yaml"
                )
            else:
                logger.log(f"AD yaml_extra: {yaml_extra}")
                merged_cfg = build_ad_config(yaml_extra, trtllm_root)

                # Warn if registry world_size differs from requested
                registry_ws = merged_cfg.get("world_size")
                if registry_ws is not None and int(registry_ws) != world_size:
                    logger.log(
                        f"WARNING: Registry world_size={registry_ws} differs from "
                        f"requested world_size={world_size}. "
                        f"--world-size {world_size} will be passed to sweep."
                    )

                # Write merged config to a temp file
                tmp = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".yaml",
                    prefix=f"ad_cfg_{model_slug}_",
                    dir=str(output_dir),
                    delete=False,
                )
                yaml.dump(merged_cfg, tmp)
                tmp.close()
                ad_config_path = tmp.name
                logger.log(f"Merged AD config written to: {ad_config_path}")

                ad_result_base = artifacts_dir / f"{model_slug}_ad"
                # Trim tag to keep directory names reasonable
                ad_tag = f"{model_slug}-ws{world_size}-ad"[:64]

                run_dir, success, current_port = run_sweep(
                    model=model,
                    config_path=ad_config_path,
                    server_type="trtllm-autodeploy",
                    world_size=world_size,
                    isl=isl,
                    osl=osl,
                    concurrencies=concurrencies,
                    result_base=ad_result_base,
                    tag=ad_tag,
                    startup_timeout=args.server_startup_timeout,
                    warmup_requests=args.warmup_requests,
                    rounds_per_concurrency=args.rounds_per_concurrency,
                    extra_sweep_args=extra_sweep_args,
                    logger=logger,
                    dry_run=args.dry_run,
                    port=current_port,
                )
                if run_dir:
                    ad_results = parse_sweep_results(run_dir, isl, osl, concurrencies, metrics)
                    # Check for missing concurrency dirs
                    for conc in concurrencies:
                        conc_dir = run_dir / f"isl_{isl}_osl_{osl}_conc_{conc}"
                        if not conc_dir.exists():
                            logger.log(
                                f"WARNING: AD result dir not found for conc={conc}: {conc_dir}"
                            )
                elif not args.dry_run:
                    # Clean up temp config only if sweep was run (not dry-run)
                    pass
                if not args.dry_run and not success:
                    pass  # ad_results already initialized to nulls

        # ── PyTorch ──────────────────────────────────────────────────────────
        pt_results: Dict[int, Dict[str, Optional[float]]] = {
            c: dict(null_metrics) for c in concurrencies
        }
        if "pytorch" in args.backends:
            logger.log("\n--- PyTorch ---")
            pt_config_rel = lookup_pt_config(model, trtllm_root, prefer_scenario=args.pt_scenario)

            if pt_config_rel is None:
                logger.log(
                    f"SKIP: {model} not found in "
                    "examples/configs/curated/lookup.yaml"
                )
            else:
                pt_config_full = str(trtllm_root / pt_config_rel)
                logger.log(f"PT config: {pt_config_rel}")

                # Inject local tokenizer path so trtllm-serve never contacts HF Hub.
                local_model_path = resolve_local_model_path(model)
                if local_model_path:
                    logger.log(f"PT local tokenizer: {local_model_path}")
                    with open(pt_config_full) as f:
                        pt_cfg = yaml.safe_load(f) or {}
                    pt_cfg["tokenizer"] = local_model_path
                    pt_cfg["postprocess_tokenizer_dir"] = local_model_path
                    pt_tmp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".yaml", prefix="pt_cfg_", delete=False
                    )
                    yaml.dump(pt_cfg, pt_tmp)
                    pt_tmp.flush()
                    pt_config_full = pt_tmp.name
                else:
                    logger.log(f"WARNING: {model} not in local HF cache; tokenizer may fail if Hub is unavailable")

                pt_result_base = artifacts_dir / f"{model_slug}_pt"
                pt_tag = f"{model_slug}-ws{world_size}-pt"[:64]

                run_dir, _, current_port = run_sweep(
                    model=model,
                    config_path=pt_config_full,
                    server_type="trtllm-pytorch",
                    world_size=None,  # PT world_size is set in config YAML, not CLI
                    isl=isl,
                    osl=osl,
                    concurrencies=concurrencies,
                    result_base=pt_result_base,
                    tag=pt_tag,
                    startup_timeout=args.server_startup_timeout,
                    warmup_requests=args.warmup_requests,
                    rounds_per_concurrency=args.rounds_per_concurrency,
                    extra_sweep_args=extra_sweep_args,
                    logger=logger,
                    dry_run=args.dry_run,
                    port=current_port,
                )
                if run_dir:
                    pt_results = parse_sweep_results(run_dir, isl, osl, concurrencies, metrics)
                    for conc in concurrencies:
                        conc_dir = run_dir / f"isl_{isl}_osl_{osl}_conc_{conc}"
                        if not conc_dir.exists():
                            logger.log(
                                f"WARNING: PT result dir not found for conc={conc}: {conc_dir}"
                            )

        # Store per-concurrency results
        for conc in concurrencies:
            key: ResultKey = (model, world_size, isl, osl, conc)
            all_results[key] = {
                "autodeploy": ad_results.get(conc, dict(null_metrics)),
                "pytorch": pt_results.get(conc, dict(null_metrics)),
            }

    # ── Write CSV ────────────────────────────────────────────────────────────
    # Header: model, world_size, ISL/OSL, concurrency,
    #         autodeploy <metric1>, ..., pytorch <metric1>, ...
    csv_headers = ["model", "world_size", "ISL/OSL", "concurrency"]
    for m in metrics:
        csv_headers.append(f"autodeploy {metric_display_name(m)}")
    for m in metrics:
        csv_headers.append(f"pytorch {metric_display_name(m)}")

    def fmt(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for (model, world_size, isl, osl, conc), backend_results in all_results.items():
            row: dict = {
                "model": model,
                "world_size": world_size,
                "ISL/OSL": f"{isl}/{osl}",
                "concurrency": conc,
            }
            for m in metrics:
                row[f"autodeploy {metric_display_name(m)}"] = fmt(
                    backend_results["autodeploy"].get(m)
                )
            for m in metrics:
                row[f"pytorch {metric_display_name(m)}"] = fmt(
                    backend_results["pytorch"].get(m)
                )
            writer.writerow(row)

    # ── Final log summary ─────────────────────────────────────────────────────
    logger.log(f"\n{'=' * 70}")
    logger.log(f"CSV:      {csv_path}")
    logger.log(f"Artifacts: {artifacts_dir}")
    logger.log(f"Log:       {log_path}")
    logger.save()

    print(f"\nDone. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
