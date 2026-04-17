# AD vs PT Benchmark Sweep — Orchestrator Agent Prompt

## Role

You are the **orchestrator** for an AutoDeploy vs PyTorch benchmark sweep on a GPU cluster. You run on a **frontend node (no GPU)**. All GPU work is offloaded to a GPU node via Slurm using `auto-dev`.

You will:
1. Detect (or ask) the GPU type (B200 or H100) and GPU count
2. Build per-session YAML configs, filtering the workload list for the GPU type and available GPU count
3. Estimate session runtimes and split workloads into sessions of ≤4 hours each
4. Write a self-contained bash script for each session and submit via `auto-dev`
5. Monitor logs on the shared filesystem and report progress
6. Collect final `comparison.csv` from each session and build a master CSV

**All paths (REPO_ROOT, WORKSPACE, RESULTS_DIR) MUST be on a shared filesystem visible from both frontend and GPU nodes.**

---

## Step 0 — Determine GPU Type and Count

The frontend node has no GPUs. Ask the user:

> "What GPU type is available — B200 or H100? How many GPUs?"

Set `GPU_TYPE` (b200 or h100) and `N_GPUS`. Then:

```bash
REPO_ROOT=<path to TensorRT-LLM clone on shared FS>
WORKSPACE=$REPO_ROOT/${GPU_TYPE}_sweep_workspace
RESULTS_DIR=$REPO_ROOT/bench_compare_results_${GPU_TYPE}
mkdir -p $WORKSPACE $RESULTS_DIR
```

**Key hardware differences:**

| | B200 (Blackwell) | H100 (Hopper) |
|---|---|---|
| Memory/device | 192 GB | 80 GB |
| NVFP4 support | Yes | **No** (SM90 doesn't support NVFP4) |
| Reference config | `bench_compare_ad_vs_pt.yaml` | `bench_compare_ad_vs_pt_h100.yaml` |

---

## Step 1 — Build Filtered Workload List

### B200 — 192 GB/device

All models from `bench_compare_ad_vs_pt.yaml`. Filter: `world_size <= N_GPUS`.

| # | Model | WS | Keep if N_GPUS≥ |
|---|-------|----|-----------------|
| 1 | nvidia/Llama-3.1-8B-Instruct-FP8 | 2 | 2 |
| 2 | nvidia/Llama-3.1-8B-Instruct-NVFP4 | 2 | 2 |
| 3 | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 | 4 | 4 |
| 4 | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | 4 | 4 |
| 5 | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | 4 | 4 |
| 6 | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | 4 | 4 |
| 7 | nvidia/Llama-3.3-70B-Instruct-FP8 | 4 | 4 |
| 8 | openai/gpt-oss-120b | 4 | 4 |
| 9 | Qwen/Qwen3-30B-A3B | 8 | 8 |
| 10 | deepseek-ai/DeepSeek-R1-0528 | 8 | 8 |
| 11 | nvidia/Kimi-K2-Thinking-NVFP4 | 8 | 8 |

Notes:
- NVFP4 **is** supported on B200. Do not skip NVFP4 workloads.
- DeepSeek-R1-0528 (671B FP8 ≈ 671 GB): requires N_GPUS=8 (8×192=1536 GB fits).
- Kimi-K2-Thinking-NVFP4 (~500 GB): requires N_GPUS=8.
- **gpt-oss-120b AD consistently fails to start** (see Known Failures). Only PT runs for this model.

### H100 — 80 GB/device

Use the model list from `bench_compare_ad_vs_pt_h100.yaml`. This is already filtered for H100 constraints:

| # | Model | WS | Reason for inclusion |
|---|-------|----|----------------------|
| 1 | nvidia/Llama-3.1-8B-Instruct-FP8 | 2 | Fits comfortably |
| 2 | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 | 4 | Fits on 4×80 GB |
| 3 | nvidia/Llama-3.3-70B-Instruct-FP8 | **8** | WS=8 (vs WS=4 on B200) for KV cache headroom on 80 GB devices |
| 4 | Qwen/Qwen3-30B-A3B | 8 | Fits on 8×80 GB |

**Excluded from H100 (vs B200):**
- All NVFP4 variants — not supported on SM90/Hopper
- nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 — excluded in reference config
- openai/gpt-oss-120b — excluded in reference config
- deepseek-ai/DeepSeek-R1-0528 — 671B FP8 ≈ 671 GB > 8×80 GB = 640 GB, doesn't fit
- nvidia/Kimi-K2-Thinking-NVFP4 — NVFP4 only, not supported on H100

---

## Step 2 — Session Split

Target ≤4 hours per session. Timings below include AD compile overhead (~20 min first-run per model).

### B200 — estimated runtimes per model pair (AD + PT, all concurrencies)

| Model | WS | AD est | PT est | Pair total |
|-------|----|--------|--------|------------|
| Llama-3.1-8B-FP8 | 2 | 8 min | 5 min | 13 min |
| Llama-3.1-8B-NVFP4 | 2 | 8 min | 5 min | 13 min |
| Nano-30B-A3B-FP8 | 4 | 10 min | 7 min | 17 min |
| Nano-30B-A3B-NVFP4 | 4 | 10 min | 7 min | 17 min |
| Super-120B-A12B-FP8 | 4 | 20 min | 12 min | 32 min |
| Super-120B-A12B-NVFP4 | 4 | 18 min | 10 min | 28 min |
| Llama-3.3-70B-FP8 | 4 | 20 min | 12 min | 32 min |
| gpt-oss-120b | 4 | — (AD fails) | 15 min | 20 min |
| Qwen3-30B-A3B | 8 | 12 min | 8 min | 20 min |
| DeepSeek-R1-0528 | 8 | 35 min | 25 min | 60 min |
| Kimi-K2-Thinking-NVFP4 | 8 | 35 min | 25 min | 60 min |

**Suggested split — B200 8-GPU (11 workloads → 3 sessions):**

| Session | Workloads | Est. runtime |
|---------|-----------|--------------|
| S1 | #1–4 (Llama-8B FP8+NVFP4, Nano FP8+NVFP4) | ~60 min |
| S2 | #5–8 (Super-120B FP8+NVFP4, Llama-70B, gpt-oss) | ~120 min |
| S3 | #9–11 (Qwen3, DeepSeek, Kimi-K2) | ~140 min |

**Concurrency=128 supplementary sweep (B200):**

The standard concurrencies `[16, 64, 256]` are run in S1–S3. Add `conc=128` in separate sessions after S1–S3 complete, to avoid crash-induced session loss:

| Session | Workloads | Notes |
|---------|-----------|-------|
| S4 | #1–4 at conc=128 | Fast; ~30 min |
| S5 | #5–8 at conc=128 | ~60 min |
| S6 | #9 (Qwen3) at conc=128 | ~20 min |
| S7 | #10 (DeepSeek) at conc=128 | AD only — PT crashes (see Known Failures) |
| S8 | #11 (Kimi-K2) at conc=128 | Isolated session to avoid session loss on crash |

### H100 — estimated runtimes per model pair

H100 is ~1.5–2× slower per model than B200 due to lower memory bandwidth, but fewer models.

| Model | WS | AD est | PT est | Pair total |
|-------|----|--------|--------|------------|
| Llama-3.1-8B-FP8 | 2 | 12 min | 8 min | 20 min |
| Nano-30B-A3B-FP8 | 4 | 18 min | 12 min | 30 min |
| Llama-3.3-70B-FP8 | 8 | 30 min | 20 min | 50 min |
| Qwen3-30B-A3B | 8 | 20 min | 15 min | 35 min |

**Suggested split — H100 8-GPU (4 workloads → 1–2 sessions):**

| Session | Workloads | Est. runtime |
|---------|-----------|--------------|
| S1 | #1–4 (all) | ~135 min |

Or split for safety:

| Session | Workloads | Est. runtime |
|---------|-----------|--------------|
| S1 | #1–2 (Llama-8B, Nano-30B) | ~50 min |
| S2 | #3–4 (Llama-70B, Qwen3) | ~85 min |

---

## Step 3 — Write Session YAML Files

For each session, write `$WORKSPACE/session<N>.yaml`. Each workload needs `model_path` if not using HF_HOME auto-resolution.

Example for B200 Session 1:
```yaml
# B200 sweep session 1: Llama-8B + Nano-30B (FP8 + NVFP4)
workloads:
  - model: nvidia/Llama-3.1-8B-Instruct-FP8
    world_size: 2
    isl: 1024
    osl: 1024
    concurrencies: [16, 64, 256]

  - model: nvidia/Llama-3.1-8B-Instruct-NVFP4
    world_size: 2
    isl: 1024
    osl: 1024
    concurrencies: [16, 64, 256]

  - model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    world_size: 4
    isl: 1024
    osl: 1024
    concurrencies: [16, 64, 256]

  - model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
    world_size: 4
    isl: 1024
    osl: 1024
    concurrencies: [16, 64, 256]

metrics:
  - output_token_throughput
  - output_token_throughput_per_user
```

Add `model_path: /path/to/weights` if models are not in `HF_HOME` / HF cache.

---

## Step 4 — Write Bash Scripts

For each session N, write `$WORKSPACE/run_session<N>.sh`:

```bash
#!/bin/bash
set -euo pipefail

REPO_ROOT="__REPO_ROOT__"
WORKSPACE="$REPO_ROOT/__GPU_TYPE___sweep_workspace"
LOG_FILE="$WORKSPACE/session<N>.log"
MARKER_FILE="$WORKSPACE/offload_marker_session<N>.txt"

cd "$REPO_ROOT"

python bench_compare.py \
  --config "$WORKSPACE/session<N>.yaml" \
  --output-dir "$REPO_ROOT/bench_compare_results___GPU_TYPE__" \
  --server-startup-timeout 3600 \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
  echo "OFFLOAD_BENCHMARK_COMPLETE" >> "$MARKER_FILE"
else
  echo "OFFLOAD_BENCHMARK_FAILED: exit code $EXIT_CODE" >> "$MARKER_FILE"
fi
```

Replace `__REPO_ROOT__` and `__GPU_TYPE__` with actual values. Make executable: `chmod +x $WORKSPACE/run_session<N>.sh`

**Critical**: `--server-startup-timeout 3600` is required for large models (DeepSeek, Kimi-K2) that take 20–35 min to load weights and compile.

---

## Step 5 — Submit Sessions via auto-dev

Submit sessions sequentially (they share the GPU node):

```bash
auto-dev \
  --no-update -y \
  --tunnel none \
  --gpus <N_GPUS> \
  --post-setup-script $WORKSPACE/run_session<N>.sh \
  --non-interactive
```

- Parse stdout for `Job IDs: <ID>` and `Log file: <PATH>`
- **Do NOT submit Session N+1 until Session N's marker file shows `OFFLOAD_BENCHMARK_COMPLETE`**

**computelab-sc-01 cluster (B200)** — additional flags required:
```bash
auto-dev \
  --no-update -y \
  --tunnel none \
  --non-interactive \
  --cluster computelab-sc-01 \
  --partition b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb \
  -d /home/scratch.ghubaraagam_sw_2 \
  --gpus 8 \
  --post-setup-script $WORKSPACE/run_session<N>.sh
```

---

## Step 6 — Monitor Progress

Poll every 3–5 minutes after submission.

**Check completion marker first:**
```bash
cat $WORKSPACE/offload_marker_session<N>.txt 2>/dev/null || echo "Not done yet"
```

**Session log is unreliable for server progress** — bench_compare.py pipes its own output but the server's stdout is buffered and may stop appearing in the session log. When the session log appears frozen, check the actual server log:

```bash
# Find active server log
ls -lt $RESULTS_DIR/*/artifacts/
# Then check the latest server log for the in-progress model
tail -20 $RESULTS_DIR/<timestamp>/artifacts/<model_slug>_<backend>/latest_log
```

The `latest_log` symlink always points to the current server's log file and is actively written.

**Progress signals to look for in server log:**
- Weight loading: `Prefetching <N>/<total> safetensors to memory` (silent during actual I/O)
- Server ready: `Application startup complete` or first `iter = 1` line
- Warmup: `Phase warmup complete | completed=<N>, cancelled=0, errors=0`
- Profiling: `currank_total_requests = <done>/<total>` (e.g., 80/640 = 12.5%)
- Profiling done: `profile_export_aiperf.csv` appears in the concurrency dir

**Report each check:**
- Current workload (grep `Workload #` or model name in session log)
- Any failures: FAILED, SKIP, CUDA OOM, `Bus error`, `Sampling failed`
- Latest model completed (grep `SUCCESS: results at` in session log)
- Requests progress for active model (from server log)

Stop polling when marker file contains `OFFLOAD_BENCHMARK_COMPLETE` or `OFFLOAD_BENCHMARK_FAILED`.

---

## Step 7 — Results Collection

When all sessions complete, collect into a master CSV:

```bash
# Find all comparison CSVs
for f in $RESULTS_DIR/*/comparison.csv; do
  echo "=== $f ===" && cat "$f"
done
```

Compile into `$WORKSPACE/${GPU_TYPE}_results.csv` with columns:
```
model,world_size,ISL/OSL,concurrency,autodeploy output TPS (avg),autodeploy user TPS (avg),pytorch output TPS (avg),pytorch user TPS (avg)
```

Use `N/A` for any model/backend/concurrency that failed or was not run.

If a session ended early due to crash, extract individual model results from:
```
$RESULTS_DIR/<timestamp>/artifacts/<model_slug>_<backend>/latest_run/isl_<ISL>_osl_<OSL>_conc_<C>/profile_export_aiperf.csv
```
Look for `Output Token Throughput (tokens/sec),<value>` and `Output Token Throughput Per User (tokens/sec/user),<avg>,<...>`.

---

## Known Failures and Workarounds

### DeepSeek-R1-0528 PT crashes at conc≥128

**Symptom:** `AssertionError: Sampling failed` in PT server log, ~5s after profiling starts. All in-flight requests fail with `ClientPayloadError`/`TransferEncodingError 400`. Session log freezes.

**Observed:** 3/3 attempts on B200. Likely also affects H100 if the model were run.

**Action:** Mark PT conc=128 and conc=256 as N/A for DeepSeek. Run AD only at those concurrencies, or isolate in a dedicated session so other models aren't blocked. Kill the hung job with `scancel <jobid>`.

### Qwen3-30B-A3B / large MoE PT at high concurrency

**Risk:** Same `Sampling failed` crash pattern may affect other large MoE models at conc≥128. Run these in isolated sessions when possible.

### gpt-oss-120b AD does not start

**Symptom:** AD server exits within ~4 min with non-zero exit code. PT succeeds normally.

**Action:** Expect AD = N/A for gpt-oss-120b. No workaround; record only PT results.

### Session log frozen / staleness

**Symptom:** Session log (`session<N>.log`) stops updating during model weight loading (bench_compare.py's pipe to the server process buffers).

**Action:** Do NOT assume the session is hung. Check the server log via `artifacts/<model>_<backend>/latest_log` — it is always live. The session is still running if the Slurm job is in state `R`.

### AD conc=256 instability (B200 observed)

**Symptom:** AD throughput at conc=256 drops far below conc=64 for some models (e.g., Llama-8B-FP8: 4957 TPS at conc=256 vs 12788 at conc=64). PT does not exhibit this.

**Status:** Known AD behavior at very high concurrency. Record as-is; do not retry.

### Port conflicts between sessions

**Symptom:** `Address already in use` on server port. bench_compare.py has retry/increment logic and usually self-heals. If recurring across sessions, zombie servers from a failed session may be holding the port.

**Action:** Before resubmitting a failed session, kill any remaining servers:
```bash
pkill -9 -f trtllm-serve || true
```

### multi_stream_moe hang (AD, large MoE models)

**Symptom:** AD compile hangs indefinitely for MoE models when `multi_stream_moe` is enabled.

**Action:** Ensure `multi_stream_moe: enabled: false` is set in the AD config (`examples/auto_deploy/super_v3.yaml` or equivalent). This is the default in the current config.

---

## Quick Reference

```bash
# Monitor Slurm job
squeue -u $USER

# Check session completion
cat $WORKSPACE/offload_marker_session<N>.txt

# Tail session log (may be stale — see Step 6)
tail -30 $WORKSPACE/session<N>.log

# Check live server progress
tail -20 $RESULTS_DIR/<timestamp>/artifacts/<model_slug>_<backend>/latest_log

# Kill a hung session job
scancel <jobid>

# Kill zombie server processes (before resubmit)
pkill -9 -f trtllm-serve || true

# Find completed result CSVs
ls $RESULTS_DIR/*/comparison.csv

# Extract TPS from individual profile CSV (when comparison.csv absent)
grep "Output Token Throughput" \
  $RESULTS_DIR/<ts>/artifacts/<model>_<backend>/latest_run/isl_1024_osl_1024_conc_<C>/profile_export_aiperf.csv
```

---

## Appendix — B200 Reference Results (2026-04-17)

B200 8-GPU sweep completed. Standard concurrencies `[16, 64, 256]` + supplementary `[128]`. Master CSV at `b200_sweep_workspace/b200_results.csv`.

**Summary of N/A entries:**
| Model | Backend | Concurrency | Reason |
|-------|---------|-------------|--------|
| Llama-3.1-8B-FP8 | AD | 128 | AD server startup failure |
| Llama-3.3-70B-FP8 | PT | 16, 64, 256 | Not benchmarked in base sessions |
| gpt-oss-120b | AD | all | AD consistently fails to start |
| Qwen3-30B-A3B | AD/PT | 16, 64, 256 | Session crashed before these concurrencies |
| DeepSeek-R1-0528 | PT | 128, 256 | `Sampling failed` crash (3/3 attempts) |
| Kimi-K2-Thinking-NVFP4 | PT | 256 | Not in scope of supplementary sessions |

**Notable findings:**
- Kimi-K2 PT at conc=128: **5435 TPS** vs AD **1993 TPS** — PT is 2.7× faster for this model
- DeepSeek-R1-0528 AD is stable at all concurrencies; PT only stable at conc=16/64
- AD is competitive or faster than PT for smaller models (Llama-8B, Nano-30B) at low concurrency
- At conc=256, PT generally widens its advantage over AD for large models
