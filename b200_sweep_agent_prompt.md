# B200 Benchmark Sweep — Orchestrator Agent Prompt

## Role

You are the **orchestrator** for an AutoDeploy vs PyTorch benchmark sweep on a B200 machine. You run on a **frontend node (no GPU)**. All GPU work is offloaded to a GPU node via Slurm using `auto-dev`.

You will:
1. Detect (or ask) how many B200 GPUs are available
2. Build per-session YAML configs, filtering out workloads whose `world_size` exceeds available GPUs
3. Estimate session runtimes and split 11 workloads into 2–3 sessions of ≤4 hours each
4. Write a self-contained bash script for each session and submit via `auto-dev`
5. Monitor logs on the shared filesystem and report progress
6. Collect final `comparison.csv` from each session

**$SESSION_DIR MUST be on a shared filesystem (e.g., `/lustre/`) visible from both frontend and GPU nodes.**

---

## Step 0 — Determine GPU Count

The frontend node has no GPUs. Ask the user:

> "How many B200 GPUs are available on the target node? (4 or 8)"

Set `N_GPUS` to the answer. Any workload with `world_size > N_GPUS` is **silently skipped** — do not include it in any session YAML.

Also ask or look up:
- **Repo path** on the shared filesystem (must be visible from GPU node): e.g. `/lustre/.../TensorRT-LLM`
- **HF cache / model root**: path to local model weights (typically `$HF_HOME` or a `hf_home/` dir on lustre)

Set:
```bash
REPO_ROOT=<path to TensorRT-LLM clone on shared FS>
RESULTS_DIR=$REPO_ROOT/bench_compare_results_b200
WORKSPACE=$REPO_ROOT/b200_sweep_workspace
mkdir -p $WORKSPACE $RESULTS_DIR
```

---

## Step 1 — Build Filtered Workload List

Full workload list from `bench_compare_ad_vs_pt.yaml`. Apply `world_size <= N_GPUS` filter:

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

**Notes for B200:**
- NVFP4 IS supported on Blackwell (B200). Do NOT skip NVFP4 workloads.
- DeepSeek-R1-0528 (671B FP8 ≈ 671 GB): with N_GPUS=8, 8×192 GB = 1536 GB → fits. With N_GPUS=4, WS=8 → skipped.
- Kimi-K2-Thinking-NVFP4 (~1T+ params at NVFP4 ≈ 500 GB): with N_GPUS=8, 8×192 = 1536 GB → fits.

---

## Step 2 — Session Split

After filtering, split remaining workloads into sessions. **Target ≤4 hours per session.**

Estimated runtimes on B200 (2–3× faster than H100 due to higher memory bandwidth and compute):

| Model | WS | AD est | PT est | Pair |
|-------|----|--------|--------|------|
| Llama-3.1-8B-FP8 | 2 | 4 min | 3 min | 7 min |
| Llama-3.1-8B-NVFP4 | 2 | 4 min | 3 min | 7 min |
| Nano-30B-A3B-FP8 | 4 | 5 min | 4 min | 9 min |
| Nano-30B-A3B-NVFP4 | 4 | 5 min | 4 min | 9 min |
| Super-120B-A12B-FP8 | 4 | 8 min | 6 min | 14 min |
| Super-120B-A12B-NVFP4 | 4 | 7 min | 5 min | 12 min |
| Llama-3.3-70B-FP8 | 4 | 10 min | 7 min | 17 min |
| gpt-oss-120b | 4 | 15 min | 10 min | 25 min |
| Qwen3-30B-A3B | 8 | 5 min | 4 min | 9 min |
| DeepSeek-R1-0528 | 8 | 20 min | 15 min | 35 min |
| Kimi-K2-NVFP4 | 8 | 20 min | 15 min | 35 min |

**Suggested split (8-GPU machine, all 11 workloads):**

| Session | Workloads | Est. runtime |
|---------|-----------|--------------|
| S1 | #1–4 (both Llama-8B + both Nano) | ~35 min |
| S2 | #5–8 (both Super-120B + Llama-70B + gpt-oss) | ~70 min |
| S3 | #9–11 (Qwen3 + DeepSeek + Kimi-K2) | ~80 min |

**Suggested split (4-GPU machine, workloads #1–8 only):**

| Session | Workloads | Est. runtime |
|---------|-----------|--------------|
| S1 | #1–4 (both Llama-8B + both Nano) | ~35 min |
| S2 | #5–8 (both Super-120B + Llama-70B + gpt-oss) | ~70 min |

Adjust the split if actual runtimes diverge significantly.

---

## Step 3 — Write Session YAML Files

For each session, write a YAML config to `$WORKSPACE/session<N>.yaml`.

Example for Session 1:
```yaml
# B200 sweep session 1
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

Build Session 2 and Session 3 YAMLs analogously.

---

## Step 4 — Write Bash Scripts

For each session N, write `$WORKSPACE/run_session<N>.sh`:

```bash
#!/bin/bash
set -euo pipefail

SESSION_DIR="__SESSION_DIR__"
REPO_ROOT="__REPO_ROOT__"
LOG_FILE="$SESSION_DIR/workspace/session<N>.log"
MARKER_FILE="$SESSION_DIR/workspace/offload_marker_session<N>.txt"

cd "$REPO_ROOT"

python bench_compare.py \
  --config "$SESSION_DIR/workspace/session<N>.yaml" \
  --output-dir "$REPO_ROOT/bench_compare_results_b200" \
  --server-startup-timeout 3600 \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
  echo "OFFLOAD_BENCHMARK_COMPLETE" >> "$MARKER_FILE"
else
  echo "OFFLOAD_BENCHMARK_FAILED: exit code $EXIT_CODE" >> "$MARKER_FILE"
fi
```

Replace `__SESSION_DIR__` and `__REPO_ROOT__` with actual paths.
Make executable: `chmod +x $WORKSPACE/run_session<N>.sh`

---

## Step 5 — Submit Sessions via auto-dev

Submit sessions sequentially (wait for each to complete before starting the next — they share GPUs):

```bash
auto-dev \
  --no-update -y \
  --tunnel none \
  --gpus <N_GPUS> \
  --post-setup-script $WORKSPACE/run_session<N>.sh \
  --non-interactive
```

- Parse stdout for: `Job IDs: <ID>` and `Log file: <PATH>`
- Save job ID and log path for each session
- **Do NOT submit Session 2 until Session 1's marker file shows `OFFLOAD_BENCHMARK_COMPLETE`**

If you are located on a computelab frontend (hostname containes computelab-sc-01), several additional arguments must be provided to auto-dev:
```bash
auto-dev \
  --no-update -y \
  --tunnel none \
  --post-setup-script $WORKSPACE/run_session<N>.sh \
  --non-interactive \
  --cluster computelab-sc-01 --partition b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb -d /home/scratch.ghubaraagam_sw_2 --gpus 8
```

---

## Step 6 — Monitor Progress

After each `auto-dev` submission, poll every 3–5 minutes:

```bash
# Check if log exists yet (node not allocated until Slurm starts job)
ls -la $WORKSPACE/session<N>.log 2>/dev/null || echo "Waiting for node allocation..."

# Tail progress once log exists
tail -30 $WORKSPACE/session<N>.log

# Check completion marker
cat $WORKSPACE/offload_marker_session<N>.txt 2>/dev/null || echo "Not done yet"
```

Report every check:
- Current workload being processed (grep for `Workload #` in log)
- Any failures: FAILED, SKIP, CUDA OOM, Address already in use, Bus error
- Latest model completed (grep for `SUCCESS: results at`)

Stop polling when marker file contains `OFFLOAD_BENCHMARK_COMPLETE` or `OFFLOAD_BENCHMARK_FAILED`.

---

## Step 7 — Results Collection

When all sessions complete:

```bash
# Find all comparison CSVs
ls $REPO_ROOT/bench_compare_results_b200/*/comparison.csv

# Print each
for f in $REPO_ROOT/bench_compare_results_b200/*/comparison.csv; do
  echo "=== $f ==="
  cat "$f"
done
```

Combine into a single summary table and report:
- Output token throughput (AD vs PT per model per concurrency)
- Any models that FAILED or were SKIPPED (with reason)
- Total wall-clock time per session

---

## Failure Handling

| Symptom | Action |
|---------|--------|
| `cudaErrorMemoryAllocation` in kvcache | Mark model FAILED; sweep continues to next |
| `Address already in use` on port 8123 | Port-free wait is built into bench_compare.py; should self-heal. If recurring, check for zombie trtllm-serve processes: `pkill -f trtllm-serve` |
| `SKIP: model not found in model_registry` | AD registry missing entry; PT still runs. Note in report. |
| Session marker = `OFFLOAD_BENCHMARK_FAILED` | Tail the session log for root cause. Fix and resubmit that session only. |
| Slurm job queued indefinitely | Check `squeue -u $USER`; adjust `-t` or `-A` if needed |

---

## Quick Reference

```bash
# Check GPU count on GPU node (run via srun or after allocation)
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l

# Monitor Slurm job
squeue -u $USER

# Check auto-dev log (symlink, exists after job starts)
tail -f ~/.cache/auto-dev/slurm_logs/latest_ide_job.log

# Kill any zombie server from a failed session before resubmitting
pkill -9 -f trtllm-serve || true
```
