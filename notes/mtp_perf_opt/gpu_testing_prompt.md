# GPU Testing Prompt — MTP draft_len_schedule Validation
Branch: `gagam/super-mtp-perf-1`  
Date authored: 2026-04-27  
Target hardware: 4× GB200 (or equivalent)

---

## Context

This branch implements three layers of MTP (Multi-Token Prediction speculative decoding)
optimization for SuperV3 (Nemotron-3-Super-120B-A12B) on AutoDeploy:

### Layer 1 — SpecDecAwareScheduler (C1)
Wraps `SimpleScheduler` to defer incoming context requests when extend (spec-dec) requests are
in-flight, preventing CUDA graph cache misses from mixed prefill+extend batches. Starvation
guard: `max_defer_steps=4`. See `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`.

### Layer 2 — draft_len_schedule (NEW, this PR's main change)
Dynamically reduces the number of draft tokens at high concurrency to lower per-step cost.
At high batch sizes the GPU processes (batch_size × (max_draft_len+1)) tokens per MTP step,
which is 7× more than no-MTP — acceptance rate (~3.5 tok/step) can't cover the overhead.
Reducing max_draft_len at runtime (e.g., from 6 to 2 at bs=128) narrows this ratio.

**Files changed:**
- `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_eagle.py` — `EagleWrapper` now
  tracks `self.runtime_draft_len` (mutable); CUDA graph output buffers always sized at
  `max_draft_len+1` columns so graph shapes stay uniform across captures.
- `tensorrt_llm/_torch/auto_deploy/transform/library/compile_model.py` — sets
  `runtime_draft_len` and calls `set_capture_batch(batch_size, max_draft_len=resolved_dl)`
  per batch-size during CUDA graph capture, so each (bs, dl) pair gets its own graph.
- `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py` — resolves `runtime_draft_len` each
  step via `get_draft_len_for_batch_size()` and clips `py_draft_tokens` to that length before
  preparing the batched input.
- `examples/auto_deploy/model_registry/configs/super_v3_mtp.yaml` — adds `draft_len_schedule`
  and `cuda_graph_config.batch_sizes` for the new (bs, dl) capture pairs.

**Current schedule in `super_v3_mtp.yaml`:**
```yaml
draft_len_schedule:
  8: 6    # bs  1- 8: 6 drafts (7 tokens/seq)
  32: 5   # bs  9-32: 5 drafts (6 tokens/seq)
  64: 3   # bs 33-64: 3 drafts (4 tokens/seq)
  128: 2  # bs 65-128: 2 drafts (3 tokens/seq)
```

---

## Configurations to Benchmark (4-way sweep)

You must run all 4 from scratch — no existing measurements can be reused on this hardware.

| Label        | Config file                                    | Description                                   |
|--------------|------------------------------------------------|-----------------------------------------------|
| `no-MTP`     | `examples/auto_deploy/model_registry/configs/super_v3.yaml`     | Baseline, no speculative decoding             |
| `MTP-base`   | `examples/auto_deploy/model_registry/configs/super_v3_mtp.yaml` but with `draft_len_schedule` removed and original batch_sizes restored | MTP with max_draft_len=6, no scheduler fix |
| `MTP-C1`     | same yaml but without `draft_len_schedule` (keep SpecDecAwareScheduler) | MTP + SpecDecAwareScheduler only           |
| `MTP-C1-DLS` | `examples/auto_deploy/model_registry/configs/super_v3_mtp.yaml` as-is | MTP + SpecDecAwareScheduler + draft_len_schedule (full implementation) |

### How to create `MTP-base` and `MTP-C1` variants

The simplest approach: create temporary yaml copies for the test.

**`MTP-base`**: copy `super_v3_mtp.yaml`, remove the entire `draft_len_schedule` block, and
temporarily disable `SpecDecAwareScheduler` by checking `ad_executor.py` — the scheduler is
enabled when `spec_config is not None`. To disable C1 without code changes, you can instead just
document that `MTP-base` is theoretically what the config was before the SpecDecAwareScheduler PR
(commit `1afc84e439`). If testing on a checkout that includes C1, `MTP-base` ≈ `MTP-C1` for
most concurrencies except 32 (where C1 gives ~6.5% improvement). For a clean 4-way test, the
preferred approach is:
- `MTP-base`: checkout commit `3d22ea72a2` (before C1) + `super_v3_mtp.yaml` without draft_len_schedule
- `MTP-C1`: HEAD of this branch + `super_v3_mtp.yaml` without draft_len_schedule
- `MTP-C1-DLS`: HEAD of this branch + `super_v3_mtp.yaml` as-is

If re-checkout is too expensive, run just `no-MTP`, `MTP-C1`, and `MTP-C1-DLS` (3-way).
The C1 vs base delta is already documented in `notes/mtp_perf_opt/mtp_perf_findings.md`.

---

## Benchmark Parameters

```bash
MODEL=/path/to/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4   # adjust to local model path
CONCS="1 4 8 16 32 64 128"
ISL=1024
OSL=1024
TP=4
```

Use `sweep` (the AutoDeploy benchmark harness) or equivalent trtllm-bench commands.
Example for each configuration:

```bash
# no-MTP
sweep \
    --model "$MODEL" \
    --config-path examples/auto_deploy/model_registry/configs/super_v3.yaml \
    --server-type trtllm-autodeploy \
    --world-size $TP \
    --concurrencies "$CONCS" \
    --isl $ISL --osl $OSL \
    --tag gb200_nomtp_1k1k_tp4

# MTP-C1 (no draft_len_schedule)
# Create super_v3_mtp_c1only.yaml by copying super_v3_mtp.yaml and removing draft_len_schedule block
sweep \
    --model "$MODEL" \
    --config-path examples/auto_deploy/model_registry/configs/super_v3_mtp_c1only.yaml \
    --server-type trtllm-autodeploy \
    --world-size $TP \
    --concurrencies "$CONCS" \
    --isl $ISL --osl $OSL \
    --tag gb200_mtp_c1_1k1k_tp4

# MTP-C1-DLS (full implementation, as-is yaml)
sweep \
    --model "$MODEL" \
    --config-path examples/auto_deploy/model_registry/configs/super_v3_mtp.yaml \
    --server-type trtllm-autodeploy \
    --world-size $TP \
    --concurrencies "$CONCS" \
    --isl $ISL --osl $OSL \
    --tag gb200_mtp_c1_dls_1k1k_tp4
```

---

## What to Measure and Report

For each configuration and each concurrency level, collect:

1. **ITL** (Inter-Token Latency) — primary latency metric, milliseconds
2. **Output TPS** (tokens per second) — primary throughput metric  
3. **TTFT** (Time to First Token) — important for MTP configs with C1 (deferral cost)
4. **p50 / p99 ITL** — to distinguish mean vs tail regression

### Expected outcomes (theoretical, not validated on GB200)

MTP step-time ratio scales roughly with token count per step:
- `runtime_dl=6`: 7 tok/seq per step → 7× token overhead vs no-MTP
- `runtime_dl=2`: 3 tok/seq per step → 3× token overhead vs no-MTP

At conc=128 with `draft_len_schedule`:
- `MTP-C1` (dl=6): ITL ≈ 2.1× no-MTP (measured ~35ms vs ~16ms on B200 — GB200 numbers will differ)
- `MTP-C1-DLS` (dl=2 at bs=128): step-time ratio ~3/7 × original, acceptance ~2.5 tok/step
  → expected ITL ≈ 35 × (3/7) / 2.5 × 3 ≈ 18ms vs no-MTP ~16ms (much closer)

At conc=64 with `draft_len_schedule`:
- `MTP-C1` (dl=6): ITL ≈ 1.64× no-MTP (on B200)
- `MTP-C1-DLS` (dl=3 at bs=64): step-time ratio ~4/7 × original
  → expected ITL improvement of ~30% at conc=64

At conc=1–8: `draft_len_schedule` uses dl=6 (same as before), so no change expected vs `MTP-C1`.

---

## Key Questions to Answer

1. Does `MTP-C1-DLS` reduce ITL at conc=64 and conc=128 vs `MTP-C1`?
2. At which concurrency does MTP-C1-DLS first become competitive with no-MTP?
3. Does `draft_len_schedule` hurt ITL at conc=1–8 (it shouldn't — dl stays at 6)?
4. What is TTFT cost of C1 on GB200? (On B200: +31% avg, +14% p99 at conc=32)
5. Is there a throughput (TPS) regime where MTP-C1-DLS beats no-MTP at higher conc?

---

## Next Steps Based on Results

**If MTP-C1-DLS is within 1.2× no-MTP at conc=64:**
→ The schedule is effective. Consider a more aggressive schedule (dl=1 at bs=128).

**If MTP-C1-DLS still loses significantly at conc=128:**
→ Fundamental arithmetic limit. Document results and move to Chunked Prefill + MTP (C5 in
the findings doc) as the next structural fix.

**If there are any correctness issues (different output tokens, crashes):**
→ Check `EagleWrapper._forward_with_kv_cache` in `modeling_eagle.py`. The critical invariant
is that `new_tokens_2d` always has shape `[num_sequences, max_draft_len+1]` regardless of
`runtime_draft_len`, and `output_ids_target` is built from `new_tokens_2d_extend.flatten()`
(NOT `new_tokens_2d.flatten()` — that would cause a shape mismatch with input_ids).

---

## Reference Files (in this notes directory)

- `mtp_perf_findings.md` — Full findings from B200 investigation: root cause analysis,
  SpecDecAwareScheduler mechanism, benchmark results across conc=1–128, candidate analysis.
- `pt_backend_mtp_investigation.md` — Comparison of PyTorch backend vs AutoDeploy MTP
  architecture; `draft_len_schedule` design rationale; corrected step-time projections.
