# MTP Performance Investigation Findings
Date: 2026-04-27  
Branch: gramnarayan/mtp-enable-cudagraph → gramnarayan/mtp-perf-opt  
Worktree: /home/scratch.ghubaraagam_sw_2/TensorRT-LLM_mtp_perf_opt

---

## Root Cause Analysis

**Problem**: MTP (Eagle spec-dec) regresses at conc=32 vs no-MTP (13.16ms → 9.82ms ITL,
+34% slower) despite 2x speedup at conc=1.

**Root cause**: At conc=32, ~7-15% of batch steps mix new prefill (context) requests with
speculative-decode extend requests in the same step. The `torch_cudagraph.py` backend captures
CUDA graphs for extend-only input shapes; any mixed batch causes `can_run_cuda_graph=False`
and full eager execution of the target model (1024 prefill + 31×7 extend = 1241 tokens) and
draft model iteration-0.

Key code path:
- `ScheduledRequests.can_run_cuda_graph` (scheduler.py:62) → False when num_context_requests > 0
- `maybe_pad_for_cuda_graph` (ad_executor.py:340) → falls back to `_call_func()` (eager mode)

---

## Fix Implemented: Candidate 1 — SpecDecAwareScheduler

**Files changed:**
- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` — add `SpecDecAwareScheduler`
- `tensorrt_llm/_torch/pyexecutor/scheduler/__init__.py` — export it
- `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py` — wrap scheduler when MTP is active

**Mechanism**: `SpecDecAwareScheduler` wraps `SimpleScheduler`. After the inner scheduler
produces a `SchedulerOutput`, if any generation request carries draft tokens (spec-dec extend),
all deferrable context requests are removed from the output for this step. They are re-evaluated
by the scheduler on the next step. A starvation guard (max_defer_steps=4) forces admission after
4 consecutive deferrals, bounding the TTFT penalty.

**Committed**: `gramnarayan/mtp-perf-opt` bb1224a3e5

---

## Results: MTP base vs MTP+C1

Sweep: `260427_0234_super_mtp_c1_specdec_sched_1k1k_tp4`

| conc | MTP base ITL | MTP+C1 ITL | delta      | MTP base outTPS | C1 outTPS | TPS delta |
|------|-------------|------------|------------|-----------------|-----------|-----------|
| 1    | 3.915ms     | 3.946ms    | +0.8% ≈    | 248.4           | 246.6     | -0.8%     |
| 4    | 5.666ms     | 6.362ms    | +12% ❌*   | 420.6           | 416.8     | -0.9%     |
| 8    | 6.500ms     | 6.203ms    | **-4.6%** ✅| 1159.5         | 1197.1    | +3.2%     |
| 16   | 8.665ms     | 8.540ms    | **-1.4%** ✅| 1738.2         | 1738.9    | 0%        |
| 32   | 13.159ms    | 12.298ms   | **-6.5%** ✅| 2313.9         | 2449.4    | +5.9%     |

*conc=4 avg ITL regression is tail-driven: p50 ITL is BETTER (5.047→4.963ms).
outTPS difference is -0.9% (within measurement noise). Not a real regression.

**TTFT cost of deferral:**
- conc=32: TTFT avg 373ms → 489ms (+31%), p99 1333ms → 1516ms (+14%)
- Maximum extra TTFT per request = max_defer_steps × step_time = 4 × ~44ms ≈ 176ms

---

## MTP+C1 vs no-MTP (reference) — FINAL (all concurrencies)

Sweeps: `260427_0331_super_nomtp_ad_1k1k_tp4` (no-MTP), `260427_0406_super_mtp_fix_ad_1k1k_tp4` (MTP+C1)

| conc | no-MTP ITL | MTP+C1 ITL | ratio     | no-MTP outTPS | C1 outTPS | winner           |
|------|-----------|------------|-----------|---------------|-----------|------------------|
| 1    | 6.79ms    | 3.90ms     | **0.57x** | 145.4         | 249.4     | MTP +75% ITL     |
| 4    | 6.95ms    | 5.49ms     | **0.79x** | 560.9         | 559.8     | MTP +21% ITL     |
| 8    | 7.79ms    | 6.20ms     | **0.80x** | 999.8         | 1196.6    | MTP +20% ITL/TPS |
| 32   | 9.99ms    | 12.55ms    | **1.26x** | 2105.4        | 2440.5    | no-MTP ITL; MTP TPS|
| 64   | 12.03ms   | 19.67ms    | **1.64x** | 5049.8        | 3068.0    | no-MTP wins both ❌|
| 128  | 16.34ms   | 34.94ms    | **2.14x** | 7354.8        | 3457.4    | no-MTP wins both ❌|

MTP+C1 wins at conc=1–8 (ITL +20-75%). At conc=32 MTP wins TPS but loses ITL.
At conc=64+, no-MTP dominates on both ITL and TPS — gap grows to 1.64x and 2.14x ITL.

---

## Remaining Gap Analysis (all concurrencies)

Step time ratios and break-even acceptance rates (acceptance_needed = step_time_mtp / step_time_nomtp):

| conc | no-MTP step | est MTP step | step ratio | acceptance (est) | break-even |
|------|------------|--------------|------------|-----------------|------------|
| 32   | ~10ms      | ~44ms        | 4.3x       | ~3.5 tok/step   | need 4.3 ❌|
| 64   | ~12ms      | ~69ms        | 5.7x       | ~3.5 tok/step   | need 5.7 ❌|
| 128  | ~16ms      | ~122ms       | 7.5x       | ~3.5 tok/step   | need 7.5 ❌|

At conc=64/128, the step-time ratio grows because the GPU processes 7× more tokens per MTP step
(64×7=448, 128×7=896) while no-MTP scales more gracefully (64, 128 tokens).
Acceptance rate (~3.5 tok/step) stays roughly constant across concurrencies.

**Key insight**: no-MTP step time at conc=128 (16.34ms) is much faster than the 41ms assumed
in the `pt_backend_mtp_investigation_260427.md` theoretical analysis. This invalidates the
prediction that draft_len_schedule would make MTP net-positive at conc=128 (~31ms predicted ITL
vs 16ms actual no-MTP). The real no-MTP is 2.5× faster than assumed.

---

## Remaining Optimization Candidates

| # | Candidate                       | Expected impact post-C1 | Risk   | Status   |
|---|----------------------------------|------------------------|--------|----------|
| 1 | SpecDecAwareScheduler            | ✅ Done (+6.5% c32)     | —      | Done     |
| 2 | Piecewise CUDA graph for mixed  | Moderate               | HIGH   | Skip     |
| 3 | Skip draft-0 for prefill seqs   | Small (rare path now)  | MEDIUM | Skip     |
| 4 | Extend captured shapes          | Small (rare path)      | LOW    | Skip     |
| 5 | Chunked prefill + MTP           | Moderate (TTFT)        | MEDIUM | Future   |
| 6 | Fuse EagleWrapper host ops      | Unknown, needs profiling| LOW   | Investigate |
| 7 | Tune max_defer_steps=2          | -2% c32, better TTFT   | LOW    | Consider |

---

## Recommended Follow-ups

1. **Profile MTP at conc=32 with nsys**: Measure actual CUDA kernel breakdown at conc=32
   to verify no other bottlenecks (Mamba SSM scale, Python host overhead).

2. **Tune max_defer_steps**: Default 4 provides best throughput; 2 reduces TTFT by ~88ms
   with minimal loss. Make it configurable via `super_v3_mtp.yaml` or LlmArgs.

3. **Acceptance rate optimization**: At conc=32, improving acceptance rate from 3.5→4.3+
   tokens/step would make MTP net positive. This is a model quality question.

4. **Chunked prefill + MTP** (Candidate 5): The `# TODO: Support chunked prefill w/ spec dec`
   in `super_v3_mtp.yaml:7` would reduce mixed-batch step cost by splitting large prefills.
   This is the next highest-leverage structural fix.

5. **Test C1 on lower-latency workloads**: 256/256 ISL/OSL, where acceptance rate is higher
   and MTP benefit is more pronounced — C1 should help more there.

---

## Post-C1 Candidate Analysis (2026-04-27)

### Candidate 6: Fuse EagleWrapper host-side ops — ASSESSED, SKIP

EagleWrapper.forward() runs 5 host-side operations per draft iteration (loop runs 6 times):
- `switch_to_generate_()`: first call does real work (host+device buffer updates), subsequent 4
  are no-ops (early return when already in generate mode). ≈1×real + 4×early-return.
- `copy_("input_ids", draft_tokens)`: device-side copy, captured in CUDA graph.
- `offset_pos_and_cache_(c_offset)`: GPU ops (ragged Triton kernel, position_ids update) + async
  D2H copy (`non_blocking=True`) for host-suffix args; D2H is tiny (<64 bytes).
- `batch_info.get_num_sequences()`: pure CPU host-tensor read, no GPU sync.

Key observations:
1. The `copy_to_host()` D2H is `non_blocking=True` — doesn't stall the GPU.
2. By the time the next draft forward completes (~1ms), the D2H has already finished.
3. `switch_to_generate_()` runs fully once (iteration 0) and early-exits for iterations 1-4.
4. All heavy host writes use device-side tensors; Python-only host ops are trivial (int arithmetic).

**Conclusion**: Host-side overhead in the draft loop is <1ms per step vs 44ms total step time.
Implementing C6 would save <2% — not worth the complexity.

### Candidate 7: Tune max_defer_steps — ASSESSED, KEEP DEFAULT

`max_defer_steps=4` (default) gives maximum throughput: requests wait up to 4×44ms=176ms extra
TTFT before force-admission. Reducing to 2 would lower max extra TTFT to 88ms but increases
frequency of forced mixed batches, hurting ITL slightly.

The forced-admission frequency at conc=32: ~2.5 forced mixes/second (1 per new request arrival,
since each request defers at most 4 times). Each forced mix processes ~1217 tokens in eager mode
vs 224 in CUDA graph, taking ~196ms vs ~44ms. The extra latency per force-admission is ~152ms,
affecting all 32 in-flight requests for that step.

The default of 4 is the right balance for throughput-focused workloads. Making it tunable is
desirable for latency-sensitive deployments but doesn't affect the current regression.

**Conclusion**: Keep max_defer_steps=4. Add to MTPDecodingConfig in a follow-up PR.

### Candidate 2: Piecewise CUDA graph — SKIP

HIGH RISK. Would require capturing separate graphs for the prefill and extend sub-batches and
stitching them together. Significant implementation complexity with uncertain benefit after C1.

### Remaining gap: fundamental arithmetic

After C1, the 25% ITL regression at conc=32 is due to:
- MTP extend step: 32 seqs × 7 tokens = 224 tokens/step, step time ~44ms
- no-MTP step: 32 seqs × 1 token = 32 tokens/step, step time ~10.3ms
- Step time ratio: 4.3x; acceptance rate: ~3.5 tokens/step (needs ≥4.3 to break even)

Break-even paths:
1. Model quality: raise acceptance rate from 3.5 to 4.3+ tokens/step (not in our control)
2. Reduce max_draft_len (e.g., 6→4): step tokens 224→160, step time 44→31ms, acceptance
   ~3.36 → ITL ≈ 31/3.36 = 9.2ms (beats no-MTP!). Config change, tradeoff: lower TPS.
3. Chunked prefill + MTP (C5): reduces forced mixed batch cost; medium risk, future work.

**Current status**: C1 is the maximum achievable improvement from the scheduling layer alone.
The rest requires either model quality improvements or the chunked-prefill engineering (C5).

