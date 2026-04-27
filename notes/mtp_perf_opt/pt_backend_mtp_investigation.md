# PyTorch Backend MTP Investigation
Date: 2026-04-27
Goal: Find reusable patterns from PyTorch backend MTP for AutoDeploy

---

## Architecture Comparison

| Aspect | PyTorch Backend | AutoDeploy |
|--------|-----------------|------------|
| MTP entry point | `MTPWorker` / `Eagle3OneModelWorker` post-target-model step | `EagleWrapper` (combined model, inside CUDA graph) |
| CUDA graph scope | Target model only; MTP accept/draft is OUTSIDE the graph | Entire EagleWrapper including draft loop is inside graph |
| Mixed batch handling | MTP disabled per-step when draft model shouldn't run; context tokens handled naturally via `attn_metadata.num_contexts` | CUDA graph shape mismatch → eager mode; fixed by `SpecDecAwareScheduler` |
| Draft length | Dynamic via `draft_len_schedule` dict (runtime tuning) | Static `num_nextn_predict_layers` in config |
| Acceptance kernel | `torch.ops.trtllm.mtp_sampling_and_accepted_draft_tokens_op` (fused CUDA kernel) | Python `mask_same.cumprod()` (GPU-vectorized but unfused) |
| Acceptance modes | Strict greedy + Relaxed topk (for thinking models) | Strict greedy only |
| Speculation gate | `SpeculationGate` rolling-window disable | None |
| Mamba cache | `MambaHybridCacheManager.update_mamba_states()` (shared class) | Same class |

---

## Key Insight: Why AD Needs SpecDecAwareScheduler but PT Backend Doesn't

The PT backend's MTP lives OUTSIDE the main model's CUDA graph. After the target model runs on
all tokens (context + generation), the MTP worker does:
  1. `sample_and_accept_draft_tokens()` → accepts/rejects, runs via fused CUDA kernel
  2. `update_mtp_hidden_states()` → updates rolling hidden state window
  3. `prepare_drafter_inputs()` → shapes inputs for draft model forward
  4. Draft model forward (separate from target model)

Because steps 1-4 are outside the graph, mixed batches just mean `num_contexts > 0` in
attention metadata — the shapes of the generation portion remain fixed.

AD bakes the entire target + draft loop into `EagleWrapper.forward()` inside the CUDA graph.
Mixed batches change the shape of inputs (e.g., 1000 prefill tokens + 31×7 extend tokens),
causing a CUDA graph shape mismatch → eager mode fallback → 5x slower step.

→ **This explains both why we need SpecDecAwareScheduler in AD and why the PT backend doesn't.**

---

## Reusable Components (Ranked by Impact)

### 1. `draft_len_schedule` — HIGH IMPACT at conc=64/128

**PT backend:** `tensorrt_llm/_torch/speculative/drafter.py` + `py_executor.py:_handle_dynamic_draft_len()`

Config syntax (from deepseek-r1-deepgemm.yaml style):
```yaml
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 6
  draft_len_schedule:
    1: 6     # batch_size 1-31: use 6 draft tokens
    32: 5    # batch_size 32-63: use 5 draft tokens
    64: 4    # batch_size 64-127: use 4 draft tokens
    128: 3   # batch_size 128+: use 3 draft tokens
```

**How it works:** Binary search on sorted dict keys → maps batch_size to draft_len.
Each step, requests' `py_draft_tokens` is padded/truncated to the resolved draft_len.
`runtime_draft_len` propagates to the model engine for acceptance logic.

**AD impact (updated with actual sweep data 2026-04-27):**

Actual measured at conc=128, max_draft_len=6:
- MTP+C1 ITL: 34.94ms (est. step_time ≈ 34.94 × 3.5 ≈ 122ms)
- no-MTP ITL: 16.34ms (measured)  ← 2.5× faster than the 41ms assumed below

At conc=128, max_draft_len=3 (estimated):
- Step tokens: 128 × 4 = 512 (vs 128 × 7 = 896)
- Step time ≈ 122 × (4/7) ≈ 70ms, acceptance ≈ 3.0 → ITL ≈ 23ms
- vs actual no-MTP ITL 16.34ms → draft_len_schedule still 41% worse on ITL

**Conclusion (corrected):** The original 41ms no-MTP assumption was wrong — actual is 16.34ms.
draft_len_schedule reduces the gap at conc=128 from 2.14x to ~1.4x worse, but MTP cannot
beat no-MTP on ITL at conc=128 regardless of draft_len. Still worth implementing for conc=32-64
range where the gap is smaller and draft reduction can meaningfully help.

**AD implementation challenge:** AD's CUDA graph is captured for `batch_size × (max_draft_len+1)`
shapes. Dynamic draft_len means new shapes at runtime. Two options:
  a) Capture multiple graphs: `batch_sizes × draft_len_combos` (complex, 2-3 weeks work)
  b) Use static max_draft_len for CUDA graph but truncate accepted tokens (simpler but wasteful)
  c) Change batch_sizes config to pre-specify (batch_size, draft_len) pairs explicitly

Option c is feasible: for each (batch_size, draft_len) in the schedule, add
`batch_size × (draft_len+1)` as an explicit shape in the graph capture list.

**Status:** NOT yet implemented in AD. Recommended for follow-up PR.

---

### 2. `SpeculationGate` — MEDIUM, monitoring-only for AD

**PT backend:** `tensorrt_llm/_torch/speculative/speculation_gate.py`

```python
SpeculationGate(window=100, threshold=0.3)
gate.record_avg_decoded(avg_decoded_tokens_per_iter, request_id)
# → returns (disabled_now, current_avg_accept)
# → permanently sets self.disabled = True when rolling avg < threshold
```

After each completed request, the rolling average of `(avg_decoded_tokens_per_iter - 1.0)`
is tracked. If it falls below `threshold` over the last `window` requests, speculation is
permanently disabled for the server lifetime.

**PT executor wires this up in `_on_request_done()`:**
- `avg_decoded_tokens_per_iter` = (num_output_tokens) / (num_decode_steps)
- Calls `gate.record_avg_decoded()` on request completion

**AD limitation:** Disabling MTP in AD requires re-initializing `EagleWrapper` with a
different model (the non-speculative target model). This is NOT feasible at runtime.

**Recommended AD adaptation:** Implement monitoring-only — log a WARNING when rolling
acceptance average falls below 1.5 tokens/step (below this, MTP ITL > no-MTP ITL at conc≤8).
This helps operators decide whether to restart without MTP.

**Implementation sketch:**
```python
# In ADExecutor._complete_request() or sampler post-processing:
if new_tokens_lens is not None:  # from EagleWrapper output
    avg_accepted = new_tokens_lens.float().mean().item()
    self._speculation_gate.record_avg_decoded(avg_accepted)
    if self._speculation_gate.is_poor():
        logger.warning(f"MTP acceptance rate low: {avg_accepted:.2f} tokens/step. Consider restarting without MTP.")
```

---

### 3. `mtp_sampling_and_accepted_draft_tokens_op` — LOW, correctness parity only

**PT backend:** `torch.ops.trtllm.mtp_sampling_and_accepted_draft_tokens_op(logits, draft_tokens, ...)`

A fused CUDA kernel that combines sampling and acceptance in one pass. Faster than
AD's Python `mask_same.cumprod(dim=1).sum(dim=1) + 1` for large batches.

**AD current:** `modeling_eagle.py:1023-1026` — GPU-vectorized but unfused:
```python
mask_same = new_tokens_2d_extend[:, :-1] == input_ids_extend[:, 1:]
new_tokens_lens_extend = mask_same.cumprod(dim=1).sum(dim=1, dtype=torch.int32) + 1
```

**Assessment:** At batch_size=32-128 and max_draft_len=6, the `cumprod+sum` is fast
(~0.1ms). The CUDA kernel would be marginally faster but not meaningfully so.
Both are inside the CUDA graph and captured once. **Not worth porting for now.**

---

### 4. Relaxed acceptance for thinking models — FUTURE only

**PT backend:** `mtp.py:sample_and_accept_draft_tokens()` with `use_relaxed_acceptance_for_thinking=True`

Uses `torch.ops.trtllm.mtp_relaxed_acceptance_op` with topk window and delta threshold.
Per-request `relaxed_delta` stored in `mtp_relaxed_delta_pool`. Tied to
`begin_thinking_phase_token` / `end_thinking_phase_token` to only apply during think phase.

**AD current:** `MTPDecodingConfig` has `use_relaxed_acceptance_for_thinking`, `relaxed_topk`,
`relaxed_delta` fields already — but AD's `EagleWrapper` doesn't implement them yet.

**Recommendation:** Port the relaxed acceptance from PT backend when AD needs to support
reasoning models (DeepSeek-R1 with AD + MTP). Medium effort, low risk.

---

## Summary Table

| Component | PT Backend File | AD Relevance | Effort | Recommendation |
|-----------|-----------------|--------------|--------|----------------|
| `draft_len_schedule` | `drafter.py`, `py_executor.py` | HIGH - 25% ITL improvement at conc=128 | Medium-High (graph capture changes) | **Implement in follow-up** |
| `SpeculationGate` (monitoring) | `speculation_gate.py` | MEDIUM - operational visibility | Low (monitoring only) | **Implement lightweight version** |
| `mtp_sampling_and_accepted_draft_tokens_op` | `mtp.py` | LOW - marginal speedup | Low but needs kernel access | Skip for now |
| Relaxed acceptance | `mtp.py` | FUTURE - reasoning models | Medium | Implement when needed |
| Architecture: MTP outside graph | `py_executor.py` | INFORMATIONAL - explains our SpecDecAwareScheduler need | N/A | Note for future AD refactor |

---

## Next Steps

1. **Await sweep results** (conc=1,4,8,32,64,128) to quantify MTP vs no-MTP gap at conc=64/128
2. **If conc=64/128 gap is large:** implement `draft_len_schedule` for AD as follow-up PR
3. **Add lightweight SpeculationGate monitoring** to `ad_executor.py` (1-2 hour task)
4. **Create `super_v3_mtp_dropin.yaml`** adding draft_len_schedule once implemented

---

## Notes on PT Backend Config for Nemotron SuperV3

To run PT backend with MTP, add to `nemotron-3-super-throughput.yaml`:
```yaml
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 6  # matches model checkpoint
  mtp_eagle_one_model: true
  # draft_len_schedule: {1: 6, 32: 5, 64: 4, 128: 3}  # optional
```

The `deepseek-r1-deepgemm.yaml` uses `num_nextn_predict_layers: 1` (1 MTP layer).
SuperV3 has 6 MTP layers. No other changes needed — PT backend auto-discovers model type.
