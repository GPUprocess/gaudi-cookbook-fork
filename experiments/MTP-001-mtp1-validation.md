# MTP-001 — Native MTP=1 speculative decoding: implementation and validation

**Category:** MTP / speculative decoding
**Date:** MTP phase (see Git history of branch `glm53-flash`)
**Author/agent:** project agent
**Git commit:** `e3c6f52` (implementation spread across the branch's MTP commits)
**Status:** BASELINE CANDIDATE — functionality VERIFIED; performance PROVISIONAL (Stage-1
controlled reproduction pending, task A001)

## 1. Problem

Batch-1 decode on 8× Gaudi 2 is latency/sync-bound (~30 tok/s, AIP 0–24%): the target model
runs one token per step and the hardware is underutilized. Multi-token prediction (the
checkpoint ships its own draft head) should raise per-step token yield without a second model.

## 2. Hypothesis

Binding the checkpoint's layer-45 MTP head to vllm-gaudi's speculative decoder — with the KDA
layers keeping candidate-slot state for verify rows — should raise batch-1 decode throughput by
the acceptance rate (expected ~1.7× token yield at k=1) while preserving correctness.

## 3. Previous research

- The base port write-up and gates (cookbook port; vllm-gaudi v0.26.0).
- The glm-5-mxfp4 port's speculative-decoding patches for vllm-gaudi (padded verify layout).
- Upstream GLM MTP design (single MTP block inside the target checkpoint; `eh_proj`; shared
  head), NVIDIA/AMD speculative-decoding references for semantics.
- GaudiGrimoire MCP: consulted for HPU graph capture constraints (shared-storage capture
  limits → eager draft decision) and lazy-mode dispatch behavior (AccThread busy-wait finding).
  Useful outcomes recorded here and in DECISIONS.md.

## 4. Previous attempts

None for this model — vllm-gaudi had no speculative path for it before this work. The
glm-5-mxfp4 port's spec patches were the prior art reused here (drift-checked against this
release and re-recorded as this port's own four patches).

## 5. Experimental design

- Control: same window/config without `--speculative-config` (non-MTP).
- Treatment: `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.
- Hardware: 8× Gaudi 2, TP8 + expert parallel (both conditions).
- Window: 32K context, 8 seqs (MTP window). NOTE: the originally reported +21% compared
  against the production window (262K/32seq) — non-equivalent; see §7.
- Prompts: natural-prompt sets (golden reference, feature gates); needle documents to 16K.
- Generation: greedy (correctness) + sampled (feature gates); warm-up per shape; repetitions
  as documented per table (Stage-1 will re-run with ≥5 reps per condition).
- Metrics: decode tok/s (bs=1 and concurrent), acceptance length, TTFT, HBM, correctness
  gates.

## 6. Implementation

- `ports/glm-5-3-flash/glm5_next_hpu/glm5_next_hpu/mtp.py`: draft wrapper bound to the
  target's layer-45 modules by reference; name normalization; completeness gate.
- `kda_layer.py`/`kda.py`: S+1 candidate-slot recurrent state; post-verify state selection.
- Four recorded vllm-gaudi patches (`mtp_attn`, `mtp_eagle`, `mtp_runner`, `mtp_sampler`):
  verify-row metadata, draft chaining, padded verify layouts, per-verify-row KV block lists.
- `serving/serve_mtp.sh`: launch configuration.
- Tests: `tests/cpu/check_*.py` (4) and `tests/hpu/mtp/` (harness + functional checks).

## 7. Results

- Acceptance length 1.73–1.98 of 2.0 (typical ≈1.8); live deployment 1.84–1.95.
- Decode bs=1: 35.7 tok/s like-for-like single run (+14% vs 31.4 control, same window);
  36.8 tok/s in the original window comparison (+21% vs a 30.3 production-window control —
  non-equivalent conditions; PROVISIONAL).
- Decode bs=8: 36.5 tok/s aggregate-equivalent single-stream figure; 4K-prompt + 512 gen:
  37.6 tok/s.
- 16-stream aggregate: 42.7 tok/s vs 59.4 non-MTP (net-negative; eager draft crossover).
- TTFT (warm, 64 gen tokens): 2.1 s @2K, 3.1 s @8K, 4.2 s @16K.
- HBM under load: 64.1 GiB/card (weights 38.6 + caches + candidate columns + graphs).
- Memory overhead: zero marginal weights; ~0.8 GiB/rank extra MLA latent cache + ~0.3 GiB/rank
  KDA candidate columns.

## 8. Correctness and accuracy

- Correctness: PASS (golden near-tie analysis, feature gates, block boundaries, 16K needles,
  2034-token generation, disconnect, soak, multi-turn, penalties, sampling, structured output,
  logprobs). See `ACCURACY.md`.
- Accuracy gate (0.5 pp): NOT YET RUN — the formal suite does not exist yet; result capped at
  BASELINE CANDIDATE.

## 9. Interpretation

The hypothesis is supported functionally (acceptance ≈1.8 of 2.0; correct output) and
directionally on performance (+14% like-for-like single run), but the measured gain is below
the naive acceptance-implied ~1.8× because the draft runs eagerly and the verify rows cost
more than a single-row step. The original +21% figure overstates the like-for-like gain
because the control used a different window; it is preserved as a historical observation.

## 10. Decision

Keep MTP k=1 deployed. Re-run as Stage-1 controlled reproduction (A001) with ≥5 repetitions
per condition and alternating order before promoting any figure to VERIFIED. The draft HPU
graph is the next lever (A014/A018).

## 11. Reproducibility

Sanitized: `ports/glm-5-3-flash/serving/serve_mtp.sh` (launch), the four patches applied in
order (see the fork README), `tests/hpu/mtp/mtp_validate.py --base http://127.0.0.1:8000` for
the functional harness, `tests/cpu/` for the unit checks. Requires 8× Gaudi 2, Gaudi SW 1.24.1,
the port image, and the four patches. Raw per-run outputs from the validation window are in
the operator's local working notes (not published); Stage-1 raw runs will be stored under
`benchmarks/`.
