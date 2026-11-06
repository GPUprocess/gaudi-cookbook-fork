# DECISIONS.md — accepted and rejected technical decisions

| Decision | Status | Rationale | Evidence |
|---|---|---|---|
| Bind the draft head to the target's layer-45 modules by reference instead of loading a second copy | ACCEPTED | Zero marginal weight memory (no duplicate ~0.9 GiB/rank); the checkpoint ships the MTP block inside the target model | EXP-MTP-001; repo `mtp.py` |
| Run the draft eagerly (no dedicated HPU graph) | ACCEPTED (for now) | Shared storage between target and draft cannot be captured in a second full graph; at k=1 the eager cost is bounded and correctness-first. Revisit: draft HPU graph is the documented lever for k≥2 and concurrency (A014/A018) | EXP-MTP-002, EXP-MTP-003 |
| Default speculative depth k=1 | ACCEPTED | k=1 net-positive single-stream; k=2 net-neutral (eager per-step cost ≥ acceptance gain); concurrent net-negative by 16 streams. Revisit after the draft graph | EXP-MTP-002, EXP-MTP-003 |
| KDA candidate-slot state (S+1 columns) for verify rows | ACCEPTED | Required for correct post-verify state selection; algebra shared with the compact Mamba layout; churn/chaining parity tested | EXP-MTP-001; repo `kda_layer.py`, `tests/cpu/` |
| Per-verify-row KV block lists (`spec_verify_blocks`) | ACCEPTED | Correct output when verify rows straddle KV-block boundaries (measured at 640/1280 crossings) | EXP-MTP-001; repo patches |
| Prefix caching stays OFF | ACCEPTED (architectural) | KDA layers store no per-block states; the engine refuses to start with prefix caching enabled | Base port docs; engine behavior |
| KDA prefill chunk 64 (env-tunable) | ACCEPTED | chunk=128 measured worse TTFT (8K: 1.85 s vs 1.29 s; 16K: 3.59 s vs 2.63 s) — larger scan intermediates cost more than halved chunk count saves | EXP-MTP-004 |
| No quantization work (Q6/Q4/etc.) during the MTP phase | ACCEPTED (policy) | Project policy prohibits it in this phase; FP8 weights + BF16 states as published | `docs/BENCHMARK_POLICY.md`, master protocol |
| 4-HPU deployment deferred until the best 8-HPU implementation is established | ACCEPTED (policy) | Hardware roadmap makes 4-HPU a final phase; TP4 feasibility is measured (76.5 GiB/card weights, capture 128), full deployment config pending | EXP-OPT-003; `docs/HARDWARE_ROADMAP.md` |
| Graph capture 2048 at TP8 production | ACCEPTED | Meets the production window; the ~25.5 GiB ladder is the first reclaimable HBM block if a bigger KV pool or smaller TP is needed | EXP-OPT-002 |
| 6/5/3-HPU tensor-parallel layouts | REJECTED | 64 attention heads % {6,5,3} = 0; layouts mathematically invalid | EXP-OPT-003 |
| 1/2-HPU GLM deployment | REJECTED (hardware limit) | 328 GiB of weights cannot fit 96 GiB/card × {1,2} | EXP-OPT-003 |

## Rejected-for-now levers (documented, not dead)

- Draft HPU graph (split pre/post-attention cores) — the lever that would flip k=2 and
  high-concurrency MTP net-positive.
- Vocabulary-parallel argmax in `get_top_tokens` — draft-side optimization lever.
- `GLM5_NEXT_KDA_CHUNK` — knob retained for future tuning; 64 measured optimal.

## Rejected experiments

None failed outright so far; the k=2 performance result is recorded as REGRESSION-neutral
(kept as a capability, not the default) rather than FAILED.


| 200 tok/s single-stream decode for GLM-5.3-Flash on 8x Gaudi 2 | REJECTED (physical limit) |
| Strategic base for Phase 2 | PENDING OWNER DECISION | The abotsis fork (919ec98f) is a
  full GLM-5.3-Flash serving stack on the same hardware with measured capabilities our
  port lacks (the draft graphs, the MoE expert cache, the pad bucketing, the device
  sampling, the query tiling). Recommended: validate our current implementation
  (Stage-1), then migrate to their fork as the Phase-2 base, porting our docs/harness
  onto it. Their single-stream decode is slower (21-24 vs 30.1) - the migration trades
  per-stream speed for the richer optimization surface; measure before committing. | RESEARCH-001
  addenda 7-10 | Weight reads at bs=1 are 38.25 GiB/card/step = 41.1 GB. At the 2,450 GB/s spec BW the step floor is 16.8 ms (59.5 tok/s ceiling without MTP); at the bridge cost model's 1,945 GB/s it is 21.1 ms; at the ~69% measured utilization (~1,696 GB/s) it is 24.2 ms. The aggregate interpretation of the target is already MET (213-217 tok/s at 16 streams, no MTP). Single-stream 200 would require MTP-4 with acceptance >= 4.2 at 100% spec BW — above the measured utilization ceiling and beyond the checkpoint's single-head acceptance decay (reference MTP-4: 3.33 of 5). | HBM specs: GaudiGrimoire tier B (whitepaper 2.45 TB/s; bridge cost model 1,945 GB/s; measured utilization ~69%); our baseline EXP-OPT-001 |
