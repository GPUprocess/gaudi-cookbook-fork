# EXPERIMENTS.md — experiment index

One row per experiment. Full reports live in `experiments/`; the published technical write-up
for the MTP work is `docs/ports/glm-5-3-flash-mtp.md`. Status labels follow
`docs/BENCHMARK_POLICY.md` (PROVISIONAL / BASELINE CANDIDATE / VERIFIED / INCONCLUSIVE /
FAILED / REGRESSION).

Pre-MTP experiments (OPT-*) were performed during the deployment/optimization phase that
preceded the MTP work; their raw artifacts and phase reports are preserved in the operator's
local working notes and are summarized here. All figures below are measured values, reported
with their conditions; none has yet passed the full `docs/BENCHMARK_POLICY.md` gate, so the
authoritative verified figures in `BASELINE.md` remain unfilled until Stage-1 reproduction
completes.

| ID | Date | Category | Hypothesis | Commit/config | HPUs | Key result | Status | Report |
|---|---|---|---|---|---|---|---|---|
| OPT-001 | pre-MTP phase | Baseline | Capture the production TP8+EP baseline | cookbook port, vllm-gaudi v0.26.0 (b059f9c), TP8+EP, GMU 0.82, KV override 1280 blocks/card, capture 2048, 256K ctx, 32 seqs | 8 | Decode 30.1/29.8 tok/s bs=1 (2 runs); 125.1 agg @8 streams; 213.2 agg @16; prefill 4,928 tok/s @128K prompt; HBM 65.5/68.2 GiB/card | BASELINE CANDIDATE | experiments/OPT-001-production-baseline.md |
| OPT-002 | pre-MTP phase | Memory | Decompose per-card HBM to find reclaimable blocks | same as OPT-001 | 8 | Weights 38.2 GiB/card; graphs/workspace ~25.5 GiB; KV ~1.8 GiB; GMU 0.95 unreachable at capture 2048 | VERIFIED (measurement) | experiments/OPT-002-hbm-decomposition.md |
| OPT-003 | pre-MTP phase | HPU scaling | TP4+EP is the minimum viable 4-HPU footprint | TP4+EP, GMU 0.95, capture 128 | 4 | Weights 76.5 GiB/card; viable with capture 128 (2048 ladder OOMs); 64 heads % {6,5,3} = 0 rules out 6/5/3-HPU TP | VERIFIED (feasibility) | experiments/OPT-003-tp4-feasibility.md |
| OPT-004 | pre-MTP phase | Stability | Post-experiment rollback restores production performance | fresh TP8 boot, production config | 8 | bs=1 30.3/30.1; 16 streams 217.2 agg; TTFT @128K 25.4 s — matches OPT-001 within noise | VERIFIED (reproduction) | experiments/OPT-004-rollback-reverification.md |
| MTP-001 | MTP phase | MTP | Native MTP=1 (checkpoint's own layer-45 head, bound by reference) improves batch-1 decode without correctness loss | fork branch glm53-flash; k=1; TP8+EP; 32K ctx; 8 seqs; window config | 8 | Acceptance 1.73–1.98 of 2.0; bs=1 35.7–36.8 tok/s (+14% like-for-like single run; +21% vs production-window control — NON-equivalent conditions, PROVISIONAL); 16-stream aggregate 42.7 vs 59.4 (net-negative, eager draft); memory 64.1 GiB/card; full correctness/feature gate suite passed | BASELINE CANDIDATE (functionality VERIFIED; performance PROVISIONAL pending Stage-1 reproduction) | experiments/MTP-001-mtp1-validation.md |
| MTP-002 | MTP phase | MTP depth | Depth 2 chains the single MTP head and improves throughput further | k=2; S=4 candidate columns; 3-row verify | 8 | Acceptance 2.04–2.52 of 3.0; decode 30.2–34.1 tok/s bs=1 (net-neutral/worse than k=1); correct across full sweep; golden divergences at same positions as k=1 | VERIFIED (functionality); REGRESSION-neutral (performance) | experiments/MTP-002-mtp2-sweep.md |
| MTP-003 | MTP phase | Concurrency | MTP benefit survives concurrent load | 16-stream like-for-like 32K/8seq, k=0/1/2 | 8 | Aggregate: k=0 59.4, k=1 42.7, k=2 41.8 tok/s; single-stream within run: k=0 31.4, k=1 35.7; acceptance holds under load (1.72–1.87 k=1) | VERIFIED (measurement) | experiments/MTP-003-concurrent-k-sweep.md |
| RESEARCH-002 | KB session | Measurement | Our sampled decode is not host-rejection-bound at k=1 (33 vs 36 tok/s, −8%); bucket-pinning env exists in our tree; gather_sum all-reduce does not | live MTP k=1 server | 1 | Sampled-decode concern closed for current shapes; two new READY tasks (B021, B022) | VERIFIED (measurement) | experiments/RESEARCH-001-draft-graph-upstream.md (addendum 2) |
| RESEARCH-001 | KB session | Research | The eager-draft bottleneck (93 ms host vs 17 ms device per step) has a measured fix: abotsis fork 919ec98f ships draft-graph machinery (private storage copies, attention-in-graph) with MTP-4 numbers: accepted 3.33 vs 2.41 bypass, aggregate 26.0 vs 20.2 | abotsis fork 919ec98f (reference; not our tree) | 8 (their runs) | Our eager-draft k=2/concurrency findings are likely artifacts of this host-launch bottleneck; A014 becomes READY (adapt, not invent); addendum: their fork ALSO solves KDA candidate-slot state (appended private slots, batched chain kernel `chunk_kda_spec_states`, condense/swap-stable ids) and a third independent port exists (inderjodhsingh/gaudi-llm-notes) — strategic base decision needed from the owner | Research complete | experiments/RESEARCH-001-draft-graph-upstream.md |
| MTP-004 | MTP phase | TTFT / prefill knobs | Draft prefill pass and KDA chunk size affect TTFT | k=1; KDA chunk 64 vs 128 | 8 | Warm TTFT 2.1 s @2K, 3.1 s @8K, 4.2 s @16K; chunk=128 WORSE (8K 1.85 s vs 1.29 s) — chunk=64 retained; `GLM5_NEXT_KDA_CHUNK` env knob added | VERIFIED (measurement) | experiments/MTP-004-ttft-chunk-size.md |

## Historical note on the reported ~21% result

The originally reported "approximately 21% higher decode throughput at batch size 1" compares
MTP=1 (36.8 tok/s, 32K/8seq window) against a non-MTP control measured on the production window
(30.3 tok/s, 256K/32seq). Those conditions are NOT equivalent. Under the like-for-like 32K/8seq
window the single-run comparison is +14% (35.7 vs 31.4 tok/s). Both observations are preserved;
neither has passed the repetition/variability requirements of `docs/BENCHMARK_POLICY.md`. The
Stage-1 task (A001) is the controlled reproduction.
