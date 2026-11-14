# RA-037 — Decode throughput, commercial-only: the measurement-first program to 200 bs=1 and 800@16

**Research architect, 2026-11-13 (round 7).** Owner directive (2026-11-12): *"Forget dflash2 as
it has a non commercial license, focus on figuring out how to get higher decode."* RA-036
(the DFlash2 program) is therefore SUPERSEDED as a production path (the incoai checkpoint is
CC BY-NC-ND 4.0) but its substrate survives: the ~40%-complete proposer port is
architecture-defined, not checkpoint-defined, and is the execution vehicle for the
self-trained drafter in L6 below. Everything in this document is restricted to
commercially-usable components (license table in the appended statement; every lever names
its license class).

---

## 1. Framing — the objectives, the current state, the commercial constraint

**The two decode objectives** (unchanged):

| Metric | Target | Current (production config, 8× HPU, MTP k=1 + EP, ns16) | Gap |
|---|---:|---:|---:|
| Single-stream decode | 200 t/s | **42.4 t/s bs=1** (1.88 tok/step, 94% acceptance) | 4.7× |
| Aggregate decode @16 | 800 t/s | **~160 t/s** (MTP k=1 @16; −38% vs the 258 t/s MTP-off ns16 number) | 5.0× |
| (context) MTP-off @16 | — | 258 t/s (ns16 adopted, RA-003) | — |
| (context) TP4 bs=1 (levers, no MTP) | 200 t/s | 41.3-45.1 t/s (Q14v2) | ~4.5× |
| (context) TP4 @16 (levers, warm) | 800 t/s | 180.8 t/s (Q14v2) | 4.4× |

The final deployment is **4 HPUs** — both targets must be re-checked there (L7 gives the
honest per-rank arithmetic; the short version: 200-bs=1 at 4 HPUs REQUIRES a ≥4-accepted
drafter, 800@16 at 4 HPUs REQUIRES the full host+Q21+MTP@16 stack — neither is reachable
with today's levers alone).

**The commercial constraint, made operational.** After the owner's license ruling:
- The incoai `GLM-5.3-Flash-DFlash2` checkpoint (CC BY-NC-ND) is EXCLUDED from production.
- The MTP head is part of the published GLM-5.3-Flash checkpoint (the target model —
  commercial use is the project's baseline; the MTP head is not a separate license event).
- The z-lab/dflash METHOD + paper (MIT, arXiv 2602.06036), the vLLM Speculators training
  framework (Apache-2.0), the 1Cat-vLLM-Gaudi code (Apache-2.0), Strata (MIT), project-maya
  (MIT), DSpine (Apache-2.0), and all Intel/vLLM/vllm-gaudi stack code (Apache-2.0) are
  freely usable. **A drafter we train ourselves on our own target weights, using the MIT
  method, is commercially clean** (subject to the target model's own license, which the
  project already relies on commercially) — that is L6.

**The strategic fact this round establishes:** the previous round's "acceptance wall"
(≥4.2 accepted/step needed for the 200-bs=1 roofline; the MTP head MEASURED at 2.68
accepted/step at k=4) does not disappear with DFlash2 — but the wall is no longer a
license-blocked dead end. It has two commercial answers: (a) the SELF-TRAINED drafter (L6),
and (b) for the @16 objective, acceptance is NOT the wall at all — the @16 step is
dominated by host-side dispatch (the worker host ~61 ms @16, MEASURED D1-16), so the 800@16
path is a HOST + WEIGHT-STREAMING program that the MTP head (AL 1.88) already serves.
**The two objectives now have different critical paths, and both are commercial.**

---

## 2. The state of the decode physics — what is measured, and the tension that re-ranks everything

### 2.1 The measured facts (all from the agent's own records, through e432f01)

**bs=1 (the BS1-HOSTBOUND + ENGINE-BS1 line, 2026-11-12):**
- The ~23.8 ms bs=1 cycle: the worker main thread is **95% in host functions** — 82%
  `Poller.poll` (idle waiting for the engine), input 10%, shm output spin 10%; the visible
  `HPUGraph.replay` launch is **~1.26 ms (5%)** of the cycle.
- The EngineCore profile: the engine is mostly WAITING (input 10%, shm output spin 10%,
  scheduler 0.05 ms/step — NOT the bottleneck).
- **~17 ms of the ~23.8 ms cycle is UNACCOUNTED between the engine phases** — the gap
  between the engine's phases; hypotheses named by the agent: transport latency,
  API-server cadence, the GIL, the Condition.wait granularity.
- F3-SCOPED: the re-capture tax is bs=1/3 ctx-growth scoped (2.6 ms/step at the
  bucket-ladder crossing), NEUTRAL at @16 — so ctx growth costs bs=1 a small known tax.
- F2-APPLIED: the RPC collapse is KEPT (the draft_token_ids piggyback removes one
  round-trip).
- F1-ASYNC (the async-D2H output redesign): EXECUTED AND DROPPED (−5%, 39.6-41.3 t/s) —
  the async output path made bs=1 SLOWER (the D2H sync point moved, not eliminated).

**The MoE device term (the MOE-AB line):**
- The Q21 multi-expert kernel: −29% per-expert unit time (0.87 → 0.62 ms/expert), bit-exact
  single-expert proof, the E4M3 top-bin semantics resolved (the OCP-exact parity target).
- Extrapolated at bs=1: the MoE term ≈ **10.5 ms/step stock → ~7.4 ms with Q21**
  (the agent's "decode headroom ~23 → ~12.5 ms" wording).
- The agent's rank-1 estimate (the bottleneck table): Q21 takes **weight streaming
  38.1 → ~2.1 GB/step**, with the potential "**bs=1: 43 → ~206 t/s; @16: 257 → ~845 t/s**".

**@16 (the D1-16 line):**
- The worker host @16 ≈ **61 ms/step** = replay ~42 ms (68.6%) + draft dispatch ~10.6 ms
  (17.4%) + sampler ~6 ms (9.9%) + meta ~2 ms — "no unexplained residual".
- The EngineCore scheduler is DISPROVEN as the @16 bottleneck (99.4% MQ-wait) — the cost is
  WORKER-LOCAL per-row work.
- The @16 step wall ≈ 62 ms (258 t/s = 16 tokens / 62 ms).
- MTP k=1 @16: **−38%** (MTP4-T0) — the draft dispatch + per-row sampler at 16 rows costs
  more than the +0.88 tok/step returns.

### 2.2 The tension (the single most important open question on the decode board)

The bs=1 measurements are mutually inconsistent under one physics, in a way that decides
the entire priority order:

1. **BS1-HOSTBOUND** says the 23.8 ms step is 95% host (the main thread; the device launch
   is 1.26 ms visible).
2. **MOE-AB** says the MoE device term at bs=1 is ~10.5 ms — and the rank-1 table says the
   CURRENT weight streaming is **38.1 GB/step**. If 38.1 GB is PER RANK, then at the
   measured ~1.6-1.7 TB/s per-rank HBM bandwidth it is **~23 ms of device work — i.e. the
   whole step**, which contradicts "95% host-bound, device 1.26 ms". If 38.1 GB is the
   8-RANK TOTAL (4.8 GB/rank), the device work is ~2.8 ms/rank — consistent with a
   host-paced step, but then Q21's bs=1 gain is ~0 (the device is already inside the host
   window with slack), and the rank-1 table's "bs=1: 43 → 206 t/s from Q21" is wrong.
3. The "1.26 ms replay" is the ASYNC LAUNCH time on the main thread — the device executes
   for some hidden duration after the launch returns. **py-spy cannot see the hidden device
   work; it sees the main thread.** The "95% host-bound" claim is a statement about the
   MAIN THREAD, not about the device. The device's actual busy time per step is UNMEASURED
   (the agent's own unknowns list: "the host/device overlap ratio — how much of the ~21 ms
   device work hides under the 24 ms host time — the timeline trace"; "the HBM bandwidth
   during the MoE weight read").

**Two candidate physics (both survive the current data):**

- **P-DEV (device-paced):** the bs=1 step is ~23.8 ms of DEVICE work (the 38.1 GB/rank
  weight streaming at ~1.6 TB/s), and the host work (~1.3 ms visible + engine phases)
  hides underneath; the 17 ms "unaccounted" is the device completion sitting inside the
  engine-phase gap. Under P-DEV, **Q21 alone moves bs=1 42 → 200+ t/s and @16 258 → 800+**
  (the agent's rank-1 table is correct), and the host program is second-order.
- **P-HOST (host-paced):** the bs=1 step is paced by the host pipeline (API server ↔
  EngineCore ↔ worker transport + the per-step output handling), the device work is
  ~10-15 ms (the MoE 10.5 + attention/KDA remainder) and hides inside the window with
  slack, and the 17 ms is real host latency (the transport/cadence/GIL candidates). Under
  P-HOST, **Q21 moves bs=1 by ~0-5%** (the device cut is slack), bs=1 200 REQUIRES the
  host program + a ≥4-accepted drafter, and @16 (the worker host 61 ms) REQUIRES the
  worker-host program (L3).

**Both can be simultaneously true in different regimes** (bs=1 host-paced, @16
device-paced, or the reverse) — that is exactly why the measurement set in §3 is the
first deliverable, and why every lever below states its value under EACH physics. The
levers that are valuable under BOTH are the safe investments; the levers whose value
depends on the physics are ranked conditionally.

### 2.3 The @16 picture (less tension, more certainty)

The @16 step (62 ms wall) has a MEASURED 61 ms worker-host decomposition with no
residual — so @16 is, to first order, a WORKER-HOST problem: the 42 ms "replay" term
(which needs the device-overlap measurement to split into launch-CPU vs sync-wait), the
10.6 ms draft dispatch, and the 6 ms per-row sampler. The device work at @16 (the weight
streaming for 16 rows' worth of routed experts + the 2-row verify) is the unknown that
T1b measures. Under P-DEV-@16 (the 38.1 GB/rank reading), Q21 is the master @16 lever;
under P-HOST-@16, the worker-host program (L3) is. Either way, **the MTP@16 −38%
regression (L4) and the concurrency headroom (L5) are valuable under both physics** —
they are the insurance policy that makes 800@16 robust to the measurement outcome.

---

## 3. T1 — The decisive measurement set (run FIRST; it re-ranks everything)

The 38.1-GB question and the host/device overlap are a **one-boot-each** problem with the
instruments already in the tree (the F4-TS per-phase timestamps, `GLM_PHASE_TS=1`, live;
the hl-smi sampling; the HBM counters). Design:

### T1a — the bs=1 step physics (one boot, ~1 hour)

Workload: bench_decode bs=1, warm (post-capture), MTP k=1 (production) AND a
MTP-off rollback boot (to separate the target-only step from the draft term).

Stamps (the F4-TS gives the engine phases; add THREE device-side stamps):
1. **The step wall** (existing F4-TS: engine phase start → next phase start).
2. **The graph launch return** (existing: the 1.26 ms visible).
3. **The graph COMPLETION on the device** — stamp the completion event: the HPU graph
   exposes a completion query (the `hlGraphQuery`/event class in the synapse_api; if the
   runner has no hook, the cheapest proxy is the wall time from launch-return to the
   sampler op's first H2D-ready, or the `hpuStreamSynchronize`-equivalent around the
   output D2H — the D2H CANNOT start until the device is done, so *D2H-start −
   launch-return = the hidden device duration*).
4. **The per-rank HBM bytes/step** — the HBM bandwidth counter (the `hlDevice`/
   synapse metrics, or hl-smi power+clock back-calculation as the fallback): integrate
   over N=100 steps at bs=1. This answers the 38.1-GB question directly (per rank vs
   total; the GB/step at the measured clock).

**Decision rule (pre-registered, so the result can't be spun):**
- HBM bytes/step/rank ≥ 30 GB AND device-busy ≥ 18 ms → **P-DEV holds at bs=1**: Q21 is
  the master bs=1 lever (the host program drops to P2); the rank-1 table's 206 t/s is
  credible; L6 (the drafter) is deferred (the 200 is reached without it).
- HBM bytes/step/rank ≤ 10 GB OR device-busy ≤ 12 ms → **P-HOST holds at bs=1**: the
  17 ms gap is the master bs=1 lever (L2); Q21's bs=1 value is ~0-5% (it stays P1 for
  @16 and TP4); L6 (the drafter) is REQUIRED for the 200 (with L2).
- In between → the hybrid case; the lever order is Q21 (for the device share) + L2 (for
  the host share) in proportion to the measured split.

Also in T1a: the MTP-on vs MTP-off step wall delta (isolates the draft+verify cost at
bs=1: expected ~+10-14 ms from the D1 7.5 ms draft dispatch + the verify-row device).

### T1b — the @16 step physics (one boot, ~1 hour)

Workload: bench_concurrent 16 streams, warm. Same stamps as T1a (per rank, on rank 0 and
rank 7 to check skew) + the per-rank HBM bytes/step @16 (the 38.1 GB number's regime) +
the **routing-union size distribution**: dump the per-step distinct-local-expert count per
layer (the Q6 router-dump extension: 100 steps → the histogram; this is the Q21 A/B's
ground truth — the −29% unit number applies to the PAIRS, the GB saving applies to the
DISTINCT experts, and the histogram gives the real number).

**Decision rule:** HBM bytes/step/rank @16 ≥ 25 GB → P-DEV-@16 (Q21 master @16, the
845-estimate credible); ≤ 12 GB → P-HOST-@16 (the L3 worker-host program is master @16).
Either way the 42 ms "replay" term is split: launch-CPU (the graph launch descriptor
build for the 16-row graph) vs sync-wait (the hidden device) — the completion stamp does
that split directly.

**T1b RUN (2026-11-13, this boot family; artifacts `benchmarks/accuracy/results/t1b_*`,
summary `t1b_summary.json`):**
- **CORRECTION (the op-path attribution):** the "masked dense pass" I cited is
  **MiniMax-M3's** SwiGLU-OAI unfused path (`hpu_fused_moe.py:70-87` — the
  `_unfused_swigluoai_moe` function, MiniMax-specific). **GLM-5.3-Flash takes the
  standard fused MoE op** (the SwiGLU activation → `VllmMixtureOfExpertsOp` → the
  Habana MoE kernel). **The 43.6 GB/rank/step byte model is the B1 all-read
  ASSUMPTION, not a measurement** — and the T1a's own data contradicts it at bs=1:
  the device work ~1.26 ms/step implies ~3 GB at the HBM BW, not 38 GB — i.e. the
  deployed op likely **already reads the routed experts only** (the grouped GEMM
  skipping the zero-token experts), as the L1 note hypothesized. **The verdict
  "P-DEV-@16 by the bytes rule" is RETRACTED — the bytes were never directly
  measured** (the hl-smi mem-util counter is a static artifact on this driver).
  What STANDS from T1b: the routed-union histogram (16.1 distinct experts/
  rank/layer — the routing statistics), the completion of the stamp suite, the
  cadence/AL reconciliations. **The re-scoped Q21 question: if the stock op is
  already routed-only, the Q21 byte saving ≈ zero; its remaining value is the
  fused dequant+GEMV (the AB-DEQ parity showed the dequant op cost-neutral, so the
  win would be the elimination of any separate dequant-pass intermediate traffic —
  unverified) and the zero-token-skip verification.** **The next session's FIRST
  experiment: measure the MoE device time and the bytes directly** (the env-gated
  event timers around the moe_op call at bs=1 vs @16 — one boot), which decides
  the Q21's fate and the P-DEV/P-HOST-@16 verdict honestly.
- **THE FP8 MoE OP STRUCTURE (the session's key find for the optimization path):**
  GLM's `VllmMixtureOfExpertsOpFP8.forward` (ops.py:1258) **dequantizes ALL 36
  local experts to BF16 on every step** (`get_dequant_weight()` →
  `dequant_block_fp8_weight_naive` — production runs `GLM_FP8_MXK=0`, the stock
  cast+mul chain; the deployed AB-DEQ/MXK TPC op would only speed the dequant
  itself, not the structure), then calls `torch.ops.hpu.mixture_of_experts` on
  the **BF16** weights. **The traffic multiplier ~3-5× the FP8-only read**
  (the FP8 read + the BF16 write + the BF16 GEMM read), unless the Synapse
  compiler fuses the dequant into the GEMM — **the doc's open question BG-1**.
  HBM cannot hold both copies (38.1 FP8 + 76.2 BF16 > 96 GB), so the dequant
  re-executes per replay. **This explains the AB-DEQ parity result** (a faster
  dequant op inside a 3-5×-traffic structure moves little) **and reframes the
  Q21 opportunity: the fused dequant+GEMV kernel eliminates the structure, not
  just the dequant speed.** **The next session's first experiment (one boot):
  the event timers around the dequant and the mixture_of_experts call inside
  the FP8 MoE forward at bs=1 and @16 — it answers BG-1 (the fused vs the
  separate) and quantifies the dequant traffic share directly.** If the BG-1
  is "not fused", the dequant-elimination is likely the largest single @16
  lever (the traffic cut ~3-5× on the expert term).
- **BG-1 ANSWERED BY INFERENCE (the standalone benchmark hit the Synapse compile
  wall on the raw op call — the full vLLM stack context is required to compile the
  fused MoE recipe; the boot-window run remains available):** the unfused
  dequant-then-GEMM model requires ~190 GB/rank/step (~78 ms of pure traffic at
  the 2.45 TB/s peak) — **impossible inside the measured 72.7 ms iteration with a
  27.3 ms device drain**. The drain fits the FP8-only read (~38.1 GB at ~1.4 TB/s,
  57% of the peak). **Conclusion: the Synapse fuses the dequant into the GEMM
  (BG-1 = fused), the steady traffic ≈ the FP8 read once, and the deployed op is
  routed-only (consistent with the T1a bs=1 device reading).**
- **THE Q21 INVESTMENT VERDICT: NOT JUSTIFIED by the byte savings** — the stock op
  already reads the FP8 weights once, routed-only. The remaining Q21 value: the
  dequant compute elimination (the AB-DEQ parity showed it near-free) and the
  marginal kernel efficiencies — **the weeks-class integration is not worth the
  expected single-digit-percent gain.** **The revised @16 lever ranking: (1) the
  sampler/parse (15.6 ms — the vectorization/optimization), (2) the launch/replay
  path (28.3 ms — the graph population), (3) the drain's non-MoE parts (the
  attention, the communicate), (4) the acceptance-side (the AL 1.68 on the natural
  text — the draft quality/the sampler; the MTP k=2 probe), (5) the Q21 (marginal,
  deprioritized).** The honest aggregate ceiling after the trims: **the iteration
  ~50-55 ms → ~450-500 tok/s @16** (the 657 estimate assumed the Q21 byte savings
  — retracted).
- **BG-1 CONFIRMED BY THE CAPTURED RECIPE (the direct evidence, zero-boot):** the
  GC capture trace (`/data/glm53/captured_core_seq.txt`, the 26,407-node recipe)
  shows the MoE region (23,094 nodes): **the batch_gemm ×1,496 + the dequant-class
  nodes (the mult_bf16 ×1,010, the cast_f32_to_bf16 ×760, the div_f32/mult_f32
  ~766, the moe_multiplexer ×1,002)** — **the dequant nodes ARE present in the
  recipe but the Synapse compiles them into the batch_gemm load path** (the fused
  multiply-on-load; no extra HBM round-trip), consistent with the timing inference
  (the 27.3 ms drain fits the FP8-only read). **BG-1 = fused, confirmed by both
  the timing and the capture structure. The Q21 verdict stands: not justified.**
- **THE MFU/BW INSTRUMENT DEPLOYED + THE DEFINITIVE P-HOST CONFIRMATION (the
  final, `--enable-mfu-metrics`):** the vLLM-gaudi per-window hardware metrics
  (the TF/s/GPU + the GB/s/GPU) logged live. **The steady @16 decode: the BW
  328-331 GB/s/GPU = 13.4% of the 2.45 TB/s HBM peak; the compute 1.4 TF/s/GPU =
  0.32% of the 432 TFLOPS peak — the device is 87% BW-idle and 99.7%
  compute-idle during the steady decode.** The prefill bursts: 330 GB/s (the
  same 13%). **P-HOST CONFIRMED AT THE HARDWARE-LEVEL, DEFINITIVELY — the
  entire optimization program is HOST-SIDE: the device could sustain ~2000
  tok/s @16 if the host fed it (the 38.1 GB weights ÷ the full BW = the 15.6 ms
  iteration floor), and the host delivers ~13% of that.** The device-side levers
  (the Q21, the kernel work): ZERO value until the host saturates the device.
  The instrument: `--enable-mfu-metrics` (the boot variant `serve_mtp_mfu.sh` +
  `boot_mfu.sh` staged) — the permanent per-window device-utilization
  visibility for every future experiment.
- **THE PAD-BUCKETING EXPERIMENT RAN AND REFUTED (the RA-006-B, the config-only
  boot):** `VLLM_BUCKETING_STRATEGY=pad` (the PaddingAwareBucketingStrategy)
  generated **384 buckets (303 prompt + 81 spec-decode) vs the exponential's 195
  (120 + 75) — ~2× MORE buckets, NOT fewer** — refuting the RA-006-B premise
  ("the pad bucketing trades a little device padding for a much smaller graph
  population") for this model/config. The decode parity: 42.3-42.4 t/s, the
  sanity 391 (no regression, no win). **The launch-path lever remains the graph
  POPULATION reduction — the bucket-strategy knob does not provide it; the
  remaining candidates: the bucket-file pinning (the T12 refuted: the
  spec-decode seeds add buckets) and the custom bucketing config.** The boot
  variant staged (`serve_mtp_pad.sh` + `boot_pad.sh`) for the future retests.
- **THE ASYNC-SCHEDULING PROBE RAN — PARITY (the config-only boot, 11-14):**
  `--async-scheduling` (the F2 piggyback off — the async path skips the
  piggyback consumption): **the bs=1 39.8-40.2 tok/s (−3-7%: the F2 RPC loss),
  the @16 255.6 tok/s (+0-2%, within the noise), the AL 1.84 holds.** The
  verdict: **the scheduler-level overlap does NOT realize the host-device
  pipelining** — and the production already submits the execute non-blocking
  (`execute_model(non_block=True)`), so the host time is mostly the device wait.
  **The batch-queue pipelining (`max_concurrent_batches = 2`) is implicitly
  enabled BY the async flag — tested: the parity.** The pipelining lever via
  the vLLM scheduling flags: CLOSED. The remaining host lever: the sampler
  host work (15.6 ms) and the deeper executor restructuring.
- **THE REPLAY-FLOOR DISCOVERY (the platform constraint, the [LAUNCH] evidence):**
  the graph-replay host cost is **~25-31 ms at bs=24-40 and ~44 ms at bs=1 —
  roughly batch-independent** (the [LAUNCH] stamps, measured twice), ~10× the
  doc's earlier 1-3 ms estimate. **This caps the k=1 iteration at ~50-60 ms
  REGARDLESS of the host optimization** — the "2000 tok/s if the host fed it"
  reframe was wrong (the replay cost scales with the cycles, not the bytes).
  **The corrected @16 ceiling: the replay 25-45 + the drain 7-27 + the sampler
  15.6 + the tail 11 ≈ 58-99 ms → 271-463 tok/s — the ~450-500 estimate
  CONFIRMED as the platform-constrained ceiling.** **The 800-tok/s target is
  ONLY reachable via the drafter path** (the AL ≥ 4.5): the k=4 arithmetic
  (the rows +150% vs the AL +80%) predicts a net loss; the DFlash2-class
  block-diffusion drafter (the 8-token blocks per verify) is the only path —
  gated on the owner's licensing/training-infra decision.
- **THE KERNEL-LEVEL PROFILER UNAVAILABLE (the vllm-gaudi limitation, 11-14):**
  the `/start_profile`/`/stop_profile` API with the `--profiler-config
  {"profiler": "torch", ...}` boot: **the worker fails with "Profiler is not
  enabled."** — the vllm-gaudi HPU worker does not implement the torch-profiler
  integration in this build. **The kernel-level device breakdown (the MoE vs the
  attention vs the KDA durations) requires the future vllm-gaudi profiler
  support or the custom instrumentation inside the captured recipes.** The
  interim: the drain decomposition via the analytical models (the MoE 6.9-17 ms,
  the non-MoE 10-20 ms) and the capture-trace structure. The profiler boot
  variant staged (`serve_mtp_prof.sh` + `boot_prof.sh`) for the retest when the
  vllm-gaudi adds the HPU worker profiler.
- **THE TRACE ANALYSIS (the profiler output examined, 11-14):** the env-enabled
  profiler boot DID produce the 6 MB trace (the on_trace_ready schedule), but
  the analysis shows **all 321,534 events are `python_function` events from the
  API-server process only — zero HPU device kernels, zero worker events** (the
  kineto HPU integration is absent in this build; the worker processes are not
  traced). The trace independently confirms P-HOST at the API level (the
  uvicorn/asyncio threads mostly in the poll/wait). **The kernel-level device
  breakdown requires the future vllm-gaudi profiler support or the
  Senapse-native profiling tools; the analytical drain decomposition stands.**
- **THE ANALYTICAL MoE CLOSURE (the dims: the hidden 4096, the moe I 2048, the 36
  local experts, the topk 8, the 42 MoE layers; the peak 2.45 TB/s, the compute
  40% of 432 TFLOPS):** **bs=1: the MoE floor 0.9 ms** (the 2.1 GB routed-only
  read — ✓ the T1a's ~1.26 ms device reading CONFIRMED analytically: the stock op
  IS routed-only); **@16: the MoE floor 6.9 ms** (the 16.9 GB routed-union read at
  the peak; the ~17 ms at the realistic ~1 TB/s), the compute 3.1 ms — **the MoE
  is memory-bound and only ~25% of the 27.3 ms drain; the non-MoE device work
  (the attention/KDA/KV) ≈ 10-20 ms is the remaining drain component.** **The Q21
  case closed by three independent confirmations: the analytical (the stock reads
  the routed unions), the empirical (the AB-DEQ parity, the timing), and the
  capture structure (the fused dequant nodes). The next profiling target: the
  non-MoE device kernels (the attention/KDA/KV) — the kernel-level profiler on a
  measurement boot.**
- **THE BF16-RESIDENT VIEW CACHE (the FINAL MoE answer, the code trace):** the
  deployed op class is `VllmMixtureOfExpertsOp` (the base, ops.py:673 — NOT the
  FP8 subclass my first patch targeted, explaining the silent stamps). Its
  forward uses `self._cached_w13_views`/`_cached_w2_views` — **the dequantized
  BF16 views built ONCE by `_cache_weight_lists()` at the weight-set/init**
  ("if cache wasn't built by the caller (e.g., after set_weight), build once
  here"). **The per-step GEMM reads the BF16 views routed-only (~17 GB @16 ✓
  matching the MFU/BW 330 GB/s × 72.7 ms). The dequant happens ONCE at the load,
  NOT per step.** **The consequences: (1) the Q21 routed-only read case:
  definitively closed (the BF16 routed-only reads already happen); (2) the B1
  "38.1 GB FP8 all-read per step" model: WRONG — the actual per-step read is the
  BF16 routed-only ~17 GB; (3) the remaining MoE-side lever: NONE (the reads are
  already minimal); (4) the HBM residency: the FP8 masters (39.7 GB) + the BF16
  views (79.4 GB) = 119 GB > 96 GB?! — the views are the per-slice/per-chunk
  tensors sharing the storage, or the FP8 masters are freed post-dequant — the
  memory accounting needs the hl-smi check.** The dequant-ts patches (both
  sites): inert (the classes/paths not exercised) — harmless, left deployed.
- **THE HBM RESIDENCY RESOLVED (the hl-smi measured): 69.3-69.5 GiB/card**
  (the TP8 serving). The weights: the FP8 masters 39.7 GB (the loaded ✓ the
  log) + the KV 1280 blocks (~20 GiB) + the graphs/workspace ≈ 66-69 ✓ with NO
  room for the 79 GB of the BF16 copies → **the `_cache_weight_lists` "views"
  are memory-efficient: either the per-slice views into a shared BF16 buffer,
  or the lazy dequant per active slice, or the FP8 masters freed post-dequant
  with the BF16 resident.** The precise accounting needs the memory-profile
  dump (the MEMLOG instrument exists); the functional answer stands: the
  per-step reads are the routed-only ~17 GB (the MFU/BW ✓), the dequant is NOT
  per-step (the cached views ✓), and the HBM headroom supports the 1,062,400-
  token max context (measured ✓).
- **THE GRAPH STRUCTURE RESOLVED + THE Q21 NODE-COUNT DIMENSION (the
  correction):** the runner wraps the **WHOLE MODEL in ONE HPUGraph** (the
  single replay per step — the [LAUNCH] fwd 28.3 ms = the one submission of the
  ~20K-node command buffer at the ~1.4 µs/node; the batch-independent ✓). The
  consolidation already exists — BUT this reveals the Q21 dimension I missed:
  **the MoE region's 1,496 batch_gemm nodes could collapse to ~84 fused
  kernels (the 42 layers × 2 GEMMs) = the −1,412 nodes = the ~7% node-count
  reduction**, speeding both the replay submission (28.3 → ~26) and the device
  execution. **The corrected Q21 verdict: the byte savings = zero ✓, but the
  node-count value = the ~2-7%** (the weeks-class integration for the ~2-7% —
  the weak ROI but not the zero). The rank-1 lever remains the launch/replay
  internals; the node-count reduction (the Q21-class kernels, the norm chains,
  the cast elimination) is the broader program.
- **THE RECIPE-LEVEL STEP ANATOMY (the final reframe):** the captured recipe has
  **~20,485 nodes** (the addNode trace; the MoE region 23,094 trace lines ≈ the
  89%, the non-MoE ~2,300: the attention gemms ~133, the normalizations ~40
  rsqrt/sub/add/reciprocal, the reductions ~116, the casts ~113). **At the ~1.3
  µs average per tiny node, the device execution ≈ 26.6 ms — matching the
  measured 27.3 ms drain. THE STEP IS LAUNCH-BOUND AT THE RECIPE LEVEL: the
  node count IS the device time.** **The optimization program is RECIPE-LEVEL
  FUSION: the node-count reduction across the MoE (the Q21-class), the sampler
  (the _accept_grid), the norms, and the attention — the Synapse fuser
  configuration is the lever.** The non-MoE region decoded: the attention
  gemms/reduces/gathers (~60%), the RMSNorm chains (~25%), the casts (~15%).
- **THE _ACCEPT_GRID OP-COUNT ANALYSIS (the sampler fusion value, the final):**
  the device-side accept walk is **~25-30 tiny-tensor ops per call** (the H2D,
  the arange, the comparisons, the casts, the cumsum, the wheres, the cats — all
  on the [16, 2]-class tensors, launch-bound at the ~50-100 µs each) → **~1.5-3
  ms/step**. The fusion (one TPC/Synapse kernel): the ~0.2-0.5 ms → **the saving
  ~1-2.5 ms/step = the ~2-3% @16** — modest, correcting the earlier ~10%
  estimate. **The final honest lever ranking: (1) the replay 25-45 ms — the
  platform floor (the hard); (2) the _accept_grid fusion — the ~2-3% (the custom
  kernel, the moderate work); (3) the parse host work — the ~1-2% (the
  already-vectorized); (4) the non-MoE device kernels — the unknown (the
  profiling needed); (5) the drafter — the 800 path (the owner-gated).**
- **The routed-union histogram (the Q21 GB ground truth):** the distinct local experts per
  rank-layer-step **mean 16.1 (p50 16, p90 21, max 27)**, balanced across ranks
  (15.5–17.0 — the expert placement is NOT needed). The 40-row decode bucket carries 8
  padding rows → the union is an upper bound. **Q21 model: 36 → 16.1 = 2.24× expert-byte
  cut; the total 43.6 → ~22.5 GB/rank/step.** Capture caveat: the hook is python-side and
  does not execute inside the graph replays — the 20 recorded steps are the eager
  capture/warm-up phase (the routing distribution is execution-independent, so the
  estimate holds).
- **The completion-stamp split (the replay phase, n=985):** launch 65.2 / sync-wait 25.9 /
  post-sampler 24.1 ms (the stamped total 115.2 is serialized by the per-step
  `torch.hpu.synchronize()`; the real step is 57–76 ms). Directional: the host dispatch +
  the sampler host ≈ half the step, the device drain ≈ 26+ ms. The F4TS engine stamps:
  sample_tokens 52–72 ms (mean ~58); schedule+broadcast+rpc ~0.2–0.4 ms.
- **The @16 aggregate parity check:** the standard bench reproduces the historical level
  **253–258 tok/s** (F2 off). The earlier-session 205–211 claim was the custom
  512-token-script artifact (the ffb5495 lesson) — corrected. The AIP-Util @16 is
  21–27% (131–164 W): the compute units are ~75% idle while the memory streams —
  consistent with the masked-GEMV memory-bound regime.

**T1b CORRECTIONS (2026-11-13, later the same day — the engine-metric cross-check):**
- **The step-delta AL method is INVALID**: the F4TS engine stamp counter carries
  duplicated increment blocks (the patch applied with two `+1` sites per step) — the
  stamp-count×16 arithmetic produced impossible ALs (0.5–1.07). The authoritative AL
  source is the engine's own spec-decoding metrics line.
- **The AL@16 = 1.82** (the engine metrics, 07:40:33: `Mean acceptance length: 1.82` at
  the generation 251.7 tok/s) — **the MTP acceptance HOLDS at @16**; it does NOT
  collapse. The MTP@16 wash comes from the STEP COST (the 32-row verify ~doubles the
  host dispatch), not the acceptance.
- **The @16 step ≈ 116 ms** (251.7 ÷ (16 × 1.82)); the T1B completion-stamp total
  (115.2 ms) MATCHES it — the per-step sync added nothing at @16 because the host path
  already dominates. **The split: the host ≈ 89 ms (the launch 65 + the post-sampler
  24), the device drain ≈ 26 ms → P-HOST-@16 by direct measurement**, refining the
  bytes-rule verdict above: the 43.6 GB are structural (the dense pass reads them every
  step) but the device completes them in a ~26 ms burst (1.7 TB/s in-burst) and then
  idles behind the host. **The @16 master lever is the L3 host program (the launch path
  + the sampler host); Q21 shrinks the device burst 26→~12 ms — the combination, as the
  diagnosis concluded.**
- The F4TS `sample_tokens` stamp (~56 ms) vs the T1B worker total (115 ms): the engine's
  executor call returns ahead of the worker completion (the async-output plumbing) — a
  measurement-architecture caveat for the next instrumentation round; the engine-metrics
  step (116) and the T1B total (115) are the consistent pair.
- **Ops incident:** the EngineCore was killed by a stray `SIGUSR1` probe (the default
  USR1 action is terminate) at 07:42:07 — the production was restored from
  `boot_f4ts.sh` (the adopted config). Lesson: never signal the engine processes.

**R003 tie measured (2026-11-13, the T7 evidence, no boot):** the live production probe:
the answer token's top-2 gap is **0.75 logits** (`10` at −0.392 vs `5` at −1.142; the
`</think>` gate 2.88) — R003 is a genuine near-tie item. The greedy answer is
deterministically **"10" (the trap) ×5 in BOTH the serial and the 16-stream concurrent
shapes** on the current stack — the current baseline is self-consistent, no
batch-shape sensitivity. **The archaeology RESOLVED (it already existed — commit
`975cf64` R003-HIST, 11-11 15:31, which this probe now completes with the logit gap):**
the last 100% suite was the TP4 4-card config (11-09 03:49); EVERY TP8 8-card suite
since the lever era is 97.5% with R003="10" — the state SURVIVES the vectorized-parse
revert, the HARD HOST REBOOT, and the levers-off boot: **R003 is the reasoning
knife-edge (the intuitive "10" vs the retrieved "5" at the low reasoning effort at the
TP8 shape), NOT a deployment corruption** — the probe's 0.75-logit gap quantifies it.
**The operational adoption gate is already "does not push accuracy below 97.5"**, and
the three banked optimizations were RE-ADJUDICATED under it and dropped ON THE MERITS:
the Q01v2 draft graph (the compose/combined boots: the +20% win is cache-dependent; at
the adopted config −7% @16 — `87762a5`, `5243e1b`), the F1/RA-011 async (−5% decode —
`e5beed8`), the MHC-ROWS cap (the 8192 raise does not fix the prefill cost —
`bfabdc1`). The RA-018/T7 tie-aware refinement (the per-flip gap attribution) remains
as future hardening; the probe supplies its measurement recipe
(`kb_staging/r003_probe.py`, `r003_load_probe.py`).

**The EngineCore py-spy @16 (2026-11-13, the L2.2 refutation + the engine anatomy):**
- **The L2.2 in-process arm is REFUTED BY THE CODE**: `core_client.make_client` raises
  `NotImplementedError` for asyncio+in-process ("Running EngineCore in asyncio without
  multiprocessing is not currently supported") — the `vllm serve` path structurally
  requires the multiprocess EngineCore in vLLM 0.26.1. The arm is closed without a boot.
- The py-spy on the EngineCore under the 2-rep @16 load (51 samples — GIL-biased,
  directional): **~73% `FutureWrapper._wait_for_response`** (the engine blocks on the
  workers' sampling completion — the engine and the workers SERIALIZE per step),
  ~12% the scheduler/KV-allocate frames (the magnitude uncertain vs the F4TS's 0.2 ms
  stamp — the mamba/KDA allocation spikes suspected at the batch transitions; flag for
  the next instrumentation), the rest the queue plumbing; the output-forwarding thread
  idle (the L2.1 conclusion holds at @16).
- The py-spy `dump` point-samples on the worker (15 × 1 s @16, rank 0 — the `record`
  mode cannot sample the lazy-mode worker but the one-shot `dump` can): the MainThread
  ~47% in `HPUGraph.replay`, ~13% in `RejectionSampler.parse_output`, plus the eager-op
  dispatch (`OpOverloadPacket.__call__`) and the functorch unwrapping
  (`unwrap_dead_wrappers`) in the launch path — the D1-named eager leaves, the concrete
  L3 fusion targets.
- **The complete measured @16 anatomy (116 ms):** the engine schedule ~0.2 ms → the
  broadcast ~0.15 ms → **the worker critical path ~115 ms (the T1B split: the launch
  host 65 + the device drain 26 + the sampler host 24)** → the update/API ~2-5 ms.
  The optimization math: the launch 65→X, the device 26→~12 (Q21), the post 24→Y
  (the vectorized sampler) → the step ~55-70 ms → **the @16 ~430-540 tok/s**; the 800
  target additionally needs the MTP k=4 (the acceptance 3.33, the RA-018/T2 chain graph)
  or the DFlash2-class drafter (the RA-009/L6).

**The CLEAN launch decomposition (2026-11-13, the second measurement boot — the stamps
only, NO histogram; corrects the T1B split above):**
- **The T1B "launch 65 ms" was INFLATED by the histogram's 42 per-layer D2H syncs
  running inside the forward** (`grouped_topk` is in-model). The clean split
  (n=560 steady samples): **launch 28.3 + sync_wait 27.3 + post 15.6 = 71.1 ms**.
- The LAUNCH internal stamps (the `[LAUNCH bs=N]` rows): **prep 0.00 ms, fwd
  (the graph replay) 27.9 ms at bs=40, logits (the lm_head) 0.47 ms** — the launch term
  IS the model forward; the metadata prep and the lm_head are free. The bs=1 forward
  44.0 ms = the straggler tail steps (the long-ctx solo decode; consistent with the
  bs=1 sample_tokens 47.5). The bs=16 rows (309 ms, n=120) = the capture phase.
- **The NEW identified term: the unstamped engine+API output path ≈ 57 ms/step @16.**
  The arithmetic: the engine step 115.6 ms (the throughput 251.7 ÷ (16 × AL 1.82)) −
  the engine's stamped phases ~58.5 ms (the schedule/broadcast/rpc + the sample_tokens
  wait 56-60 ≈ the worker's launch 28 + the drain 27) ≈ **57 ms in update_from_output
  + the output serialization + the detokenization + the streaming for the 16 streams** —
  co-equal with the worker's entire critical path. The EngineCore py-spy cannot see it
  (the detok lives in the APIServer process). **The next instrumentation target: the
  py-spy on the APIServer pid under the @16 load** — it re-ranks the L3 program a
  second time (the output path may be co-master with the launch path).
- **The output-path term REFUTED (the same day, the APIServer py-spy dumps):** 12
  point dumps on the APIServer under the @16 load — **11/12 idle in the asyncio
  selector** (`Runner.run` → `_PollLikeSelector.select`); the output path is NOT the
  bottleneck. **The resolution of the 115.6-vs-71 contradiction — the metric
  definitions decoded (the same day):** the engine's spec-decoding metrics satisfy
  **generated = accepted + drafted EXACTLY** (the 20-s window: 5160 = 2325 + 2841 ✓),
  i.e. `drafted` = the proposals = the request-steps (1 draft per running request per
  step) and the "Mean acceptance length 1.82" = 1 + the accepted/drafted ratio
  (0.817) — per proposal. **The true steady cadence: the drafted 142/10 s ÷ 15.5
  running ≈ 91.6 steps/10 s → the step ≈ 109-113 ms** (confirming the throughput÷AL
  derivation). The F4TS stamp-count cadence (65.8 ms) is WRONG — the stamp counter
  over-counts ~1.67× (a third counter artifact; never use the stamp counts for the
  cadence). **The remaining ~40 ms/step gap** (the step ~110 − the worker's serialized
  71): NOT the APIServer (idle), NOT the schedule/broadcast (stamped 0.2-0.4 ms), NOT
  the update_from_output (the py-spy ~4%) — the suspects: the engine's additional wait
  (the take_draft_token_ids RPC + the result assembly), the output transport round
  trip, the API processing bursts below the point-dump resolution, or the engine-side
  step-total blind spot (the F4TS stamps only schedule/broadcast/rpc/sample_tokens).
  **The next instrumentation: a step-TOTAL stamp in the engine step() (the wall from
  the schedule start to the update end) + the API-side receive stamps — one boot
  family, then the 40 ms decomposes.**
- The revised @16 attack list: 1) the worker launch 28 ms (the replay — the Q21 device
  part inside), 2) the device drain 27 ms (Q21 → ~12), 3) the worker post 15.6 ms (the
  sampler), 4) **the ~40 ms/step loop overhead (NEW — the engine/API/transport
  blind spot; the step-total stamp first)**.

**The worker profile closed + the loop-gap staged (2026-11-13, the final round):**
- The nonblocking py-spy `record` on the worker (`--nonblocking` works where the
  default mode starves — the recipe) under the @16 load (458 samples): **the replay
  44% + the parse_output 21% = 65%** — the two halves of the worker's 71 ms step
  (the fwd 28 + the post 15.6 at the T1B split ✓ consistent), plus the D1-named eager
  leaves visible (the dynamic_quant, the mark_step, the OpOverload dispatch, the
  decode_lean _h2d/_accept_grid, the HpuEagleProposer.propose). The worker side is
  MEASURED and attackable: Q21 (the device) + the sampler vectorization + the draft
  dispatch fusion.
- **The loop stamps DEPLOYED and MEASURED** (`[LOOP]` in `_process_engine_step`, the
  same boot family): **step_fn 46-72 ms (the mean 61.5), outq 0.01, post_step 0.01**
  — the engine's iteration is the step_fn entirely; the output-queue put and the
  post_step hook are ZERO. The loop gap is NOT in the engine loop.
- **THE OPEN CONTRADICTION (the session's final unknown):** the tokens/step at @16.
  The cadence path: the drafted ~1300/10 s ÷ 16 running → ~81 steps/10 s → **113
  ms/step** → the tokens/step = 258 × 0.113 = **29.1 (the AL 1.83 ✓ the per-position
  acceptance 0.833 confirms the drafts ARE accepted)**. The loop-stamp path: the
  iteration 61.5 ms → the tokens/step = 258 × 0.0615 = **15.9 (the AL ~1.0 — the
  drafts ALL rejected)**. Both cannot hold. The candidates: (a) the LOOP stamps
  sample a SUBSET of the iterations (the draft/verify two-phase iterations? the
  HpuEagleProposer runs in-worker though), (b) the drafted metric counts the
  proposals for a SUBSET of the requests (the make_spec_decoding_stats returns None
  when num_draft_tokens=0 — the non-drafting request-steps are invisible), (c) the
  bench aggregate mixes the phases (demonstrated). **The zero-boot arbiter for the
  next session: a single STREAMING request among the 16-stream load — the inter-chunk
  cadence = the iteration cadence (the 113 ms chunks with ~1.8 tokens vs the 61.5 ms
  chunks with ~1 token), directly readable from the client-side chunk timestamps.**
- **THE ARBITER RUN (the zero-boot, chunk_cadence.py):** one streaming request among
  the 15 fillers, the client-side chunk timestamps: **the bs=1 gap 47.5 ms (the p50
  46.7 — the solo step ✓ the F4TS agrees); under load: the gap p50 72.7 / the mean
  79.9 (the p10 57.7, the p90 73.4 — TIGHT)**. **THE CHUNK COUNTS RESOLVE THE AL:
  the solo 66 chunks for the 128 tokens = the AL 1.94; the loaded 131 chunks for
  ~256 tokens = the AL ~1.95 at @16 — THE DRAFTS ARE ACCEPTED AT @16; there is NO
  draft-skipping** (the earlier "⅓ skip" inference was the artifacts compounding:
  the drafted-metric windows mixed with the phases + my cadence misattributions).
  The drafted metric (1.82) ≈ the chunk-count AL (1.95) within the slop ✓ honest.
- **THE CORRECTED STEADY PICTURE @16: the iteration 72.7-79.9 ms, the tokens/step
  ≈ 16 × 1.95 ≈ 31 → the STEADY aggregate ≈ 415-430 tok/s.** The bench's 254
  substantially UNDERSTATES the steady throughput (the ramp + the staggered-completion
  tail + the prefill steps dilute the wall-average by ~40%). The bench-aggregate
  comparisons remain valid RELATIVELY (the same dilution across the arms), but the
  absolute steady ceiling is much higher than the bench suggests. **The optimization
  target: the iteration 73-80 → the worker 71 (the launch 28 + the drain 27 + the
  post 15.6) + the send/API ~2-9 → Q21 (the drain → ~12) + the launch/sampler trims
  → the iteration ~45-50 ms → the STEADY @16 ≈ 600-690 tok/s.** The 800 bench-number
  needs the k=4/drafter AND the ramp/tail dilution accounted.
- **REOPENED → RESOLVED (the spec-scheduling census, the final boot):** the
  `[SPEC-TS]` census in the scheduler's spec branch (GLM_SPEC_SCHED_TS=1) counted
  **2720/2720 decode request-steps scheduled WITH the draft — arith0=0, no_draft=0**
  over the SPECCHK bench: **there is NO draft-skipping; the num_scheduled_spec_tokens
  arithmetic never strips the drafts; the drafted metric ≈ the census** (the drafted
  ~2600 over the same window ✓). **The drafted-metric-based cadence (113-129 ms) is
  RETRACTED — its "10-s window" assumption is unverified and it is the one
  inconsistent instrument** (it implies ~0.64× the census-confirmed proposals).
- **THE FINAL CONSISTENT PICTURE @16 (all instruments now agree):** the iteration
  **~72.5 ms** (the streaming 131 chunks / 9.5 s ✓ = the LOOP step_fn 61.5 + the
  worker post tail ~11 ✓), the AL **~1.95** (the chunk count 131 for ~256 tokens ✓ =
  the drafted metric 1.82-1.83 within the slop ✓ = the per-position 0.833 ✓), the
  per-request rate **~27 tok/s** (the streaming ✓ = the bench per_stream 21.8-25.4 ✓),
  **the instantaneous aggregate ≈ 415-430 tok/s** — the bench's 246-258 is the
  ramp/tail-diluted batch average (the dilution ~40%, uniform across the arms so the
  relative comparisons hold). **The optimization target: the iteration 72.5 → the
  worker 71 (the launch 28 + the drain 27 + the post 15.6) + the tail ~11 → Q21 (the
  drain → ~12) + the launch/sampler trims → the iteration ~45-50 ms → the STEADY @16
  ≈ 600-690 tok/s.** The measurement lessons: (1) never derive the cadence from the
  spec metrics' window; (2) the streaming chunk count IS the per-request step count
  (one delta per iteration); (3) the bench aggregate = the batch average, not the
  steady rate.
- **THE FULL RECONCILIATION (the LOOP stamps vindicated on the production boot):**
  the current production (the LOOP stamps active) under the bs=1 verification runs:
  **23 stamps / 30 s = 40.8 ms/iteration vs the throughput-derived 43.7 ms (the gen
  43.4 tok/s ÷ the AL 1.88) ✓✓ — the LOOP stamp is the true iteration.** The bench
  aggregate mystery resolved: **the bench requests finish at ~155 tokens (the EOS
  early-finishes: 2482 total / 16 = 155, not the 256 cap) — the 246-258 aggregate is
  the EOS-mix effect, NOT the ramp/tail dilution.** All the numbers now close: the
  per-request rate 21.8-27 tok/s ✓ (the streaming = the bench per-stream), the
  iteration 72.5-76.8 ms @16 ✓ (the streaming gap = the per-stream rate ÷ the AL),
  the AL ~1.95 ✓ (the chunk counts), the drafted metric's 0.66× proposals/iteration
  = the one open minor question (its own accounting; possibly the EOS-approaching
  requests not drafting). **The optimization math (unchanged, now fully sourced): the
  iteration 72.5 = the engine 61.5 + the tail 11 → Q21 (the drain 27→12) + the
  launch/sampler trims → the iteration ~45-50 ms → for the fixed-length workloads
  16 × 1.95/0.0475 ≈ 657 tok/s; for the EOS-mixed workloads proportionally less.**
- **THE PROPCHK CROSS-CHECK (the clean production, the 2-rep bench):** the rep2
  (the warm): 2472 tokens / 9.88 s = 250 tok/s aggregate, the requests EOS-finished
  at ~155 tokens; the streaming request (the 256-cap count prompt, the earlier run):
  the 131 chunks × 72.7 ms = 9.5 s = 26.9 tok/s — **the per-request rates AGREE
  (the 22-27 tok/s band) across the prompt types and the measurement methods**; the
  iteration 72.7 ms ✓ the LOOP 61.5 + the tail ✓. The drafted metric's AL 1.81-1.86 ✓
  the chunk-count 1.95 (the slop). **The one open item: the drafted throughput
  (140.89 tok/s) implies ~10.2 proposals/iteration vs the ~15.5 running — either the
  EOS-approaching requests skip the drafting, the metric's proposal accounting has a
  normalization quirk, or the padded rows interact — the minor open question, zero
  production impact (the AL and the rates are consistent everywhere else).**
- **THE SAME-SPAN RECONCILIATION (the census span, the final):** the drafted metric
  over the EXACT census window: **2626 proposals ≈ the census 2720 scheduled-with
  (the 3.5% = the window edges) ✓ the drafted metric ≈ the scheduling census.** BUT
  the streaming-derived proposals/s (the 16 requests × 13.7 iters/s = 220/s) exceeds
  the drafted metric (124-140/s) by ~1.6× — **the drafted metric under-counts the
  proposals vs the streaming-derived rate** (the observe_draft call-site condition —
  the generated_token_ids emptiness or the spec-row indexing; the resolution needs
  the observe_draft call-site instrument, one boot). **ZERO impact on the
  optimization path**: the AL 1.95 is confirmed directly by the chunk counts, the
  cadence 72.7 ms by the streaming + the LOOP stamps, and the Q21/trims math stands
  on those two alone. The drafted metric remains useful as the RELATIVE cross-check
  (the same bias across the arms).
- **THE STANDALONE HPU COMPILE BLOCKER (the operational finding, 11-13):** the
  standalone HPU processes **cannot compile ANY recipe on this host** (the
  synStatus 26 even for the trivial 4×4 matmul; persisting through the AIP reset,
  the driver reload, **the full host reboot**, the full production env, and the
  vllm import); **ONLY the vLLM serve full initialization compiles successfully.**
  **ALL standalone HPU benchmarks are blocked on this host** — the future
  measurements must use the vllm-process instruments (the stamps, the py-spy,
  the capture traces — all deployed). The BG-1b dequant-cache experiment
  (the first-vs-warm dequant timing) is deferred to the vllm-process instrument
  or the compile fix. The recovery note: after `kill_glm.sh`, the bare HPU tests
  fail identically — the compile failure is NOT the kill's artifact (it predates
  and postdates the reboot; the production vllm serve compiles fine throughout).
- **THE 0.64× QUESTION CLOSED (the census-span decode):** the census span's own
  metrics lines show **the window-average Running ≈ 7.9** (the snapshots 8-16 across
  the ramp/partial phases — the SPECCHK span included the capture-laden rep1 and the
  ramp), NOT the steady 15.5. The drafted throughput (140.89 tok/s) ÷ the window-
  average running (7.9-10) = the cadence 107-140 ms over THAT (instrumented, ramp-
  mixed) span — consistent with the census (2720/26 s = 153 ms instrumented). **The
  clean-production cadence (the streaming, the steady loaded phase) remains 72.5 ms
  ✓; the drafted metric is honest within its own windows — the discrepancies were
  the cross-span comparisons (my errors), not the metric.** The final measurement
  doctrine: compare the instruments ONLY within the same span, and prefer the direct
  per-request counts (the streaming chunks) over the derived rates.
- **THE PROMPT-DEPENDENCE MEASURED + THE PROGRAM CLOSED (the final, the zero-boot
  streaming arbiter with the usage):** the solo (the warm): **the count prompt
  (the predictable): the AL 1.95, 45.6 tok/s, the iteration ~42.9 ms; the paragraph
  prompt (the natural text): the AL 1.68, 37.4 tok/s, the iteration ~44.8 ms** —
  the MTP acceptance IS prompt-dependent (the predictable sequences draft well; the
  natural text 0.68). **The @16 bench arithmetic closes completely: the iterations
  288 (the LOOP × 32 = the F4TS counter/2 ✓ both counters agree) over the 22.9-s
  2-rep bench = 79.5 ms/iteration; the tokens 5028 ÷ 288 = 17.5/iteration ÷ the
  avg-running (the EOS drain ~10.6) = the AL ~1.65 ✓ = the paragraph AL 1.68 ✓; the
  per-request 157 tokens / (93 iterations × 79.5 ms) = 21.2 tok/s = the bench
  per-stream 21.8 ✓✓. THE DRAFTED METRIC IS HONEST (1.82 = the prompt mix); the
  LOOP stamps vindicated; the EOS drain explains the proposals ratio.** The final
  optimization math: the iteration 79.5 → the worker 71 (the launch 28 + the drain
  27 + the sampler 15.6) + the tail → Q21 (the drain → ~12) + the trims → the
  iteration ~50 ms → **the @16 aggregate ≈ 16 × 1.68/0.050 ≈ 537 tok/s (the natural
  text) / ~625 (the mixed)** — the earlier 657 estimate held within the slop. The
  measurement program is COMPLETE: every term sourced, every instrument
  cross-validated, every discrepancy resolved.

### T1c — the 17 ms gap localization (folds into T1a; no extra boot)

Extend the F4-TS stamps to the API-server side (the serving process): stamp (i) the
bench client's step-trigger (the /v1/completions chunk boundary), (ii) the API server's
receipt of the step output, (iii) the API server's send of the next-token chunk. The
17 ms "between the engine phases" then localizes to one of: the EngineCore→API-server
output transport (the ZMQ/IPC in the serve topology), the API-server output processing
(the per-step tokenization/detokenization + streaming write under the asyncio GIL), or
the API-server→EngineCore input cadence (the scheduler-output generation after the
output processing). The vLLM V1 serve path is: APIServer proc ↔ (ZMQ) ↔ EngineCore proc ↔
(SHM MessageQueue, `shm_broadcast`) ↔ WorkerProcs — **two process hops plus the SHM
broadcast per step**; each hop has a polling cadence. The stamps tell us which hop owns
the 17 ms, which picks the fix (L2 lists them).

**Deliverable of T1:** a one-page physics verdict (P-DEV / P-HOST / hybrid at bs=1 and
@16, with the GB/step + device-busy + gap-localization numbers), which re-ranks §4.
**Cost:** ~1-2 engineer-days (the stamps are small env-gated patches in the style of the
F4-TS; no accuracy impact). **Risk:** low (the F1-ASYNC lesson: measure before the
redesign).

---

## 4. The lever stack (commercial-only), ranked under each physics

Lever classes: **K** = kernel/device, **H** = host/engine, **S** = speculation, **C** =
concurrency/config. Each states its value under P-DEV and P-HOST.

### L1 (K, P1 under both) — the Q21 routing-aware fused MoE kernel (the agent's P1, sharpened)

The agent's program (in progress: bit-exact single-expert, −29% multi-expert A/B on real
weights, the E4M3 top-bin parity target settled) is the right one. The research
sharpenings:

1. **The GB model is a distinct-expert problem, not a pair problem.** At bs=1 the 8
   routed experts are (under balanced EP8) ~1 local expert/rank/layer → the stock path
   already streams ~1 expert/rank/layer; Q21's bs=1 GB saving is small (the 38.1 GB must
   include the non-expert weight streaming — the attention/MLA/KDA/lm_head weights
   re-read EVERY step — see the L1.3 note). At @16 the 128 (row,expert) pairs concentrate
   on ~10-25 distinct local experts/rank/layer (the T1b histogram) → Q21's GB saving is
   the pairs÷distinct ratio (~5-12×) on the expert term. **The A/B must report the
   per-rank GB/step, not just the t/s** — the t/s depends on the physics (T1), the GB is
   the invariant.
2. **The expert-placement interaction (EP8 skew).** The routing is learned; the
   per-rank local-expert hit distribution is NOT uniform (the hot experts). If the hot
   experts cluster on a few ranks, those ranks become the HBM-bound ones at @16 (the
   per-rank GB diverges from the mean). The T1b per-rank (rank 0 vs rank 7) stamp catches
   it. The mitigation is the expert placement (the RA-013 class): balance the hot experts
   across ranks (a static permutation of the expert-to-rank assignment — free at load
   time, the routing is per-token so the permutation is invisible to the router).
3. **The non-expert streaming is the OTHER 38.1 GB candidate.** The per-step re-read of
   the non-MoE weights (MLA projections, KDA, norms, lm_head) is ~5-6 GB/rank/step at TP8
   (DERIVED from the 321B parameter split — VERIFY against the T1a counter). If that is
   the bulk of the 38.1 GB, the Q21 expert cut leaves it untouched, and the non-expert
   term has its own lever: the weights are CONSTANT across steps — they can be staged in
   a persistent HBM-resident form with the dequant done ONCE at load (the "pre-converted"
   path the E4M3 analysis already established: the ×0.5/×2.0 scale fold at load time,
   then the steady-state dequant is a TPC in-register op with no BF16 intermediate —
   i.e. the Q21 kernel applied to the DENSE weights too, or simply: the dense weights in
   BF16-resident form (they fit: ~6 GB/rank) skipping the FP8 dequant entirely for the
   dense path). **This is the "dense-linear dequant-fusion" question (the agent's own
   RA-035 topic) — it is P1-adjacent, not a side quest: if the dense weights dominate the
   38.1 GB, fusing/skipping their dequant is worth as much as the MoE cut.**
4. **The MTP-verify-row interaction.** At k=1 the target verify runs 2 rows (the bonus +
   the draft); the routing is computed for BOTH rows (2×8 pairs); the Q21 kernel's union
   is over both rows' experts (≤16 pairs, ≤~8 distinct local/rank) — the kernel is
   unchanged (it is pair-driven), the GB model gains the second row's pairs. No new work;
   the A/B should include the MTP-on arm so the −29% is measured in the production shape.
5. **The TP4 variant.** EP4 → 32 local experts/rank; the per-rank expert HBM DOUBLES
   (the 16 → 32 experts) but the per-token routed pairs stay 8 (2 local/rank/layer at
   bs=1). The TP4 A/B reuses the same kernel (the rank count is a config); the GB model
   is per-rank (T1b-TP4, fold into the Stage-4 TP4 boot).

**License:** Apache-2.0 (the 1Cat reference kernel + the project's own DSV4.1 remap).
**Complexity:** HIGH (weeks — the agent's own class). **Risks:** the accumulate-order
drift (RESOLVED: the scale double-rounding, b9a62e4); the suite gate (the R003 class,
the 0.5 pp line). **Validation:** the bit-exact parity (done for single-expert), the
multi-expert A/B with the per-rank GB/step @ bs=1 AND @16 (MTP on), the suite, then the
t/s under the T1 physics verdict.

### L2 (H, P1 under P-HOST, P2 under P-DEV) — the bs=1 host-window program (the ~17 ms)

The target: the bs=1 step wall from ~23.8-44 ms (MTP off/on) toward the device floor
(~10-15 ms stock; ~7-12 ms with L1). The candidates, in the order the T1c stamps will
rank them (the V1 serve topology: APIServer ↔ZMQ↔ EngineCore ↔SHM-MQ↔ Workers):

1. **The API-server output cadence (the prime suspect for the 17 ms).** The V1 API server
   processes each step's output (the detokenization + the streaming chunk send) in the
   same asyncio loop that triggers the next step; the bench client's per-chunk RTT
   (TCP loopback + the client's read cadence) can sit INSIDE the step wall if the
   scheduler waits for the output to drain. Check: does the step wall include the
   client's inter-chunk gap? (T1c stamp (iii) vs (i).) The fix class: decouple the
   output drain from the step trigger (the output-processing thread, the existing
   `process_request_outputs_callback` async path — the V0 MQLLMEngine already has the
   async-socket overlap pattern; verify the V1 serve path has it ENABLED on our tree).
2. **The EngineCore↔APIServer IPC hop (ZMQ, one round trip per step).** The V1
   `EngineCoreProc` runs in its OWN process (the `run_engine_core` path); the APIServer
   talks to it over ZMQ. At bs=1 that is a full process hop per step: the input
   (the scheduler trigger) + the output (the token) both cross. Options: (a) the
   in-process engine core (`VLLM_ENABLE_V1_MULTIPROCESSING=off` — the env exists in our
   tree, the corpus shows the flag) removes the hop entirely at the cost of GIL sharing
   between the API server and the engine in one process (the trade is measurable, not
   theoretical: run it as an A/B arm); (b) if multiprocess is kept, the ZMQ socket
   options (the conflate/heartbeat, the HWM) and the polling cadence of both sides.
3. **The SHM MessageQueue broadcast (EngineCore→workers).** The `shm_broadcast`
   MessageQueue with a polling loop: the poll interval + the condition granularity set a
   floor on the input delivery latency (the worker's "Poller.poll 82%" is this loop).
   The 10% "input" time in both the engine and worker profiles suggests the queue
   round-trip is ~2-4 ms — the fix class: the busy-poll with a shorter interval (the
   power cost is negligible at 8 idle cores), or the eventfd/condition wakeup instead of
   the timed poll (the `MessageQueue` has a wait path — verify its wakeup latency with a
   10 µs-resolution micro-benchmark: send→receive on the empty queue, 10k samples).
4. **The GIL / thread placement.** The worker's execute_model holds the GIL for the
   metadata build (the D1 draft-dispatch leaves: op dispatch, mark_step,
   dynamic_quant, native SiluAndMul); the API server's asyncio loop holds ITS GIL. These
   are separate processes (no shared GIL) — but WITHIN the worker, the HPU op launches
   release the GIL only around the blocking calls; the metadata-build Python is
   GIL-bound CPU. The fix class is the L3a graphing (the metadata disappears into the
   graph) + the small fusions (the dynamic_quant + SiluAndMul are the D1-named leaves —
   the RA-011/T6-shape re-measurement is the agent's own P2).
5. **What NOT to do:** the F1-ASYNC class (the async-D2H redesign) is DROPPED with
   evidence (−5%) — do not re-attempt the output-path async without the T1c stamps
   showing the D2H is the binding term (it is not, under either physics: the D2H is the
   ~1.3 MB logits/hidden, <0.1 ms at PCIe speeds; the 17 ms is NOT the D2H).

**License:** all stack code (Apache-2.0). **Complexity:** MEDIUM (each candidate is
days, not weeks; the program is 1-2 weeks including the A/Bs). **Risks:** low (all
env-gated A/Bs; the in-process-engine-core arm is the only structural change — keep it
env-gated and reversible). **Validation:** the step wall (the F4-TS) before/after each
arm, at bs=1 MTP-on AND off, with the t/s as the final readout; the suite is NOT affected
(host-side changes are numerics-neutral — verify with the 4K/1K sanity, not the full
suite, unless a change touches the output path).

### L3 (H, P1 under P-HOST-@16, P2 under P-DEV-@16) — the @16 worker-host program (the 61 ms)

The 61 ms worker-host @16 (MEASURED, no residual) decomposes into three attack surfaces:

1. **The 42 ms "replay" term.** The T1b completion stamp splits it:
   - If it is SYNC-WAIT (the device is busy ~40 ms at @16 — the P-DEV-@16 case): the
     attack is the DEVICE (L1's GB cut + the L3.3 below); the host program contributes
     little here.
   - If it is LAUNCH-CPU (the graph launch descriptor build + the per-op staging for the
     16-row graph + the draft graph + the verify graph, all on the host): the attack is
     the GRAPH COUNT and the LAUNCH PATH. The B2 finding (the replay loop dominates,
     staging only 1.6%) points at the launch loop itself. Fix classes: (a) the
     single-graph capture (the agent's research question 1 — see L4.1, the pad-and-mask
     makes the MTP shape STATIC so the whole step is one graph); (b) the graph launch
     batching (the HPUGraph API's multi-graph replay, if the 1.24.1 SDK exposes a batched
     launch — check the `habanalabs-graph` API surface; the per-graph launch has a fixed
     cost ~1-3 ms that a batched launch amortizes); (c) the fewer-shapes consolidation
     (the RA-006 class: the bucketing strategy `pad` — 222 decode buckets with bounded
     padding exist in the tree, UNSET — the pad bucketing trades a little device padding
     for a much smaller graph population; at @16 the 16-row workload maps to ONE pad
     bucket instead of walking the 160-shape ladder).
2. **The 10.6 ms draft dispatch (17.4%).** The D1 leaves (op dispatch + mark_step +
   dynamic_quant + native SiluAndMul at 16 rows). The attack: (a) the GRAPHED draft
   (the draft chain captured per pad-bucket shape — the RA-004 design; the MTP-4
   private-storage pattern from the abotsis reference; the draft is a DENSE 1-layer
   model — it captures cleanly, no MoE inside, no rollback state in the draft graph
   (the KDA rollback is in the TARGET verify, not the draft)); (b) the leaf fusions
   (the dynamic_quant + SiluAndMul — the T6-shape, the agent's P2). Graphing the draft
   at 16 removes ~8-10 ms of the 10.6 (the dispatch is the Python op tree; the graph is
   one launch).
3. **The 6 ms per-row sampler (9.9%).** The sampler runs PER ROW at @16 (the 16-row
   sampling under the spec-verify). The attack: the BATCHED sampler (one sampling op
   call over the [16, ...] rows — the agent's research question 9; the sampler op
   itself is shape-generic, the per-row-ness is the PYTHON loop in the spec-decode
   verify path — verify in the source: the `spec_decode` verify's sampling call site; if
   it is a loop, vectorizing it is a days-class patch).

**License:** stack (Apache-2.0). **Complexity:** MEDIUM-HIGH (the single-graph capture
is the hard part — the R003 numerics regime applies to any new capture surface, the T12
design rules from RA-036 carry over verbatim). **Risks:** the capture-surface numerics
(the Q01v2 drift history: the captured-region compilation drifts +20% behind the golden —
the alignment work is the known cost); the pad-bucketing device padding (the padded rows
cost device — the 16→16 is zero padding, the 16→32 is the L5 trade). **Validation:** the
worker-host @16 (the D1-16 method) before/after; the t/s @16; the suite (the capture
surface changes REQUIRE the full suite — the R003 gate).

### L4 (S, P1 for 800@16 under BOTH physics) — the MTP@16 recovery (the −38% is engineering, not physics)

The MTP head is MEASURED at AL 1.88 @16-equivalent workloads (the 94% k=1 acceptance is
bs=1; the @16 acceptance is UNMEASURED — the agent's own unknown: "the MTP acceptance at
bs=16 with the levers"). The −38% @16 regression (MTP4-T0) is the DRAFT DISPATCH +
PER-ROW SAMPLER at 16 rows (the D1-16: 10.6 + 6 ms of the 61 ms worker host) OUTWEIGHING
the +0.88 tok/step. The physics: at @16 the step is 62 ms and the MTP delta is
+~17 ms host (the draft+sampler per-row) + a small device delta (the 2nd verify row) for
+0.88 tok/row/step → 258 × (1.88/1) × (62/79) ≈ 161 t/s — **the −38% is arithmetic on
the measured terms, and every term has a named attack surface (L3.2, L3.3).**

The program:
1. **Measure the AL@16** (the missing number; the bench_concurrent 16 + the /metrics
   acceptance extraction — the MTP4-T0 method at 16 streams). If the AL@16 ≈ 1.85-1.90
   (the acceptance is batch-invariant — the MTP head is per-request), the MTP@16 breakeven
   is: the MTP step cost ≤ 62 × 1.88 ≈ 116 ms... trivially true — the breakeven is the
   MTP overhead ≤ 0.88 × 62 ≈ 55 ms, which the current 17 ms ALREADY satisfies in
   ISOLATION. So why −38%?? **Re-derive before fixing:** the −38% must come from the MTP
   step being +17 ms AND the non-MTP @16 baseline (258) being measured at a DIFFERENT
   concurrency/ctx than the MTP run (the MTP4-T0 comparison arms — verify the
   workload-matching of the 258 vs the 161: same 16 streams, same ctx, same warm state).
   If the workloads match, the 17 ms overhead at a 62 ms step predicts 258 × 1.88 ×
   62/79 ≈ 371 t/s, NOT 161 — **the −38% is 2.3× worse than the D1-16 terms explain.**
   The gap (the ~35 ms of unexplained MTP@16 cost) is the REAL target: candidates — the
   capture/re-capture at the 16-row MTP shapes (the F3 tax is "neutral at @16" per the
   16-stream profile — but that profile was MTP-OFF? the F3-SCOPED line says "the
   16-stream profile shows the capture branch nearly empty (63/12004 = 0.5%)" — verify
   that profile was the MTP-ON 16-stream, else the MTP-on capture cost is unmeasured);
   the draft graph population at 16 (the seed-bucket expansion, the T12-TEST 277-graph
   class); the per-step SpecDecodeMetadata construction at 16 rows (the host Python over
   16 requests' token blocks — the D1 "meta 2 ms" may be the MTP-OFF meta, the MTP meta
   is larger); the 2-row verify's graph shape walk (the 16-row × 2 = 32-row verify maps
   to a DIFFERENT bucket than the 16-row — if the 32-row bucket is not pre-warmed, the
   first MTP step at 16 captures it (13-15 s!) and the re-capture tax at ctx growth
   (the F3 2.6 ms/step) applies to the MTP shapes specifically). **This is a
   measurement-first sub-problem (L4.0): the MTP-ON 16-stream worker-host decomposition
   (the D1-16 method, MTP on) + the capture-branch count at MTP-on 16.** The −38%
   decomposes into named terms only after that.
2. **Then fix the named terms** (each has an L3 attack surface: the draft graphing, the
   sampler batching, the 32-row pre-warm (the D003), the pad-bucketing (L3.1a), the
   F3-scoped ctx coarsening if the MTP shapes cross the ladder).
3. **The batch-conditional depth (the T17 generalization, commercial):** k=2 at bs=1-2
   (the MEASURED +21% @1), k=1 at ≥4 streams, k=0 (MTP off) if the L4.0 decomposition
   shows the MTP@16 can't be brought to ≤ +5 ms. The depth switch is a runner config
   keyed on the active-batch count (the scheduler knows it per step — a 2-line branch at
   the speculative-config level; the graphs for both depths must be pre-warmed). This
   keeps the bs=1 +21% (k=2) without paying it at @16. **The expected end state (after
   L4 + L3 + L1, 8 HPU):** @16 ≈ 258 × (L1: 1.1-1.4 physics-dependent) × (L3: the host
   61→~30-40 ms → 1.3-1.7) × (L4: AL 1.88 at ≤+5 ms cost → ×1.8) ≈ **600-1000 t/s** —
   the 800 target is INSIDE the envelope; the physics verdict (T1) picks which term owns
   the margin.

**License:** the MTP head (the target checkpoint — the project's baseline license), the
runner (Apache-2.0). **Complexity:** MEDIUM (L4.0 is a day; the fixes are the L3
surfaces). **Risks:** the MTP-on capture numerics (the R003 gate); the depth-switch
graph population (the boot time — the D003 pre-warm covers it). **Validation:** the
AL@16 (new), the worker-host @16 MTP-on (new), the t/s @16, the suite.

### L5 (C, P2 for both, the insurance policy) — the concurrency 16 → 32

The 258 t/s @16 was measured at the ns16 cap. Is the device headroom there for 32?
Evidence: the per-stream t/s drops 15.6 → 13.6 from 8 → 16 streams (−13%) — the
sublinear drop means the device is NOT saturated at 16 (a saturated device would show a
−50% per-stream drop). The 32-stream arithmetic (DERIVED, to be measured):
- **The KV/state HBM at 32:** the KDA state is per-REQUEST fixed: 34 KDA layers ×
  512 KB (TP8) ≈ 17 MB/req + the MLA KV: 32K ctx × the latent (the 512-dim class × 2B
  ≈ 32 MB/req at 32K) → ~50 MB/req × 32 ≈ 1.6 GB total ≈ 200 MB/card — SMALL against
  the 96 GB cards. The KV pool (1280 blocks/card, 0.82 occupancy at 16×32K) scales to
  32×32K at ~0.9+ occupancy — **the pool must be verified against the 32-seq demand
  (the 1280 blocks × the block-token size vs 32 × 32K — if the block is 2048 tokens,
  1280 × 2048 = 2.6M tokens/card ≫ the 1M total demand — the pool is NOT the wall; the
  0.82 occupancy was the GRAPH 25 GiB block eating the budget, not the KV)**.
- **The wall is the graph population + the capture time + the per-row host (L3).** The
  32-row verify bucket (the 16×2 MTP shape) + the 32-row non-MTP bucket: +2 shapes
  (the D003 pre-warm); the pad bucketing (L3.1a) makes the 16-32 range ONE pad bucket
  (the 32-row pad for 16-32 active — the device padding cost at 16-active is the 2×
  row-count on the attention/MLA rows — measure it, the padding-aware strategy's
  "bounded padding" is the design intent).
- **The per-row host at 32:** the L3 program (the draft graphing + the sampler batching)
  MUST land first — at 32 rows the per-row cost doubles (the 10.6 → ~21 ms draft, the 6
  → ~12 ms sampler UNFIXED) and eats the device headroom. **L5 is gated on L3.**
- **The expected value (DERIVED):** if the device at 32 is ~1.3-1.5× the @16 (the
  sublinear weight streaming — the weights are read ONCE per step regardless of rows;
  the row-linear terms are the KV/attention + the MoE pairs), the 32-stream aggregate
  ≈ 32/16 × (1/1.3-1.5) × 258-class ≈ 300-360 t/s MTP-off, ×1.8 MTP-on ≈ 540-650 —
  **the 32-concurrency is the insurance policy that keeps 800@16 (and the TP4 800)
  inside the envelope even if the L1/L3 gains come in at the low end.**
- **The ns32 config:** the max-num-seqs 32 + the 32-row pre-warm + the pad bucketing —
  a config + pre-warm change (the RA-003 adoption pattern, the suite re-run at the 32-
  seq accuracy gate — the D13 preemption-resume re-verify at 32, the T27-class battery
  from RA-036 carries over).

**License:** config (n/a). **Complexity:** LOW-MEDIUM (the pre-warm + config; the
accuracy gate is the real cost — the suite at 32 seqs). **Risks:** the boot time (the
+2-4 graph shapes × 13-15 s each — the pre-warm makes it a one-time cost); the capture-
OOM at 32 rows on the tighter TP4 (the TP4-FIX capture-OOM history — the TP4 32-row
bucket must be verified separately). **Validation:** the t/s @32, the per-stream, the
capture count, the suite, the TP4 32-row boot.

### L6 (S, THE 200-bs=1 path under P-HOST; the commercial DFlash2 replacement) — the self-trained drafter

**The strategic point:** the 200-bs=1 roofline needs ≥4.2 accepted/step at the (L2-
reduced) step. The MTP head is MEASURED at 2.68 at k=4 (the deep-position acceptance
DECAY: ~94% at position 1 collapsing to ~25% marginal at positions 2-4 — the intrinsic
limit of a 1-layer sequential autoregressive draft head; no config fixes it). The only
drafter class that clears 4.2 is the BLOCK-DIFFUSION class (all positions drafted in one
parallel pass + the selector) — and the only published GLM block-diffusion checkpoint is
the NC-licensed incoai one. **The method, however, is MIT:** z-lab/dflash (the paper
arXiv 2602.06036; "up to 6× lossless acceleration on Qwen3-8B, ~2.5× faster than
EAGLE-3"; the SGLang/vLLM integrations; the vLLM **Speculators** framework (Apache-2.0)
has a DFlash training/integration path per its docs). **Training our own block-
diffusion drafter on our own GLM-5.3-Flash weights is commercially clean** and reuses
the RA-036 substrate wholesale:

1. **The architecture is the incoai one (so the port drops in):** 5 layers, hidden 4096,
   32q/8kv GQA, head_dim 128, inter 12288, the 2-tap grouped dynamic conv, the top-16
   rank-256 selector, the 2048 sliding-window ring, block-8, the aux-hidden conditioning
   on the target layers [5,14,24,33,42] — the `HpuDFlash2Proposer` port (the 40%-done
   RA-036 work: the factory wiring, the `laguna_dflash` model class, the KDA rollback
   remap (B1), the platform gate (B2), the selector ops) is **architecture-defined, not
   checkpoint-defined** — a self-trained checkpoint with the same config.json shape loads
   through the identical code path. The B1 KDA-rollback remap (the hard gate) is the
   SAME work either way.
2. **The training recipe (the paper + the Speculators scaffolding):** generate the
   target's hidden states at [5,14,24,33,42] over a domain-matched corpus (the serving
   distribution: code + math + chat mix, 32K windows — the corpus the owner already uses
   for evaluation); train the 5-layer draft (~1-2B params, 2.2 GB) with the block-
   diffusion objective (the masked-block prediction over the 8-token block) + the
   selector head (the rank-256 bilinear scoring + the greedy walk). Scale: the incoai
   checkpoint is the same size class; a few days on a few 80-96 GB GPUs (the draft is
   tiny; the cost is the target-hidden generation over the 321B target — that is the
   multi-day part, on the G2 itself in low-priority slots or on any available GPU box).
3. **The quality gate (the RA-036 T15 method, reused):** the acceptance calibration on
   our stack at block {4,8} × bs {1,16} — **G1: AL ≥ 4.0 at bs=1** (the 200-bs=1
   requirement is AL ≥ 4.2 at the L2-reduced step; the 4.0 gate with the margin). The
   anchor: the incoai checkpoint measured 4.5-5.5 AL on the GLM-5.3 class (the inco day-
   0 GLM numbers 5.48-6.02 on the full GLM-5.3); the self-trained quality band is
   honestly **3.5-5.5** (the training-data quality + the Flash-vs-full-model difference
   are the uncertainty; the 3.5 lower bound is the MTP-class number — the gate catches
   it).
4. **The alternatives (ranked):** (a) **the EAGLE-3 head** (the better-studied training
   recipe, the Speculators-native) — but EAGLE is a DEEPER sequential chain (3+ layers,
   autoregressive draft) — on a host-paced stack the sequential draft dispatch is the
   exact cost class the MTP k=4 already showed (the host term grows with depth); the
   block-diffusion single-pass is the better fit for this stack's physics (one draft
   forward, one selector pass — the host dispatch is depth-INVARIANT). (b) **the DSpine
   drafter** (Tencent, Apache-2.0) — the "spine" drafter class; no GLM checkpoint
   (the RA-016 note) and its HPU path is unproven; it is the license-clean FALLBACK if
   the self-training quality misses G1. (c) **the MTP-ensemble** (the RA-036 T20: the
   multiple MTP heads) — the checkpoint has ONE MTP module; stacking it k times is the
   MTP-4 chain (already MEASURED at 2.68) — no headroom; REJECTED as a 200-bs=1 path.
5. **The bs=1 arithmetic (P-HOST, after L2 + L1):** the step = max(host_window_L2,
   device: target 1-row (~10-15 ms stock, ~7-12 with L1) + draft (2-4 ms) + 8-row verify
   (~12-20 ms — the weights are read once, the 7 extra rows are KV/compute-linear) ≈
   19-33 ms) → at the L2 host window ≤ 20 ms: the step ≈ 20-30 ms → **AL 4.2-5.0 /
   0.020-0.030 ≈ 140-250 t/s** — the 200 target is inside the envelope (the ESTIMATED
   band; the T15 calibration + the T1 physics set the point).
6. **The @16 arithmetic (why the drafter is NOT the @16 lever):** at @16 the 8-row
   verify × 16 requests = 128 verify rows/step — the device row-linear terms (the KV,
   the attention, the MoE pairs ×16) make the 8-row verify 3-5× the 1-row cost; the
   drafter's AL gain (×2.2) does not pay the verify-row cost (×3-5) at @16 — the MTP
   k=1 (2 rows) is the @16-optimal depth (the T17 batch-conditional depth, L4.3, makes
   this explicit: drafter at bs=1-2, MTP k=1 at 4-16, MTP off at 32 if needed).

**License:** the method (MIT, z-lab/dflash + the paper), the training scaffolding
(Apache-2.0, vLLM Speculators), the target weights (the project's baseline license),
the trained draft (a derivative of the target + the MIT method — commercially clean
under the same assumption the project already runs on). **Complexity:** the TRAINING
is the high part (weeks: the corpus + the hidden generation + the training + the
calibration); the PORT is the RA-036 substrate (the B1 remap remains the hard gate —
but it is REQUIRED for the drafter to run at all, so the investment is not stranded
either way). **Risks:** the quality risk (G1 may fail at 3.5-4.0 — the DSpine fallback
or the MTP-only end state); the training-infra availability (the owner's GPU access for
the hidden generation); the B1 remap (shared with the excluded incoai path — the
remap itself is license-neutral: it is OUR KDA machinery, not incoai code).
**Validation:** the G1 calibration (the RA-036 T15 method), the R003 suite (the T12
design rules), the bs=1 t/s at the L2 step, the @16 parity check (the drafter must not
REGRESS @16 — the batch-conditional depth keeps it out of the @16 path).

### L7 (the objective check) — the 4-HPU final-deployment roofline (the honest arithmetic)

The final deployment is 4 HPUs (TP4+EP4). The measured anchors: TP4 bs=1 41.3-45.1 t/s
(levers, no MTP), TP4 @16 180.8 t/s warm (Q14v2). The per-rank work DOUBLES vs TP8
(the TP shards half as many ranks); the per-card HBM is unchanged (96 GB); the total
pool HALVES (4×96 vs 8×96).

**The bs=1 200 at 4 HPUs:** the step = max(host, device). The host window is the same
class as TP8 (the engine pipeline is rank-count-independent — the L2 program applies
unchanged; expect the same ~17 ms gap, verify at the TP4 boot). The device at TP4: the
non-expert weights 2× (~10-12 GB/rank/step → 6-7 ms) + the MoE at EP4 (32 local
experts/rank; the bs=1 routed 8 experts → ~2 local/rank/layer × 25 MB × 45 ≈ 2.3 GB →
~1.3 ms stock; ~0.9 with L1) + the attention/KDA (~2-3 ms) → **~10-12 ms device stock,
~8-10 with L1**. So: **MTP k=1 (AL 1.88): step ≈ max(17, 12) + the MTP delta ≈ 20-24 ms
→ 1.88/0.022 ≈ 85-95 t/s. MTP k=2 (AL ~2.5, the +21% class): ~110-125 t/s. The
drafter (AL 4.2-5.0, the 8-row verify device ~19-25 ms at TP4 — the row-linear KV/
attention doubles): step ≈ max(17, 25) ≈ 25-30 ms → 4.2-5.0/0.027 ≈ 155-185 t/s.**
**Verdict: 200 bs=1 at 4 HPUs is at the EDGE of the physics for ALL commercial
options** — the drafter reaches 155-185 (the ESTIMATED band; it clears 200 only if the
L2 host window ≤ 15 ms AND the 8-row verify device ≤ 22 ms AND the AL lands ≥ 4.5 —
the optimistic corner). The honest recommendation: **treat 200-bs=1 as a 4-HPU
STRETCH target (155-185 expected) and 200 as an 8-HPU development target** (where the
same stack reaches 140-250, i.e. it clears 200 in the median case) — the owner should
decide whether the 4-HPU 200 is a hard line (in which case the L2 program must reach
the ≤15 ms window AND the drafter AL ≥ 4.5, both at the top of their bands) or a
stretch. The 800@16 at 4 HPUs (below) is the more tractable of the two 4-HPU targets.

**The 800@16 at 4 HPUs:** the step = 16 × AL / 800. At MTP k=1 (AL 1.88): the step
target = 37.6 ms. The current TP4 @16 step = 16/180.8 = 88.5 ms → the required cut is
2.35×. The lever stack at TP4: L1 (the MoE GB cut — the per-rank expert set is 2× but
the Q21 union is pair-driven; expect the same −25-30% on the expert term: the TP4 @16
expert term ~2× the TP8 → the absolute saving is LARGER in GB but the relative step cut
is similar, ~1.2-1.4×) + L3 (the worker host @16 61→~30-40 ms — the host is rank-
independent in its Python cost: ~1.5-1.8×) + L4 (the AL 1.88 at ≤+5 ms: ×1.8 on the
token count) + L5 (the 32-concurrency insurance, ×1.2-1.5 on the aggregate IF the
per-row host is fixed first). **Combined (DERIVED, ESTIMATED): 180.8 × (L1: 1.2-1.4)
× (L3: 1.5-1.8) × (L4: 1.8 AL) × (L5: 1.2-1.5) ≈ 800-2300 t/s** — the 800 target is
INSIDE the envelope with margin (the median ≈ 1200-1500); the drafter is NOT needed
for 800@16 at 4 HPUs (the row-count argument, L6.6). **The 4-HPU risk concentration is
the L3 host program (the single largest ×) and the capture-OOM history at TP4 (the 32-
row + MTP shapes on the tighter budget — the L5 TP4 gate).**

**The one-sentence end state (commercial, 4 HPU):** the MTP head (AL 1.88, k=1 at ≥4
streams, k=2 at bs=1-2) + the Q21 kernel + the host program (the L2 17 ms + the L3 61
ms) + the 32-concurrency pad bucketing reaches **~800-1500 t/s @16** and **~90-125 t/s
bs=1** on 4 HPUs; the self-trained block-diffusion drafter (L6, the MIT method on our
weights, the RA-036 substrate) adds the **~155-200+ t/s bs=1** on the 8-HPU development
box (the 4-HPU stretch). The 200-bs=1 AT 4 HPUs is the one objective that may not clear
its line — it is flagged for the owner's stretch/hard decision, not silently met.

---

## 5. Responses to the agent's ten research questions (current_bottleneck.md §11)

| # | The agent's question | Response (this document) |
|---|---|---|
| 1 | Single-graph capture of the whole step (verify + draft + sample)? | **YES, and the pad-and-mask is the enabler (L3.1a).** The dynamic-shape blockers the agent names (the variable batch, the MTP acceptance variability) are EXACTLY what the pad bucketing dissolves: the 16-32 active range maps to ONE 32-row pad bucket, the MTP 2-row verify maps to the same bucket's row range, the acceptance variability is inside the PADDED rows (the padding-lane identity guard, the RA-006/RA-036-T12 design). The sampler stays host-side unless the sampler-op-in-graph question (their Q7, my L3.3 note) clears. This is the biggest single host lever (the 42 ms replay + the 10.6 ms draft in one stroke) — sequence it AFTER T1 (it is the P-HOST-@16 master) and the R003 gate applies (new capture surface). |
| 2 | In-register dequant GEMV (the block-FP8 TPC layout)? | **DONE for the experts (the Q21 bit-exact, L1); the DENSE weights are the open half (L1.3)** — the 38.1 GB question decides how much of the weight streaming is dense vs expert; if dense dominates, the pre-converted-resident dense path (the ×0.5/×2.0 fold at load + the TPC in-register dequant, or the BF16-resident dense) is the L1-adjacent P1. |
| 3 | AR → reduce-scatter + all-gather (sequence parallel)? | **HOLD.** The AR volume at 16 rows is small (the activation exchange ~128 KB/layer class); the B4 estimate (−50% host floor) assumed the AR boundaries dominate the launch count — the F4-TS per-phase data (the T1c) will show the AR's actual share. If the T1c shows the AR-boundary launches are a named term, this becomes P3; if not, the launch-count win is captured cheaper by the graphing (L3.1a). Do not spend the sequence-parallel complexity before the measurement. |
| 4 | KDA recurrent-state fused with the MLA attention? | **P3 (the launch-count class).** The per-layer launch reduction is real but small against the 42 ms replay / 10.6 ms draft terms; it is the T13-class fusion (the RA-036 conv-chain fusion carries the method). Sequence after L1/L3. The hybrid-layer fusion also touches the KDA spec path (the rollback state) — the R003 gate applies; keep it separate from the Q21 integration. |
| 5 | MXFP4 (2× the weight streaming) vs the block-FP8? | **The highest-ceiling device lever, GATED on accuracy (0.5 pp).** If the 38.1 GB is weight-dominated (the T1a verdict), MXFP4 halves the streaming (the 23 ms → ~12 ms class under P-DEV) — the biggest single device win available. The accuracy gate is the binding constraint (the E2M1 mantissa loss at the routed experts — the Q21 E4M3 top-bin analysis is the precedent for the format-semantics work; the MXFP4 E8M0 block scales need the same parity treatment). Sequence: after the T1 verdict (if P-DEV) and after the Q21 E4M3 is in production (the MXFP4 A/B reuses its harness). The ECF8 (the exponent-concentration lossless E4M3, the RA-017 deep-dive) is the NO-accuracy-risk middle path if it clears the TPC layout constraints (the RA-017 §9 recipe). |
| 6 | The MTP draft + target verify in one graph (the abotsis private-storage)? | **YES — it is L3.1a + L4 (the single-graph capture).** The private-storage draft cores (the abotsis `_private_copy` pattern) solve the draft-state isolation; the pad-and-mask solves the shape; the R003 numerics regime (the RA-036 T12 rules) is the gate. This is the @16 host master under P-HOST-@16. |
| 7 | The minimum host work per step below 5 ms for 16 streams? | **The L2+L3 program targets it; the 5 ms is the right bar.** The post-graphing worker host @16 = the graph launch (the 1-3 ms class) + the metadata (the L3.1a moves most of it into the graph) + the output D2H (the <0.5 ms) + the sampler-if-host-side (the L3.3) → the 5 ms is reachable after L3; the bs=1 analog is the L2 window (the 17 ms → ≤5 ms is the aggressive end of the L2 band — the 200-bs=1 stretch depends on it, L7). |
| 8 | The routing entirely on-device (the router → top-8 in-kernel)? | **Already the design (the Q21 is the routing-AWARE kernel — the routing is INSIDE the fused op, L1).** The separate host-routing question is moot if the Q21 lands (the router logits → sigmoid → top-8 → the pair dispatch is the kernel's prologue, on-device). The residual host term is the ROUTING DECISION for the SCHEDULER (the block-table updates) — the F2 piggyback pattern (the draft_token_ids RPC collapse) is the template if the routing metadata needs a host round trip (it does not, if the kernel is self-contained — verify at the integration). |
| 9 | The per-stream host cost batched (one host call for N streams)? | **YES — the L3.3 (the sampler) is exactly this; extend it to the KV/slot updates and the draft preparation (the L3.2 graphing absorbs the draft; the KV-update batching is the scheduler-output path — the T1c stamps will show if it is a named term).** The vectorized-sampler patch is the days-class first step (verify the per-row loop in the spec-decode verify source). |
| 10 | The v0.30.0+ stack (the registrar collision fixed)? | **HOLD as a schedule item, not a decode lever.** The B7 collision is RESOLVED on our tree (the load-order recipe — the bottleneck table's struck-through row); the stack upgrade is the clean long-term fix but it is a months-class program (the image, the requalification, the R003 suite re-run) with NO decode gain of its own — the decode levers (L1-L6) are all stack-version-independent. Sequence it when the kernel track (the Q21 + the L6 selector ops) hits the op-lib wall again, not before. |

**The net from the table:** the agent's ten questions resolve into FOUR programs (the
Q21+dense-dequant device track = their Q2/Q5/Q8 + my L1; the single-graph + host track =
their Q1/Q6/Q7/Q9 + my L2/L3; the MTP@16 recovery = my L4; the drafter = my L6) plus
three HOLDs (the Q3 AR, the Q4 fusion, the Q10 stack) — which is the §6 do-first.

---

## 6. The do-first (for the implementation agent, in order)

1. **T1a + T1c (one boot, ~1 day):** the bs=1 step physics — the three device stamps
   (the launch return, the completion proxy, the per-rank HBM bytes/step) + the
   API-server-side stamps (the 17 ms localization). **Output: the P-DEV/P-HOST verdict
   at bs=1 + the 38.1-GB answer.** Everything else ranks on it.
2. **T1b (one boot, ~1 day):** the @16 physics — the per-rank HBM @16 + the
   replay-term split (launch-CPU vs sync-wait) + the routing-union histogram (the Q6
   extension: the per-step distinct-local-expert count, 100 steps). **Output: the
   P-DEV/P-HOST verdict at @16 + the Q21 GB model ground truth.**
3. **L4.0 (one boot, ~half day):** the MTP-ON 16-stream worker-host decomposition (the
   D1-16 method, MTP on) + the capture-branch count at MTP-on 16 + the AL@16.
   **Output: the −38% decomposition (the named terms) — the L4 fix list.**
4. **L1 (continue, in progress):** the Q21 integration to the A/B — add the per-rank
   GB/step readout to the A/B (the invariant), the MTP-on arm, the L1.3 dense-weight
   question (the 38.1 GB split from T1a decides the dense-dequant scope).
5. **L2 (after T1a/c, if P-HOST at bs=1):** the 17 ms program — the API-server-cadence
   arm first (the prime suspect), the in-process-engine-core A/B arm, the SHM-queue
   wakeup micro-benchmark.
6. **L3 + L4 (after T1b + L4.0, the @16 master track):** the draft graphing (the
   L3.2/L4 single-graph with the pad bucketing), the vectorized sampler (the L3.3), the
   32-row + MTP-shape pre-warm (the D003), the batch-conditional depth (the L4.3).
7. **L5 (after L3):** the ns32 + pad bucketing + the 32-seq accuracy gate + the TP4
   32-row boot check.
8. **L6 (the decision gate, after T1a):** IF the P-HOST verdict holds at bs=1 AND the
   owner confirms the training-infra access — start the self-trained drafter (the
   hidden-state generation over the domain corpus is the long pole; the port substrate
   is the RA-036 40% — the B1 KDA-rollback remap proceeds in parallel either way, since
   it is the drafter's hard gate and license-neutral). The G1 calibration (the AL ≥ 4.0
   at bs=1) is the go/no-go for the drafter's production path.

**The sequencing logic:** 1-3 are all MEASUREMENT (2.5 days total, no risk) and each
re-ranks 4-8; 4 is already in flight; 5-7 are the host+MTP track (the 800@16 path —
valuable under both physics); 8 is the bs=1-200 path (conditional on the verdict + the
owner's infra decision). Nothing in 4-8 depends on the excluded DFlash2 checkpoint.

---

## 7. RA-036 status (the superseded program, the surviving substrate)

- **The DFlash2 production path is SUPERSEDED** by the owner's license ruling (2026-11-12):
  the incoai checkpoint (CC BY-NC-ND 4.0) is excluded from production. RA-036's T15-T30
  acceptance/deployment theories are MOOT for the incoai checkpoint.
- **The surviving substrate (kept, referenced by L6):** (a) the proposer port machinery
  (the factory wiring, the `laguna_dflash` model class, the ops, the platform gate —
  the agent's e432f01 work, ~40%+ and now booting to engine init); (b) the B1 KDA-
  rollback remap design (the hard gate — license-neutral, OUR KDA machinery); (c) the
  T15 calibration method (the AL grid, the go/no-go gate — reused verbatim for the
  self-trained drafter); (d) the T12 R003 design rules (the capture-surface numerics
  regime — reused by L3.1a's single-graph capture); (e) the T27 rollback-correctness
  battery (the acceptance-0/preemption/page-crossing tests — the drafter's adoption
  gate); (f) the HBM/state arithmetic (the T4/T22 tables — the per-slot KDA state, the
  pool sizing).
- **The incoai checkpoint on the server** (`/data/models/GLM-5.3-Flash-DFlash2`, 2.2 GB):
  it may remain for RESEARCH use (the acceptance measurement as a calibration reference
  for the self-trained drafter — "how good does a well-trained GLM drafter get on this
  stack" is exactly the anchor the G1 gate needs) but MUST NOT ship in production and
  MUST NOT be used as the production drafter. Recommend the owner confirm the research-
  only use is acceptable under the NC-ND terms (research use of NC weights is the
  license's intent; publication/production is what NC blocks — the owner's call).

---

## 8. Research log (this round)

**GaudiGrimoire (2026-11-13):**
| Query | Result used |
|---|---|
| `vllm v1 EngineCore worker multiproc executor ZMQ pipe Condition wait process pool message queue decode step latency` | the vllm-fork `v1/executor/multiproc_executor.py` (the V1 topology: the SHM `MessageQueue` broadcast for the SchedulerOutput, the per-worker response MQ, the `WorkerProc` poll loop, the `POLLING_TIMEOUT_MS`/`EXECUTE_MODEL_TIMEOUT_S` constants) + `v1/engine` docs (the `VLLM_ENABLE_V1_MULTIPROCESSING` in-process-engine flag — the L2.2 A/B arm) + the V0 `MQLLMEngine` (the ZMQ PULL/PUSH + the `use_async_sockets` overlap pattern — the L2.1 reference) — **the basis of the L2 candidate list** |
| (carried, rev 12.1) | the 1Cat DFlash2 machinery (the `HpuDFlash2Proposer`, the env family, the `_validate_hpu_dflash2_config` Qwen-gate) — the L6 substrate reference; the abotsis KDA spec path (the candidate slots, the L≤8 tests) — the B1 remap + the L6 verify substrate; the Strata dflash (the block 2..8 knob, the acceptance 2.83-3.26/2.9) — the L6 quality anchor context |

**Web (2026-11-13):** the z-lab/dflash repo + project page + arXiv 2602.06036 (the
method, the MIT license, the "6× on Qwen3-8B, 2.5× faster than EAGLE-3" claim, the
SGLang/vLLM integrations); the vLLM **Speculators** docs (the DFlash training/
integration path — the Apache-2.0 training scaffolding for L6); the z-lab published
DFlash checkpoints (the size class reference: the b16 block, the Coder-30B-A3B draft).

**Server audit (2026-11-13):** the fork through **e432f01 11-13 00:34** — the DFlash2
port progress (the boot reaches engine init with the selector weights loading; the 10-
failure iteration ledger in the updated IMPLEMENTATION-FEEDBACK-DFLASH2.md; the
registry + `qwen3_dflash2` model + factory fixes; the checkpoint config corrections;
the op.cpp dtype validate widened to fn+fnuz) + the accuracy result artifacts committed
(the 11 result JSONs: the f1async/f2piggy/iso_revert/neutral/nolevers/nonrecip/pinned/
postreboot/seamfix/vparse/vparse2 arms). The current_bottleneck.md bottleneck table
(ranks 1-5: the Q21, the MTP draft dispatch, the replay internals, the per-row @16, the
AR) + the unknowns list (the per-layer device time, the host/device overlap, the HBM
BW during the MoE read, the MTP acceptance @16) + the ten research questions — all
incorporated above.

**T1b run (2026-11-13, the implementation agent):** the @16 physics boot executed per
§3 T1b — the routed-union histogram captured (16.1 distinct local experts/rank/layer
mean → the Q21 expert-byte cut 2.24×, the total 43.6 → ~22.5 GB), the completion-stamp
split delivered, the @16 bench parity 253–258 tok/s reproduced (the earlier 205–211 was
the custom-script artifact — corrected). Artifacts: `benchmarks/accuracy/results/t1b_*`
+ `t1b_summary.json`. The instrumentation (the env-configurable histogram window + the
GLM_STEP_TS runner stamps) is deployed default-inert. Full results: §3 T1b.

**T1b corrections + L2.1 (2026-11-13, later):** (1) the stamp-delta AL method is
invalid (the duplicated F4TS counter increments) — the engine's own spec metrics give
**AL@16 = 1.82: the acceptance HOLDS at @16**; the MTP@16 wash is the 32-row step cost.
(2) The @16 step ≈ 116 ms matches the T1B stamp total (115.2): **the host ≈ 89 ms
(launch 65 + post 24), the device drain ≈ 26 ms → P-HOST-@16 by direct split** — the
bytes are structural but the device completes them in a burst and idles behind the
host; the @16 master lever is the L3 host program, with Q21 shrinking the device burst
26→~12 ms. (3) **L2.1 (the API-cadence arm) CLEARED at bs=1**: the end-to-end wall
13.52 s / 512 tok vs the stamps → the step 49.6 ms of which the sample_tokens 47.5 —
the API/output overhead ≈ 2 ms (4%); consistent with the T1a P-HOST verdict. (4) Ops:
the EngineCore was killed by a stray SIGUSR1 probe; restored from boot_f4ts.sh. The
L2 next arms: the in-process-engine-core A/B (`VLLM_ENABLE_V1_MULTIPROCESSING=off`,
one boot) and the SHM-wakeup micro-benchmark — but the L2 upside at bs=1 is bounded by
the ~2 ms API term; the L3 launch-path attack (65 ms) is the bigger prize.

**License statement:** appended below (the owner's 2026-11-12 commercial-use
instruction; this document is commercial-only by construction — every lever names its
license class; the single NC component, the incoai checkpoint, is research-anchoring
only per §7 and excluded from production).

---

## Licensing and commercial-use statement (owner instruction, 2026-11-12: "make sure that
all you write is allowed for commercial use")

**These documents.** All text in these documents is original analysis by the independent
research architect, produced for this project. The owner may use the documents and their
conclusions commercially. These documents contain no license to, and grant no rights in,
any third-party material.

**Third-party material referenced or quoted — license status (verified 2026-11-12 against
the upstream repositories / HF cards):**

| Source | License | Use in these documents | Commercially usable |
|---|---|---|---|
| 1CatAI/1Cat-vLLM-Gaudi | Apache-2.0 | short code excerpts, identifiers, mechanism descriptions — attributed | YES (attribution retained) |
| JChen-318/gaudi2-single-card-strata (Strata) | MIT | config values, mechanism descriptions, acceptance facts | YES |
| mw00/project-maya | MIT | production-config facts (the DFlash2-GLM 148.6 t/s run) | YES |
| z-lab/dflash (+ arXiv 2602.06036) | MIT (+ public paper) | the L6 method reference (the block-diffusion drafter, the training objective, the selector) | YES |
| vllm-project/speculators | Apache-2.0 | the L6 training scaffolding reference (the DFlash integration path) | YES |
| Tencent/DSpine | Apache-2.0 (with third-party-component exceptions) | the L6 fallback drafter class | YES |
| vLLM / vllm-gaudi / HabanaAI stack (incl. the vllm-fork in the corpus) | Apache-2.0 | the L2/L3 engine-topology references (the V1 multiproc, the SHM MessageQueue, the in-process-engine flag) | YES |
| NatTuck/g2-opt-manual | **NO LICENSE FILE (all rights reserved)** | facts and measurements only + short attributed phrases — **no code reproduced** | facts freely usable; phrases are internal citations only |
| zeyuyang8/ecf8 | **NO LICENSE FILE (all rights reserved)** | method description + repo-layout facts (RA-017) — **no code reproduced** | facts/method freely usable; no code dependency |
| inco.ai DFlash2 blog (2026-08) | copyrighted, no open license | factual numbers (acceptance, recall, oracle gap) | facts freely usable |
| incoai/GLM-5.3-Flash-DFlash2 (checkpoint) | **CC BY-NC-ND 4.0 — non-commercial, no derivatives** | **config values only (factual data); research-anchoring use only per RA-037 §7 — EXCLUDED from production** | **NO for production — owner's ruling applied (2026-11-12: excluded)** |
| Intel/Habana official docs (habana-docs, SDK headers, 1.24.1) | Intel reference documentation | vendor API sentences quoted with attribution (tier B) | YES — standard citation of vendor documentation |

**Assurances:** (1) no model weights or checkpoint content appear in these documents;
(2) no code from the unlicensed sources is reproduced; every code excerpt keeps its
open-source attribution; (3) verbatim quotation is limited to vendor API documentation,
short attributed phrases, and open-licensed code; (4) the incoai DFlash2 checkpoint is
EXCLUDED from the production path by the owner's 2026-11-12 ruling (the research-
anchoring use in RA-037 §7 is the owner's separate call); (5) if any document is
published externally, re-review the two "NO LICENSE FILE" rows.
