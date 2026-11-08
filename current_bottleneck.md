# Current Bottlenecks — GLM-5.3-Flash on 8× Gaudi 2

**Date:** 2026-11-08
**Config:** the adopted stack (MTP k=1, TP8+EP8, max-num-seqs 32, the Q18 levers,
the tensor cache enabled, lin buckets, 32K window)
**Method:** the controlled scaling measurements at 1–64 streams (the steady-state
reps, warm-up discarded) + the device-byte model from the verified checkpoint
geometry + the step-anatomy study (GLM-5.3 MXFP4, same model family)

---

## 1. The step-time model (MEASURED + DERIVED)

The decode step time decomposes into a **fixed per-step cost** and a **linear
per-stream cost**:

| Streams (N) | Aggregate (t/s) | Step time = N × 1.85 / agg (ms) |
|---|---|---|
| 1 | **40.1** (39.4–41.4) | **45.1** |
| 4 | ~115 | ~64 |
| 8 | ~172 | ~86 |
| 16 | ~259 | ~114 |
| 32 | ~379 | ~156 |

Linear fit: **Step_time(N) = 40.7 + N × 3.8 ms**

### 1a. The fixed step cost: ~35–40 ms

| Component | Cost (ms) | Source |
|---|---|---|
| Host: ~322 recipe launches (with the tensor cache: the per-replay allocation removed; the input staging + the launch + the symbol resolution remain) | ~15–20 | MEASURED (the mxfp4 step-anatomy study 24 ms without the cache; the tensor cache reduces it; D1 pending on our FP8 tree) |
| Device: the MoE expert-weight streaming (38.1 GB/card ÷ 2.45 TB/s) | **15.5** | DERIVED from the verified checkpoint geometry (288 experts × 42 layers × 25.2 MB/expert/layer ÷ 8 ranks; cross-checked to the 328.3 GB checkpoint within 1.4%) |
| Device: the dense layers (3 × 12288×4096×3 FP8) | ~0.5 | DERIVED |
| Device: the MTP module (1 layer, the draft KV fill) | ~0.5 | DERIVED |
| Framework overhead (the scheduler, the sampling metadata, the KV sync) | ~4 | DERIVED (the residual) |
| **Total fixed** | **~35.5–40.7** | |

The fixed cost is **batch-independent**: the stock fused-MoE op reads every local
expert's weights at every step regardless of how many tokens are routed.

### 1b. The marginal per-stream cost: ~3.8 ms/stream

| Component | Per stream (ms) | Source |
|---|---|---|
| Host: the sampler + the scheduler + the logprob processing | ~1.5–2.0 | DERIVED (the step-anatomy study: the host work scales with rows; D1 pending) |
| Host: the MTP draft dispatch (eager, 1 layer + the shared head) | ~1.0 | DERIVED (the Q18 levers cut the sync + the enqueue; the draft is still 1 eager pass) |
| Device: the incremental GEMV compute (8 routes × 25.2 MB / stream, memory-bound) | ~0.5 | DERIVED (the weight reads are in the fixed cost; this is the arithmetic + the activation movement) |
| Device: the attention (the KV cache read scales with the batch) | ~0.3 | DERIVED |
| **Total marginal** | **~3.8** | |

---

## 2. The bottleneck hierarchy (ranked by the optimization priority)

### B1: The MoE expert-weight streaming — 15.5 ms/step (34% of bs=1)

The stock `mixture_of_experts` op reads **all 36 local experts' weights** from HBM
at every call, regardless of routing. At TP8+EP8: 288 ÷ 8 = 36 experts/rank ×
25.2 MB/expert/layer × 42 MoE layers = 38.1 GB/card/step. At 2.45 TB/s HBM peak:
15.5 ms of pure weight streaming.

**Effects:**
- The largest single device-side component of the step time
- **Batch-independent**: cannot be amortized by increasing the batch
- At bs=1, only 8 of 288 experts are needed per layer (2.8%) → **97% of the
  expert-weight HBM traffic is wasted** at bs=1
- **The 200 t/s single-stream objective is PHYSICALLY IMPOSSIBLE with the stock
  op**: 200 ÷ 1.85 = 108 steps/s → 9.3 ms/step, below the 15.5 ms weight-read
  floor. The only path: read only the routed experts (~2.1 GB/step at bs=1 →
  0.86 ms floor → 200 t/s becomes feasible with 10× margin).

**Levers:** the Q21 routing-aware kernel (RA-002: the fused in-register dequant
GEMV reading only the routed experts), the Q16 gather (tested: the gather+GEMM
decomposition lost 45% — the fused in-register approach is required).

### B2: The host-launch pipeline — ~15–20 ms/step (33–40% of bs=1)

The vllm-gaudi lazy-mode execution model issues ~322 recipe replays per decode
step. Each replay costs host time for the input staging + the launch + the
output collection. The tensor cache (ADOPTED) removed the per-replay allocation
but the staging and the launch remain.

**Effects:**
- Caps the bs=1 step (the device is idle, waiting for the host)
- Caps the aggregate at every batch size
- The AIP utilization of 0–24% during decode is the direct signature
- At 16 streams: the host cost is per-step → it limits the batch scaling

**Levers:** RA-001 (ADOPTED), RA-008 (the recipe-boundary reduction), RA-006-A
(the pad-and-mask verify graph), the Q21 kernel (one fused MoE op replaces
dozens of small ops).

### B3: The per-stream host cost — ~3.8 ms/stream (marginal)

Every additional concurrent stream adds ~3.8 ms to the step time:
- The sampler + the scheduler + the logprob processing
- The MTP draft dispatch (eager, 1 MLA + 1 MoE layer)
- The incremental device compute

**Effects:**
- The aggregate scales sub-linearly: 2× streams → +33% aggregate
- The per-stream throughput DROPS from 40.1 (bs=1) to ~10.3 t/s (bs=32)
- The 800 t/s @16 target: 16 × 3.8 = 61 ms of marginal host cost alone exceeds
  the 37 ms/step budget → the marginal cost must drop to ≤ 1 ms/stream

**Levers:** RA-004 (the graphed draft chain), the sampler optimization, the
scheduler optimization.

### B4: The all-reduce latency — ~3.4 ms/step (fixed)

162 all-reduce operations per decode step (one per layer per TP boundary), each
~20 µs in the small-message regime.

**Effects:** 3.4 ms of the fixed cost; each AR forces a mark_step() recipe
boundary.

**Levers:** RA-008 (the collective-boundary reduction), the HCCL optimization.

### B5: The EP imbalance — ~10 ms/step at high batch

The slowest EP rank's expert kernels cost 2× the mean rank (the balls-in-bins
statistics of the expert routing).

**Effects:** doubles the effective MoE kernel time at high batch; grows with the
batch.

**Levers:** the expert placement optimization, the EP4×TP2 layout (tested: -22%
on the stock stack — requires the custom kernels).

---

## 3. The step-time budget arithmetic

### Current (the adopted stack, ~40 t/s bs=1):

| Component | bs=1 (ms) | bs=16 (ms) | bs=32 (ms) |
|---|---|---|---|
| Host: the launches (fixed, with the tensor cache) | 15–20 | 15–20 | 15–20 |
| Host: the per-stream work (marginal × N) | 3.8 | 61 | 122 |
| Device: the MoE weight streaming (fixed) | 15.5 | 15.5 | 15.5 |
| Device: the attention + the compute (marginal × N) | 1.5 | 24 | 48 |
| Framework (fixed) | 4 | 4 | 4 |
| **Total (model)** | **~40.8** | **~120** | **~214** |
| **Measured** | **45.1** | **~114** | **~179** |

### The 800 t/s @16 target budget:

Required step time: 16 × 1.85 / 800 = **37 ms/step**

| Component | Current @16 | Target | Required reduction |
|---|---|---|---|
| Host: the launches | 15–20 | 5 | −67–75% |
| Host: the per-stream | 61 | 16 | −74% |
| Device: the weights | 15.5 | 15.5 | 0% (all 288 touched at bs=16) |
| Device: the attention + the compute | 24 | 12 | −50% |
| Framework | 4 | 3 | −25% |
| **Total** | **~114** | **~37** | **−67%** |

### The 200 t/s single-stream target budget:

Required step time: 1 × 1.85 / 200 × 1000 = **9.25 ms/step**

| Component | Current bs=1 | Target | Required reduction |
|---|---|---|---|
| Host: the launches | 15–20 | 2 | −87% |
| Host: the per-stream | 3.8 | 1 | −74% |
| Device: the MoE weights | 15.5 | **0.86** | **−94%** |
| Device: the attention | 1.5 | 0.5 | −67% |
| Framework | 4 | 2.5 | −55% |
| **Total** | **~40.8** | **~7.4** | **−82%** |

---

## 4. The bottleneck interplay and the optimization sequencing

The four bottlenecks are **not independent**:

1. **B1 (the weight streaming) and B2 (the host launches) are coupled**: the
   Q21 fused MoE kernel replaces ~50 small ops with ONE op, cutting both the
   launch count AND enabling the routed-expert-only weight read.

2. **B3 (the per-stream host cost) and the draft dispatch are coupled**: the
   MTP draft is the largest per-stream host component. RA-004 (the graphed
   draft chain) reduces it but requires the draft to be in-graph.

3. **B4 (the all-reduce) creates the launch boundaries**: each AR forces a
   mark_step(), splitting the lazy graph.

4. **B5 (the EP imbalance) scales with the batch**: the placement optimization
   requires the custom kernels to be effective.

**The optimal sequencing:**
1. RA-001 (**ADOPTED**: the tensor cache, +8-10% bs=1)
2. RA-008 (the recipe-boundary reduction) — independent
3. **Q21/RA-002 (the fused MoE kernel)** — addresses B1+B2+B4; requires the
   PT2 op-loading fix (the DSV4.1-proven mechanism, staged)
4. RA-004 (the graphed draft chain, k=2..4) — after the Q21 kernel
5. The EP placement — after the Q21 kernel

---

## 5. Summary for the theory/hypothesis agent

| # | Bottleneck | The share at bs=1 | The share @16 | The objective impact | The key lever |
|---|---|---|---|---|---|
| 1 | The MoE weight streaming | 34% | 14% | **THE blocker for 200 t/s** | The Q21 routing-aware kernel |
| 2 | The host-launch pipeline | 38% | 15% | The host floor at every batch | RA-001 (adopted) + RA-008 + the Q21 single-op |
| 3 | The per-stream host cost | 8% | 54% | The aggregate scaling bottleneck | RA-004 + the sampler optimization |
| 4 | The AR latency + the EP imbalance | 8% | ~5% | The recipe boundaries + the MoE wait | RA-008 + the expert placement |

**The 800 t/s @16 requires ALL FOUR bottlenecks addressed simultaneously.**
**The 200 t/s single-stream requires the Q21 kernel** (B1 is the physical
prerequisite). The Q21 kernel is the highest-priority next step because it
addresses B1+B2+B4 in a single change.
