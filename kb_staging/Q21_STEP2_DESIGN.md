# Q21 Step-2: Routing-Aware Fused MoE Kernel — Design Document

**Target: GLM-5.3-Flash on Gaudi 2, the decode path @16 streams.
**Goal: the 800 tok/s @16 (the current: 250-258 bench / ~415-430 steady).
**Mechanism: the recipe node-count reduction (the step is launch-bound at the
recipe level: 20,485 nodes × ~1.3 µs = the 26.6 ms device + the 28.3 ms replay).

## 1. The problem

The deployed MoE op (`VllmMixtureOfExpertsOp` with the BF16-resident views)
generates **~18,200 recipe nodes** (89% of the 20,485-node recipe):
- The batch_gemm × 1,496 (the expert-chunked BF16 GEMMs)
- The dequant-fused nodes (the BF16 views built once at load, the reads per step)
- The moe_multiplexer × 1,002 (the routing dispatch)
- The routing glue (the scatter, the gather, the casts, the weight-multiplies)

At ~1.3 µs/node (the launch-bound dispatch), this costs **~23.7 ms of the
27.3 ms device drain** and **~25-28 ms of the replay submission** — the
dominant share of both.

## 2. The solution: the 2-kernel fused MoE

Replace the entire MoE dispatch chain with **2 TPC custom kernels per layer**:

### The pass 1: the w13 GEMV + SiLU (the intermediate production)
- The input: x [T, H] BF16, the w13 FP8 [E_LOCAL, 2I, H], the w13_scales [E_LOCAL, 2I/128, H/128], the topk_ids [T, K]
- The output: act [T, K, I] BF16 (the SiLU-activated intermediate per expert)
- The index space: (T/TILE) points; each point: the TILE tokens × the K experts × the w13 GEMV [2I] + the SiLU
- The act store: the HBM (the global intermediate buffer [T, K, I] BF16 ≈ 32 × 8 × 2048 × 2 = 1 MB — tiny)

### The pass 2: the w2 GEMV + the weighted accumulate
- The input: the act [T, K, I] BF16, the w2 FP8 [E_LOCAL, H, I], the w2_scales, the topk_ids, the topk_weights
- The output: y [T, H] BF16
- The index space: (T/TILE) × (H/128) points; each point: the TILE tokens × the 128 h-columns; for each token: the K experts' w2 GEMV reading the act from the HBM, the weighted accumulate into y

### The node-count model

| Config | The recipe nodes | The replay submit | The device time |
|---|---|---|---|
| The stock (the current) | ~20,485 | 28.3 ms (1.4 µs/node) | 26.6 ms (1.3 µs/node) |
| The Q21 fused (2 kernels/layer) | ~2,800 | ~3.9 ms | ~3.6 ms |
| **The reduction** | **−86%** | **−86%** | **−86%** |

The post-Q21 iteration: the replay 3.9 + the device 3.6 + the sampler 15.6 +
the tail 11 ≈ **34 ms → 16 × 1.68/0.034 = ~790 tok/s @16** → the 800 target
with the sampler fusion (the `_accept_grid`: the ~2-3% further) → ~810-820.

## 3. The implementation phases

### Phase A: the correctness prototype (the standalone eager)
1. The PyTorch reference: the dequant → the bmm → the SiLU → the bmm → the
   weighted-sum in the eager PyTorch on the HPU (the correctness reference)
2. The TPC kernel: the pass 1 (the w13 GEMV + the SiLU) — the single-token
   correctness-first variant (the pattern from the block_fp8_gemv_glm.c)
3. The TPC kernel: the pass 2 (the w2 GEMV + the weighted accumulate)
4. The bit-exact parity vs the PyTorch reference (the tolerance: the fp32
   accumulation, the bf16 output — the same as the stock)

### Phase B: the performance optimization
1. The multi-token pipelining (the TILE tokens per point, the VLM staging)
2. The weight prefetch (the double-buffer the FP8 chunks)
3. The scale caching (the per-expert scales in the VLM)
4. The integration into the vLLM dispatch (the PT2 op + the FusedMoE hook)

### Phase C: the production integration
1. The PT2 custom op registration (the lazy pattern from the block_fp8_dequant_glm)
2. The vLLM dispatch: the FusedMoE.forward → the Q21 op (the env-gated fallback)
3. The accuracy suite (the 97.5% gate)
4. The A/B: the decode + the @16 + the prefill vs the adopted

## 4. The interface

```python
# The PT2 custom op schema
custom_op::block_fp8_moe_fused_glm(
    x: Tensor,              # BF16 [T, H]
    w13: Tensor,            # FP8 [E_LOCAL, 2I, H]
    w13_scale: Tensor,      # F32 [E_LOCAL, 2I//128, H//128]
    w2: Tensor,             # FP8 [E_LOCAL, H, I]
    w2_scale: Tensor,       # F32 [E_LOCAL, H//128, I//128]
    topk_ids: Tensor,       # int32 [T, K]
    topk_weights: Tensor,   # F32 [T, K]
    expert_offset: int,     # the EP rank's first global expert id
) -> Tensor                 # BF16 [T, H]
```

The two-pass structure: the PT2 op internally launches the pass 1 then the
pass 2 (the two TPC kernel submissions per MoE layer = the 84 launches for
the 42 layers, replacing the ~18,200 stock nodes).

## 5. The risk assessment

| Risk | Mitigation |
|---|---|
| The TPC register pressure (the K=8 experts × the VLM staging) | The HBM staging for the act intermediate (the tiny: 1 MB) |
| The Synapse compile of the fused multi-stage kernel | The incremental: the pass 1 + the pass 2 as the separate kernels first |
| The numerical parity (the fp32 acc vs the stock bf16) | The correctness-first: the fp32 accumulation (the same as the single-expert kernel) |
| The dynamic routing (the data-dependent expert ids) | The topk_ids read in-kernel; the expert weight addressing: the runtime offset (the TPC supports the dynamic addressing) |
| The integration complexity (the FusedMoE hook) | The env-gated dispatch (the GLM_Q21_FUSED=1); the fallback to the stock |

## 6. The expected impact

| Metric | The stock | The Q21 fused | The delta |
|---|---|---|---|
| The recipe nodes | 20,485 | ~2,800 | −86% |
| The replay submission | 28.3 ms | ~3.9 ms | −86% |
| The device drain | 27.3 ms | ~3.6 ms | −86% |
| The iteration @16 | 72.7 ms | ~34 ms | −53% |
| **The @16 aggregate** | **250-258** | **~768-790** | **+200%** |
| The bs=1 decode | 40-43 | ~47-50 | +15% (the replay reduction) |

**The 800 target: reachable with the Q21 step-2 + the sampler fusion.**
