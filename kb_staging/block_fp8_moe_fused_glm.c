// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_fused_glm — Q21/RA-002 step 2: routing-aware fused multi-expert
// MoE kernel for the GLM-5.3-Flash decode path (Gaudi 2 TPC, tpc-clang -march=gaudi2).
//
//   y[T,H] = Σ_k w_k · (W2[e_k,H,I] · SiLU(W13[e_k,2I,H] · x[T,H] /s13[e_k]) /s2[e_k])
//
//   Replaces the stock MoE dispatch chain:
//     batch_gemm ×1,496 + dequant-fused + moe_multiplexer ×1,002 + routing glue
//     ≈ 18,200 recipe nodes (89% of the 20,485-node recipe)
//   with 2 TPC kernel launches per layer (the w13 pass + the w2 pass).
//
// Design: Q21-STEP2 v1 — the skeleton with the interface + the data flow.
// The TPC intrinsics follow the block_fp8_gemv_glm.c pattern (the VLM staging,
// the 128-row n-blocks, the 256-value K-chunks, the fp32 accumulation).
//
// Index space: (T/8) × (H/128) points.
//   - dim0: h-block (128 output columns per point)
//   - dim1: token-group (8 tokens per point)
// Each point: for the 8 tokens × 8 topk experts = 64 (token, expert) pairs,
// computes the w2 GEMV for the 128 output columns, reading the w13 output
// from VLM (staged by the w13 pass) and the w2 FP8 weights from HBM.

#ifndef MOE_FUSED_KCH
#define MOE_FUSED_KCH 32
#endif
#ifndef MOE_FUSED_TOPK
#define MOE_FUSED_TOPK 8
#endif
#ifndef MOE_FUSED_TOKENS_PER_POINT
#define MOE_FUSED_TOKENS_PER_POINT 8
#endif

// The intermediate activation after the w13 GEMV + SiLU: [T, I] in VLM.
// Staged by the w13 kernel (pass 1); consumed by the w2 kernel (pass 2).
__local__ bfloat128 v8_act[TOKEN_TILE * I_CHUNKS];

void main(
    tensor w13,         // FP8 [E_LOCAL, 2I, H] — the stacked expert w13 weights
    tensor w13_scales,  // F32 [E_LOCAL, 2I/128, H/128]
    tensor w2,          // FP8 [E_LOCAL, H, I] — the stacked expert w2 weights
    tensor w2_scales,   // F32 [E_LOCAL, H/128, I/128]
    tensor x,           // BF16 [T, H] — the input activations
    tensor topk_ids,    // int32 [T, TOPK] — the global expert ids per token
    tensor topk_weights,// F32 [T, TOPK] — the routing weights per token
    tensor y            // BF16 [T, H] — the output
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int T = get_dim_size(x, 0);
    const int H = get_dim_size(x, 1);
    const int I = get_dim_size(w2, 2);   // the moe intermediate
    const int E_LOCAL = get_dim_size(w13, 1) / (2 * I);  // the stacked dim

    const int hb = is[0];   // the h-block index
    const int tg = is[1];   // the token-group index

    // For each token in the group:
    for (int t_local = 0; t_local < MOE_FUSED_TOKENS_PER_POINT; ++t_local) {
        const int t = tg * MOE_FUSED_TOKENS_PER_POINT + t_local;
        if (t >= T) break;

        // Read the topk expert ids and weights for this token
        // (the H2D or the direct tensor read depending on the residency)
        int expert_ids[MOE_FUSED_TOPK];
        float expert_ws[MOE_FUSED_TOPK];
        for (int k = 0; k < MOE_FUSED_TOPK; ++k) {
            int5 idc = {t, k, 0, 0, 0};
            expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
            expert_ws[k] = s_f32_ld_g(gen_addr(idc, topk_weights));
        }

        // Accumulate the output for the 128 h-columns of this h-block
        float128 acc;
        acc.v1 = 0; acc.v2 = 0;

        for (int k = 0; k < MOE_FUSED_TOPK; ++k) {
            const int e = expert_ids[k];  // the global expert id
            const int e_local_idx = e;    // the local index (the single-rank or the offset-adjusted)

            // The w2 GEMV for this expert: y_partial[h_block] = W2[e, h_block, :] · act[t, :]
            // The dequant: the FP8 → the BF16 with the block scales (the in-flight)
            for (int ic = 0; ic < I / 256; ++ic) {
                // the w2 scale: [E, H/128, I/128] — the block scale for this (hb, ic)
                int5 sc = {ic * 2, hb, e_local_idx, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales));

                // the w2 FP8 load: [E, H, I] — the rows hb*128..hb*128+127, the cols ic*256..
                // (the same 256-value chunk pattern as the single-expert kernel)
                for (int half = 0; half < 2; ++half) {
                    int5 coord = {ic * 256 + half * 128, hb * 128, e_local_idx, 0, 0};
                    // the activation: from VLM (staged by the w13 pass) or the direct load
                    // v8_act[t_local * I_CHUNKS + ic] = the SiLU output for this (t, ic)
                    // TODO: the VLM staging interface between the w13 and the w2 passes

                    const minifloat256 p = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                    const bfloat128 v = v_convert_f8_to_bf16_b(p) * mult;
                    // the activation dot: act[t, ic*256+half*128 .. +127] · v
                    // TODO: the activation VLM read + the MAC
                    // acc = v_bf16_mac_acc32_b(v, act_chunk, acc);
                }
            }

            // the weighted accumulate: y[t, hb*128..] += expert_ws[k] × acc
            // TODO: the weighted accumulation into the output
        }

        // store the output for the (t, hb) block
        int5 yc = {hb * 128, t, 0, 0, 0};
        // v_bf16_st_tnsr(yc, y, yv);
    }
}
