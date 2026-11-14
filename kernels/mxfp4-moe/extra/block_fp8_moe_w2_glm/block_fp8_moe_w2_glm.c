// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w2_glm — Q21/RA-002 step 2, pass 2: routing-aware multi-expert
// block-FP8 dequant + GEMV + weighted accumulate for the GLM-5.3-Flash MoE
// decode path (Gaudi 2 TPC, tpc-clang -march=gaudi2).
//
//   y[T, H] = Σ_k w_k · (W2[e_k, H, I] · act[T, K, I] / s2[e_k])
//
//   For each token t and each of its K=8 routed experts e_k:
//     1. Load the FP8 weight rows for expert e_k
//     2. Dequant: the BF16 = the FP8 × the block scale
//     3. GEMV: the dot with the act[t, k, :] (from the w13 pass)
//     4. Accumulate: y[t, :] += w_k × the dot result
//
//   in0: w2           E4M3 [E_LOCAL, H, I]   the stacked expert w2 weights
//   in1: w2_scales    F32  [E_LOCAL, H/128, I/128]
//   in2: act          BF16 [T, K, I]          the SiLU-activated intermediate (from the pass 1)
//   in3: topk_ids     int32 [T, K]            the global expert ids per token
//   in4: topk_weights F32  [T, K]             the routing weights per token
//   out0: y           BF16 [T, H]             the MoE output
//
// Index space: (T) × (H/128) points.
//   - dim0: h-block (128 output rows per point)
//   - dim1: token index
// Each point: for the token t, computes the 128 output h-rows accumulating
// over ALL K=8 routed experts.

#ifndef MOE_W2_KCH
#define MOE_W2_KCH 16  // the VLM act-chunks: I/128 <= 16 covers I = 2048
#endif
#ifndef MOE_W2_TOPK
#define MOE_W2_TOPK 8
#endif

__local__ bfloat128 v8_act[MOE_W2_TOPK * MOE_W2_KCH];  // the act[t, k, :] staged per expert

void main(
    tensor w2,           // FP8 [E_LOCAL, H, I]
    tensor w2_scales,    // F32 [E_LOCAL, H/128, I/128]
    tensor act,          // BF16 [T, K, I]
    tensor topk_ids,     // int32 [T, K]
    tensor topk_weights, // F32 [T, K]
    tensor y             // BF16 [T, H]
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int I = get_dim_size(act, 2);    // 2048
    const int T = get_dim_size(y, 1);      // the tokens
    const int H = get_dim_size(y, 0);      // 4096
    const int E_LOCAL = get_dim_size(w2, 2);
    const int IB = I / 128;                // the i-blocks: 16
    const int HB = H / 128;                // the h-blocks: 32
    const int KB2 = I / 256;               // the 256-value i-chunks: 8

    const int hb = is[0];   // the h-block index (0..HB-1)
    const int t = is[1];    // the token index
    if (t >= T) return;

    const ushort128 lid = read_lane_id_2b_b();

    // Read the K=8 routed expert ids and weights for this token
    int expert_ids[MOE_W2_TOPK];
    float expert_ws[MOE_W2_TOPK];
    for (int k = 0; k < MOE_W2_TOPK; ++k) {
        int5 idc = {k, t, 0, 0, 0};
        expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
        expert_ws[k] = s_f32_ld_g(gen_addr(idc, topk_weights));
    }

    // Stage the activation rows act[t, k, hb*128 .. hb*128+127] for all K experts
    // Each expert: the 128 values (the i-range matching this h-block's dot)
    // The act layout: [T, K, I] — the TPC dims: {I, K, T}
    for (int k = 0; k < MOE_W2_TOPK; ++k) {
        for (int ib = 0; ib < IB; ++ib) {
            int5 ac = {ib * 128, k, t, 0, 0};
            v8_act[k * MOE_W2_KCH + ib] = v_bf16_ld_tnsr_b(ac, act);
        }
    }

    // For each of the 128 output h-rows in this h-block:
    // compute the weighted sum over the K=8 experts
    for (int row = 0; row < 128; ++row) {
        const int row_global = hb * 128 + row;

        float128 weighted_acc;
        weighted_acc.v1 = 0; weighted_acc.v2 = 0;

        for (int k = 0; k < MOE_W2_TOPK; ++k) {
            const int e = expert_ids[k];
            const int e_local = e;
            const float w = expert_ws[k];

            float128 acc;
            acc.v1 = 0; acc.v2 = 0;

            // The dot over I in the 256-value chunks (the dequant + the MAC)
            for (int bk = 0; bk < KB2; ++bk) {
                // The block scale: w2_scales[e_local, row_global/128, bk*2]
                int5 sc = {bk * 2, row_global / 128, e_local, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales));
                sc[0] += 1;
                const bool second = bk * 256 + 128 < I;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales), 0, 0, second);

                // The FP8 weight load: w2[e_local, row_global, bk*256 .. bk*256+127]
                // The TPC coords: {I_offset, H_offset, E_offset} (the dim0 = the innermost = I)
                int5 coord = {bk * 256, row_global, e_local, 0, 0};
                const minifloat256 p1 = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat128 v1 = v_convert_f8_to_bf16_b(p1) * mult;
                acc = v_bf16_mac_acc32_b(v1, v8_act[k * MOE_W2_KCH + bk * 2], acc);

                coord[0] += 128;
                const minifloat256 p2 = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16, 0, second);
                const bfloat128 v2 = v_convert_f8_to_bf16_b(p2) * mult2;
                acc = v_bf16_mac_acc32_b(v2, v8_act[k * MOE_W2_KCH + bk * 2 + 1], acc);
            }

            const float dot = v_f32_reduce_add(acc.v1 + acc.v2);
            // the weighted accumulate into the shared accumulator
            const float128 weighted = (float128)(dot * w);
            weighted_acc.v1 += weighted.v1;
            weighted_acc.v2 += weighted.v2;
        }

        // The reduce + the store: y[t, row_global] = the weighted sum
        const float64 s = weighted_acc.v1 + weighted_acc.v2;
        float128 s128;
        s128.v1 = s; s128.v2 = s;
        const bool128 la = v_u16_cmp_eq_b(lid, row, 0, (bool128){0}, 1, 0);
        bfloat128 yv = v_bf16_mov_vb(v_convert_f32_to_bf16_all_b(s128), 0, (bfloat128){0}, la, 0);

        // Store all 128 rows: one vector store per lane group
        // (the same pattern as the single-expert kernel: the lane-selected insert
        //  accumulated across the row loop, then one store per h-block)
        // For the correctness-first: the scalar store per row.
        int5 yc = {row_global, t, 0, 0, 0};
        // Extract the row's value from the yv vector (the lane `row`)
        // For the scalar approach: store directly.
        s_bf16_st_g(gen_addr(yc, y), (bfloat16)s);
    }
}
