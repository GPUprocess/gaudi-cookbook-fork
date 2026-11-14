// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w2_glm v2 — Q21/RA-002 step 2, pass 2: routing-aware multi-expert
// block-FP8 GEMV + accumulate for the GLM-5.3-Flash MoE decode path.
//
//   y[T, H] = Σ_k (W2[e_k, H, I] · act_preweighted[T, K, I] / s2[e_k])
//
//   The act is PRE-WEIGHTED by the routing weights in PyTorch before this
//   kernel (act *= topk_weights.unsqueeze(-1)), so this kernel does the
//   plain accumulate over the K=8 experts.
//
//   in0: w2           E4M3 [E_LOCAL, H, I]   the stacked expert w2 weights
//   in1: w2_scales    F32  [E_LOCAL, H/128, I/128]
//   in2: act          BF16 [T, K, I]          the PRE-WEIGHTED SiLU-activated intermediate
//   in3: topk_ids     int32 [T, K]            the global expert ids per token
//   out0: y           BF16 [T, H]             the MoE output
//
// Index space: T × (H/128) points (2-D).
//   - dim0: h-block (128 output h-rows per point)
//   - dim1: token index
// Each point: for the token t, the h-block hb (the 128 output h-rows):
//   for each row: for each of the K=8 routed experts: the GEMV over I
//   with the act[t, k, :] read from the HBM per chunk (the no VLM staging
//   — the act doesn't fit the 16 KB VLM for all the 8 experts).
//
// The vectorized patterns: the same as block_fp8_gemv_glm.c (the step 1).

void main(
    tensor w2,           // FP8 [E_LOCAL, H, I]
    tensor w2_scales,    // F32 [E_LOCAL, H/128, I/128]
    tensor act,          // BF16 [T, K, I]
    tensor topk_ids,     // int32 [T, K]
    tensor y             // BF16 [T, H]
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int I = get_dim_size(act, 0);    // 2048 (the TPC dim0 of act)
    const int T = get_dim_size(act, 2);    // the tokens (the TPC dim2 of act)
    const int H = get_dim_size(y, 0);      // 4096 (the TPC dim0 of y)
    const int E_LOCAL = get_dim_size(w2, 2);
    const int KB2 = I / 256;               // the 256-value i-chunks per expert: 8
    const int HB = H / 128;                // the h-blocks: 32

    const int hb = is[0];   // the h-block index (0..H/128-1)
    const int t = is[1];    // the token index
    if (t >= T) return;

    const ushort128 lid = read_lane_id_2b_b();

    // Read the K=8 routed expert ids for this token
    int expert_ids[8];
    for (int k = 0; k < 8; ++k) {
        int5 idc = {k, t, 0, 0, 0};
        expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
    }

    bfloat128 yv = (bfloat128)0.0f;  // the 128-lane output for the (t, hb)

    // The row loop: the 128 output h-rows in this h-block
    for (int row = 0; row < 128; ++row) {
        const int row_global = hb * 128 + row;  // the h-row in the H dim

        float128 acc;
        acc.v1 = 0; acc.v2 = 0;

        // The expert loop: the K=8 routed experts (the plain accumulate)
        for (int k = 0; k < 8; ++k) {
            const int e = expert_ids[k];

            // The dot over I in the 256-value chunks (the dequant + the MAC)
            for (int bk = 0; bk < KB2; ++bk) {
                // The block scale: w2_scales[e, row_global/128, bk*2]
                int5 sc = {bk * 2, row_global / 128, e, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales));
                sc[0] += 1;
                const bool second = bk * 256 + 128 < I;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales), 0, 0, second);

                // The act read: act[t, k, bk*256 .. bk*256+127] and [..+128 .. +255]
                // From the HBM directly (the no VLM staging — the act doesn't fit the VLM
                // for all the 8 experts simultaneously)
                int5 ac1 = {bk * 256, k, t, 0, 0};
                int5 ac2 = {bk * 256 + 128, k, t, 0, 0};
                const bfloat128 a1 = v_bf16_ld_tnsr_b(ac1, act);
                const bfloat128 a2 = v_bf16_ld_tnsr_b(ac2, act);

                // The FP8 weight load: w2[e, row_global, bk*256 .. +127]
                int5 coord = {bk * 256, row_global, e, 0, 0};
                const minifloat256 p1 = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat128 v1 = v_convert_f8_to_bf16_b(p1) * mult;
                acc = v_bf16_mac_acc32_b(v1, a1, acc);

                coord[0] += 128;
                const minifloat256 p2 = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16, 0, second);
                const bfloat128 v2 = v_convert_f8_to_bf16_b(p2) * mult2;
                acc = v_bf16_mac_acc32_b(v2, a2, acc);
            }
        }

        // The reduce-add → the lane insert for the row
        const float64 s = v_f32_reduce_add(acc.v1 + acc.v2);
        float128 s128;
        s128.v1 = s; s128.v2 = s;
        const bool128 la = v_u16_cmp_eq_b(lid, row, 0, (bool128){0}, 1, 0);
        yv = v_bf16_mov_vb(v_convert_f32_to_bf16_all_b(s128), 0, yv, la, 0);
    }

    // Store: y[t, hb*128 .. hb*128+127] — the 128-lane vector ✓
    int5 yc = {hb * 128, t, 0, 0, 0};
    v_bf16_st_tnsr(yc, y, yv);
}
