// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w2_glm v7 — THE PER-ROW-SCALE CONTRACT (the runtime format).
//
//   y[t, row] = Σ_k s2[e_k, row] * Σ_i W2[e_k, row, i] * act[t, k, i]
//
//   in0: w2           E4M3 [E_LOCAL, H, I]   TPC dims {I, H, E}
//   in1: w2_scales    F32  [E_LOCAL, H]      TPC dims {H, E}   (ONE scale per row)
//   in2: act          BF16 [T, K, I]          TPC dims {I, K, T}  (the PRE-WEIGHTED)
//   in3: topk_ids     int32 [T, K]            TPC dims {K, T}   (the LOCAL ids, -1 = the non-local)
//   out0: y           BF16 [T, H]             TPC dims {H, T}
void main(
    tensor w2,
    tensor w2_scales,
    tensor act,
    tensor topk_ids,
    tensor y,
    int npts
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int I = get_dim_size(act, 0);
    const int T = get_dim_size(act, 2);
    const int H = get_dim_size(y, 0);
    const int IP = I / 256;
    const int nhb = H / 128;
    const int W = T * nhb;
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int w0 = ip * W / npts;
        const int w1 = (ip + 1) * W / npts;
        for (int wp = w0; wp < w1; ++wp) {
            const int t = wp / nhb;
            const int hb = wp % nhb;
            int expert_ids[8];
            for (int k = 0; k < 8; ++k) {
                int5 idc = {k, t, 0, 0, 0};
                expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
            }
            // THE PER-ROW PARTIAL STORES (the w13-proven idiom: the store reads ONLY
            // the lane 0 - the reduce's other lanes are the partial sums, never read).
            for (int row = 0; row < 128; ++row) {
                const int row_global = hb * 128 + row;
                float64 row_acc; row_acc = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    const int e = expert_ids[k];
                    if (e < 0 || e >= get_dim_size(w2, 2)) continue;  // the non-local pair
                    float128 acc; acc.v1 = 0; acc.v2 = 0;
                    for (int p = 0; p < IP; ++p) {
                        int5 ac1 = {p * 256, k, t, 0, 0};
                        int5 ac2 = {p * 256 + 128, k, t, 0, 0};
                        const bfloat128 a1 = v_bf16_ld_tnsr_b(ac1, act);
                        const bfloat128 a2 = v_bf16_ld_tnsr_b(ac2, act);
                        int5 coord = {p * 256, row_global, e, 0, 0};
                        const minifloat256 pk = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat256 wall = v_convert_f8_to_bf16_all_b(pk);
                        acc = v_bf16_mac_acc32_b(wall.v1, a1, acc);
                        acc = v_bf16_mac_acc32_b(wall.v2, a2, acc);
                    }
                    const float64 dot = v_f32_reduce_add(acc.v1 + acc.v2);
                    // THE PER-ROW SCALE: the scales [E, H] -> the TPC dims {H, E}: {row_global, e}.
                    const float s = s_f32_ld_g(gen_addr((int5){row_global, e, 0, 0, 0}, w2_scales));
                    row_acc = row_acc + dot * s;
                }
                float128 s128; s128.v1 = row_acc; s128.v2 = 0;
                int5 yc = {row_global, t, 0, 0, 0};
                v_bf16_st_tnsr_partial(yc, y, v_convert_f32_to_bf16_all_b(s128), 0, 0);
            }
        }
    }
}
