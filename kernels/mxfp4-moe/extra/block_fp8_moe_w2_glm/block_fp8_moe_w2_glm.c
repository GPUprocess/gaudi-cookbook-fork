// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w2_glm v8 — THE PRE-QUANTIZED FP8 ACT (the stock-semantics match).
//
//   y[t, row] = sum_k s2[e_k, row] * sum_g as[t, k, g] * sum_{i in g} actq[t, k, i] * W2[e_k, row, i]
//
//   in0: w2        E4M3 [E_LOCAL, H, I]     TPC dims {I, H, E}
//   in1: w2_scales F32  [E_LOCAL, H]        TPC dims {H, E}   (ONE scale per row)
//   in2: act_q     E4M3 [T, K, I]           TPC dims {I, K, T} (the per-token-group-128 quantized)
//   in3: act_s     F32  [T, K, I/128]       TPC dims {I/128, K, T} (the dequant scales: amax/448)
//   in4: topk_ids  I32  [T, K]              TPC dims {K, T}   (the LOCAL ids, -1 = the non-local)
//   out0: y        BF16 [T, H]              TPC dims {H, T}
void main(
    tensor w2,
    tensor w2_scales,
    tensor act_q,
    tensor act_s,
    tensor topk_ids,
    tensor y,
    int npts
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int I = get_dim_size(act_q, 0);
    const int T = get_dim_size(act_q, 2);
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
            for (int row = 0; row < 128; ++row) {
                const int row_global = hb * 128 + row;
                float64 row_acc; row_acc = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    const int e = expert_ids[k];
                    if (e < 0 || e >= get_dim_size(w2, 2)) continue;
                    float128 acc1; acc1.v1 = 0; acc1.v2 = 0;
                    float128 acc2; acc2.v1 = 0; acc2.v2 = 0;
                    for (int p = 0; p < IP; ++p) {
                        int5 ac1 = {p * 256, k, t, 0, 0};
                        int5 ac2 = {p * 256 + 128, k, t, 0, 0};
                        const minifloat128 qa1raw = v_f8_ld_tnsr_b(ac1, act_q, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat128 a1 = v_convert_f8_to_bf16_all_b(qa1raw);
                        const minifloat128 qa2raw = v_f8_ld_tnsr_b(ac2, act_q, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat128 a2 = v_convert_f8_to_bf16_all_b(qa2raw);
                        int5 coord = {p * 256, row_global, e, 0, 0};
                        const minifloat256 pk = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat256 wall = v_convert_f8_to_bf16_all_b(pk);
                        acc1 = v_bf16_mac_acc32_b(wall.v1, a1, acc1);
                        acc2 = v_bf16_mac_acc32_b(wall.v2, a2, acc2);
                    }
                    const float dot1 = v_f32_reduce_add(acc1.v1 + acc1.v2);
                    const float dot2 = v_f32_reduce_add(acc2.v1 + acc2.v2);
                    const float s = s_f32_ld_g(gen_addr((int5){row_global, e, 0, 0, 0}, w2_scales));
                    const float as1 = s_f32_ld_g(gen_addr((int5){p * 2, k, t, 0, 0}, act_s));
                    const float as2 = s_f32_ld_g(gen_addr((int5){p * 2 + 1, k, t, 0, 0}, act_s));
                    row_acc = row_acc + dot1 * (as1 * s) + dot2 * (as2 * s);
                }
                float128 s128; s128.v1 = row_acc; s128.v2 = 0;
                int5 yc = {row_global, t, 0, 0, 0};
                v_bf16_st_tnsr_partial(yc, y, v_convert_f32_to_bf16_all_b(s128), 0, 0);
            }
        }
    }
}
