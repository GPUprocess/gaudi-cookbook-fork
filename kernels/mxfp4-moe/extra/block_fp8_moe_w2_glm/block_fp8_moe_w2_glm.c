// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w2_glm v8b — THE PRE-QUANTIZED FP8 ACT (the stock-semantics match).
//
//   y[t, row] = sum_k s2[e_k, row] * sum_g as[t, k, g] * sum_{i in g} actq[t, k, i] * W2[e_k, row, i]
//
//   in0: w2        E4M3 [E_LOCAL, H, I]     TPC dims {I, H, E}
//   in1: w2_scales F32  [E_LOCAL, H]        TPC dims {H, E}   (ONE scale per row)
//   in2: act_q     E4M3 [T, K, I]           TPC dims {I, K, T} (the per-token-group-128 quantized)
//   in3: act_s     F32  [T*K, I/128]       TPC dims {I/128, T*K} (the dequant scales: amax/448; the flattened rows)
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
                    // THE W2 SCALE: the scales [E, H] -> the TPC dims {H, E}: {row_global, e}.
                    const float s = s_f32_ld_g(gen_addr((int5){row_global, e, 0, 0, 0}, w2_scales));
                    float64 row_acc_k; row_acc_k = 0.0f;
                    for (int p = 0; p < IP; ++p) {
                        int5 ac1 = {p * 256, k, t, 0, 0};
                        // THE F8 ACT: the 256-lane load + the convert (the weight idiom).
                        const minifloat256 qa = v_f8_ld_tnsr_b(ac1, act_q, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat256 qab = v_convert_f8_to_bf16_all_b(qa);
                        int5 coord = {p * 256, row_global, e, 0, 0};
                        const minifloat256 pk = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat256 wall = v_convert_f8_to_bf16_all_b(pk);
                        float128 acc1; acc1.v1 = 0; acc1.v2 = 0;
                        float128 acc2; acc2.v1 = 0; acc2.v2 = 0;
                        acc1 = v_bf16_mac_acc32_b(wall.v1, qab.v1, acc1);
                        acc2 = v_bf16_mac_acc32_b(wall.v2, qab.v2, acc2);
                        const float64 dot1 = v_f32_reduce_add(acc1.v1 + acc1.v2);
                        const float64 dot2 = v_f32_reduce_add(acc2.v1 + acc2.v2);
                        // THE PER-GROUP ACT SCALES: the groups p*2, p*2+1.
                        const float64 as1 = s_f32_ld_g(gen_addr((int5){p * 2, k + t * 8, 0, 0, 0}, act_s));
                        const float64 as2 = s_f32_ld_g(gen_addr((int5){p * 2 + 1, k + t * 8, 0, 0, 0}, act_s));
                        row_acc_k = row_acc_k + dot1 * as1 * (s * 1.0f) + dot2 * as2 * (s * 1.0f);
                    }
                    row_acc = row_acc + row_acc_k;
                }
                float128 s128; s128.v1 = row_acc; s128.v2 = 0;
                int5 yc = {row_global, t, 0, 0, 0};
                v_bf16_st_tnsr_partial(yc, y, v_convert_f32_to_bf16_all_b(s128), 0, 0);
            }
        }
    }
}
