// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm v13 — THE PER-ROW-SCALE CONTRACT (the runtime format).
//
// The HPU FP8 path requantizes the checkpoint's block-FP8 to the per-row-scale FP8 at
// load: w13_weight [E_LOCAL, 2I, H] fp8 + w13_weight_scale_inv [E_LOCAL, 2I] f32
// (ONE scale per output row). The kernel: raw[t,k,col] = s13[e,col] * sum_h w13[e,col,h]*x[t,h].
//
// THE COVERAGE FIX (the slice loop + the npts param) and the EP -1 SENTINEL unchanged.
void main(tensor w13, tensor w13_scales, tensor x, tensor topk_ids, tensor raw, int npts) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int H = get_dim_size(x, 0);
    const int T = get_dim_size(x, 1);
    const int I2 = get_dim_size(w13, 1);
    const int K = get_dim_size(topk_ids, 0);
    const int HP = H / 256;
    const int W = T * K * I2;
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int w0 = ip * W / npts;
        const int w1 = (ip + 1) * W / npts;
        for (int w = w0; w < w1; ++w) {
            const int col = w % I2;
            const int k = (w / I2) % K;
            const int t = w / (I2 * K);
            const int e = s_i32_ld_g(gen_addr((int5){k, t, 0, 0, 0}, topk_ids));
            if (e < 0 || e >= get_dim_size(w13, 2)) {
                float128 z; z.v1 = 0; z.v2 = 0;
                const bfloat128 zb = v_convert_f32_to_bf16_all_b(z, SW_LINEAR);
                int5 oc0 = {col, k, t, 0, 0};
                v_bf16_st_tnsr_partial(oc0, raw, zb, 0, 0);
                continue;
            }
            float128 acc; acc.v1 = 0; acc.v2 = 0;
            for (int p = 0; p < HP; ++p) {
                int5 c1 = {p * 256, col, e, 0, 0};
                const minifloat256 pk = v_f8_ld_tnsr_b(c1, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat256 wv = v_convert_f8_to_bf16_all_b(pk);
                int5 x1 = {p * 256, t, 0, 0, 0};
                const bfloat128 xv1 = v_bf16_ld_tnsr_b(x1, x);
                int5 x2 = {p * 256 + 128, t, 0, 0, 0};
                const bfloat128 xv2 = v_bf16_ld_tnsr_b(x2, x);
                acc = v_bf16_mac_acc32_b(wv.v1, xv1, acc);
                acc = v_bf16_mac_acc32_b(wv.v2, xv2, acc);
            }
            float64 total = v_f32_reduce_add(acc.v1 + acc.v2);
            // THE PER-ROW SCALE: the scales [E, 2I] -> the TPC dims {2I, E}: {col, e}.
            const float s = s_f32_ld_g(gen_addr((int5){col, e, 0, 0, 0}, w13_scales));
            total = total * s;
            float128 tv; tv.v1 = total; tv.v2 = 0;
            const bfloat128 tb = v_convert_f32_to_bf16_all_b(tv, SW_LINEAR);
            int5 oc = {col, k, t, 0, 0};
            v_bf16_st_tnsr_partial(oc, raw, tb, 0, 0);
        }
    }
}
