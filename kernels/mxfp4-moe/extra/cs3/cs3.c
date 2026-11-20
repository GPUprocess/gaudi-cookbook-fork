// CS3: THE FIX CANDIDATE - the v11 compute with
//  (1) H from TPC dim0 (get_dim_size is TPC-ordered, innermost-first - mxk8-verified),
//  (2) the work-slice loop over npts scalar param (the mxk8 coverage idiom).
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
            const int t = (int)(w / (I2 * K));
            const int e = s_i32_ld_g(gen_addr((int5){k, t, 0, 0, 0}, topk_ids));
            float128 acc; acc.v1 = 0; acc.v2 = 0;
            for (int p = 0; p < HP; ++p) {
                int5 sc = {p * 2, col / 128, e, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));
                sc[0] += 1;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));
                int5 c1 = {p * 256, col, e, 0, 0};
                const minifloat256 pk = v_f8_ld_tnsr_b(c1, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat256 wv = v_convert_f8_to_bf16_all_b(pk);
                int5 x1 = {p * 256, t, 0, 0, 0};
                const bfloat128 xv1 = v_bf16_ld_tnsr_b(x1, x);
                int5 x2 = {p * 256 + 128, t, 0, 0, 0};
                const bfloat128 xv2 = v_bf16_ld_tnsr_b(x2, x);
                acc = v_bf16_mac_acc32_b(wv.v1 * mult, xv1, acc);
                acc = v_bf16_mac_acc32_b(wv.v2 * mult2, xv2, acc);
            }
            const float64 total = v_f32_reduce_add(acc.v1 + acc.v2);
            float128 tv; tv.v1 = total; tv.v2 = 0;
            const bfloat128 tb = v_convert_f32_to_bf16_all_b(tv, SW_LINEAR);
            int5 oc = {col, k, t, 0, 0};
            v_bf16_st_tnsr_partial(oc, raw, tb, 0, 0);
        }
    }
}
