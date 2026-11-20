// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm v12 — Q21/RA-002 step 2, pass 1 (the CS3-verified fixes).
//
//   raw[T, K, 2I] = W13[e_k, 2I, H] · x[T, H]   (the block-FP8 dequant in-kernel)
//
// THE TWO ROOT-CAUSE FIXES (Q21-E31, the CS1/CS2/CS3 probe proof, 2026-11-19):
//  1. get_dim_size is TPC-ORDERED (innermost-first, 0-based): H = get_dim_size(x, 0).
//     The v11 read dim1 (= T) into H -> HP = 0 -> the h-loop never ran -> zeros.
//  2. tpcPoints() caps the index space at maxAvailableTpc (24): the kernel must
//     loop over its work SLICE (the mxk8 idiom) with npts passed as a scalar param.
//
//   in0: w13        E4M3 [E_LOCAL, 2I, H]   TPC dims {H, 2I, E}
//   in1: w13_scales F32  [E_LOCAL, 2I/128, H/128]  TPC dims {H/128, 2I/128, E}
//   in2: x          BF16 [T, H]             TPC dims {H, T}
//   in3: topk_ids   int32 [T, K]            TPC dims {K, T}
//   out0: raw       BF16 [T, K, 2I]         TPC dims {2I, K, T}
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
            // THE EP RULE: the ids arrive GLOBAL; the host pre-maps them to the LOCAL
            // ids with -1 for the non-local pairs. A -1 pair stores 0 (no loads) —
            // the between-math and the pass-2 accumulate then contribute 0, and the
            // per-rank partial y sums to the full y through the existing TP all-reduce.
            if (e < 0 || e >= get_dim_size(w13, 2)) {
                float128 z; z.v1 = 0; z.v2 = 0;
                const bfloat128 zb = v_convert_f32_to_bf16_all_b(z, SW_LINEAR);
                int5 oc0 = {col, k, t, 0, 0};
                v_bf16_st_tnsr_partial(oc0, raw, zb, 0, 0);
                continue;
            }
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
