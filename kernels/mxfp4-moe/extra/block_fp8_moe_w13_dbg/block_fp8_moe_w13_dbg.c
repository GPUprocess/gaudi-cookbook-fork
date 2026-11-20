void main(tensor w13, tensor w13_scales, tensor x, tensor topk_ids, tensor raw) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int H = get_dim_size(x, 1);
    const int I2 = get_dim_size(w13, 1);
    const int HP = H / 256;
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int col = ip % I2;
        const int k = (ip / I2) % 8;
        const int t = ip / (I2 * 8);
        const int e = s_i32_ld_g(gen_addr((int5){k, t, 0, 0, 0}, topk_ids));
        const int blk = col / 128;
        if (blk == 0) {
            for (int p = 0; p < HP; ++p) {
                int5 xc = {p * 256, t, 0, 0, 0};
                const bfloat128 xv = v_bf16_ld_tnsr_b(xc, x);
                int5 oc = {p * 256, k, t, 0, 0};
                v_bf16_st_tnsr(oc, raw, xv);
            }
        }
        else if (blk == 1) {
            int5 wc = {0, col, e, 0, 0};
            const minifloat256 pk = v_f8_ld_tnsr_b(wc, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
            const bfloat256 wv = v_convert_f8_to_bf16_all_b(pk);
            int5 oc = {0, k, t, 0, 0};
            v_bf16_st_tnsr(oc, raw, wv.v1);
        }
        else if (blk == 2) {
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
            int5 oc = {0, k, t, 0, 0};
            v_bf16_st_tnsr(oc, raw, v_convert_f32_to_bf16_all_b(acc, SW_LINEAR));
        }
    }
}
