// dbgE v9-style: the ONE (t, col) per the point; the x load at the (t, col); the store at the raw[t, 0, col]
void main(tensor w13, tensor w13_scales, tensor x, tensor topk_ids, tensor raw) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int NB = get_dim_size(w13, 1) / 128;
    const int T = 4;
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int m = ip % 64;
        const int nb = (ip / 64) % NB;
        const int t = ip / (64 * NB);
        if (t >= T) continue;
        const int col = nb * 128 + m;
        int5 xv = {col, t, 0, 0, 0};
        const bfloat128 vv = v_bf16_ld_tnsr_b(xv, x);
        for (int k = 0; k < 8; ++k) {
            int5 yc = {col, k, t, 0, 0};
            v_bf16_st_tnsr(yc, raw, vv);
        }
    }
}
