void main(tensor w13, tensor w13_scales, tensor x, tensor topk_ids, tensor raw) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int NB = get_dim_size(w13, 1) / 128;
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int nb = ip % NB;
        const int t = ip / NB;
        int5 ca = {0, t, 0, 0, 0};
        const bfloat128 a = v_bf16_ld_tnsr_b(ca, x);
        int5 cb = {128, t, 0, 0, 0};
        const bfloat128 b = v_bf16_ld_tnsr_b(cb, x);
        float128 acc; acc.v1 = 0; acc.v2 = 0;
        acc = v_bf16_mac_acc32_b(a, b, acc, (0) << 1, 1, 0);
        // THE DE-INTERLEAVE: the acc = [p0, p2, ..., p126 | p1, p3, ..., p127]:
        // the convert down-convert with the lane placement:
        // the acc.v1 (the 64 even-product f32s) -> the output's even lanes;
        // the acc.v2 (the 64 odd-product f32s) -> the odd lanes (via the income merge):
        const bfloat128 ev = v_convert_f32_to_bf16_b(acc.v1, 0, 0, (bfloat128)0, 1);
        const bfloat128 nat = v_convert_f32_to_bf16_b(acc.v2, 1, 0, ev, 1);
        int5 oc = {nb * 128, 0, t, 0, 0};
        v_bf16_st_tnsr(oc, raw, nat);
    }
}
