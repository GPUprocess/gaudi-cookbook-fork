void main(tensor in, tensor out) {
    const int5 is = get_index_space_offset();
    const int5 ca = {is[0] * 128, 0, 0, 0, 0};
    const bfloat128 a = v_bf16_ld_tnsr_b(ca, in);
    const int5 cb = {is[0] * 128 + 128, 0, 0, 0, 0};
    const bfloat128 b = v_bf16_ld_tnsr_b(cb, in);
    float128 acc; acc.v1 = 0; acc.v2 = 0;
    acc = v_bf16_mac_acc32_b(a, b, acc, (0) << 1, 1, 0);
    const int5 o1 = {is[0] * 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o1, out, v_convert_f32_to_bf16_all_b(acc));
    const float64 s = v_f32_reduce_add(acc.v1 + acc.v2);
    float128 s128; s128.v1 = s; s128.v2 = s;
    const int5 o2 = {is[0] * 128 + 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o2, out, v_convert_f32_to_bf16_all_b(s128));
}
