void main(tensor in, tensor out) {
    const int5 ca = {0, 0, 0, 0, 0};
    const bfloat128 a = v_bf16_ld_tnsr_b(ca, in);
    const int5 cb = {128, 1, 0, 0, 0};
    const bfloat128 b = v_bf16_ld_tnsr_b(cb, in);
    const bfloat128 a_echo = a;                     // the a echo (the load check)
    const bfloat128 ab = a * b;                     // the element-wise multiply
    float128 acc; acc.v1 = 0; acc.v2 = 0;
    acc = v_bf16_mac_acc32_b(a, b, acc, (0) << 1, 1, 0);
    const int5 o1 = {0, 0, 0, 0, 0};
    v_bf16_st_tnsr(o1, out, a_echo);                // the out[0, 0..127] = the a echo
    const int5 o2 = {128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o2, out, v_convert_f32_to_bf16_all_b(acc));  // the out[0, 128..255] = the acc
}
