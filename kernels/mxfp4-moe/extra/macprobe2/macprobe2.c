// mac5 — the MAC with the SCALED operand (the w13 pattern): the acc = the MAC(a*mult, b, 0)
// in:  BF16 [2, 512]: the row 0 = the a, the row 1 = the b; plus the scales need... 
// the simplest: the mult: the constant 2.0 via the scalar; the a*mult vs the a: the distinguishable
void main(tensor in, tensor out) {
    const int5 ca = {0, 0, 0, 0, 0};
    const bfloat128 a = v_bf16_ld_tnsr_b(ca, in);
    const int5 cb = {128, 0, 0, 0, 0};
    const bfloat128 b = v_bf16_ld_tnsr_b(cb, in);
    const bfloat128 am = a * a;   // the a*a: the distinguishable from the a*b
    float128 acc; acc.v1 = 0; acc.v2 = 0;
    acc = v_bf16_mac_acc32_b(am, b, acc, (0) << 1, 1, 0);
    const int5 o1 = {0, 0, 0, 0, 0};
    v_bf16_st_tnsr(o1, out, v_convert_f32_to_bf16_all_b(acc));
}
