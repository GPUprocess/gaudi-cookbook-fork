// mult_probe — the bf16 element-wise multiply: a * b
// in:  BF16 [2, 512]: the row 0 = the a, the row 1 = the b
// out: BF16 [1, 256]: the out[0, 0..127] = the a*b (the first 128), the out[0, 128..255] = the a (the echo)
void main(tensor in, tensor out) {
    const int5 ca = {0, 0, 0, 0, 0};
    const bfloat128 a = v_bf16_ld_tnsr_b(ca, in);
    const int5 cb = {128, 0, 0, 0, 0};
    const bfloat128 b = v_bf16_ld_tnsr_b(cb, in);
    const bfloat128 ab = a * b;
    const int5 o1 = {0, 0, 0, 0, 0};
    v_bf16_st_tnsr(o1, out, ab);
    const int5 o2 = {128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o2, out, a);
}
