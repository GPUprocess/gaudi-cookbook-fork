// xdump_probe — dumps x[0, 0..255] through the bf16 load/store path.
// in:  BF16 [4, 512]  (the x-shaped trigger; the TPC dims {512, 4})
// out: BF16 [1, 256]  — the out[0, 0..127] = the x[0, 0..127], the out[0, 128..255] = the x[0, 128..255]
void main(tensor x, tensor out) {
    const int5 is = get_index_space_offset();
    const int5 c1 = {is[0] * 128, 0, 0, 0, 0};
    const bfloat128 v1 = v_bf16_ld_tnsr_b(c1, x);
    const int5 o1 = {is[0] * 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o1, out, v1);
    const int5 c2 = {is[0] * 128 + 128, 0, 0, 0, 0};
    const bfloat128 v2 = v_bf16_ld_tnsr_b(c2, x);
    const int5 o2 = {is[0] * 128 + 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o2, out, v2);
}
