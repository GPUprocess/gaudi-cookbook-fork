// f8dump_probe — dumps the fp8 load + ALL_LANES convert of w13[0, 0, 0..255].
// in:  FP8 [8, 512, 512]  (the w13-shaped; the TPC dims {H=512, 2I=512, E=8})
// out: BF16 [1, 256]      — the out[0, 0..127] = the convert .v1, the out[0, 128..255] = the convert .v2
void main(tensor w, tensor out) {
    const int5 is = get_index_space_offset();
    const int5 c = {is[0] * 256, 0, 0, 0, 0};
    const minifloat256 p = v_f8_ld_tnsr_b(c, w, SW_UNPACK | SW_UNPCK_8_TO_16);
    const bfloat256 b = v_convert_f8_to_bf16_all_b(p);
    const int5 o1 = {is[0] * 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o1, out, b.v1);
    const int5 o2 = {is[0] * 128 + 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(o2, out, b.v2);
}
