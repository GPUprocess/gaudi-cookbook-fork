// SPDX-License-Identifier: Apache-2.0
//
// f8_unpk_probe — the fp8 load + convert + dual-group unpack RUNTIME probe.
// in:  FP8 [256] — the arbitrary distinct values
// out: BF16 [256] — the [0..128) = the UNPACK_EVEN (the expected in[2i]),
//                   the [128..256) = the UNPACK_ODD (the expected in[2i+1])
void main(tensor in, tensor out) {
    const int5 is = get_index_space_offset();
    const int5 c = {is[0] * 256, 0, 0, 0, 0};
    const minifloat256 p = v_f8_ld_tnsr_b(c, in, SW_UNPACK | SW_UNPCK_8_TO_16);
    const bfloat128 b = v_convert_f8_to_bf16_b(p);
    const bfloat128 e = v_bf16_mov_dual_group_unpack_b(b, 0xFFFFFFFF, 0, 0, 0, 0, SW_UNPACK_EVEN_LANES, (bfloat128)0, 1);
    const bfloat128 o = v_bf16_mov_dual_group_unpack_b(b, 0xFFFFFFFF, 0, 0, 0, 0, SW_UNPACK_ODD_LANES, (bfloat128)0, 1);
    v_bf16_st_tnsr({is[0] * 128, 0, 0, 0, 0}, out, e);
    v_bf16_st_tnsr({is[0] * 128 + 128, 0, 0, 0, 0}, out, o);
}
