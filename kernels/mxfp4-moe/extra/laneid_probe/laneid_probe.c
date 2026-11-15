void main(tensor in, tensor out) {
    const int5 is = get_index_space_offset();
    const ushort128 lid2 = read_lane_id_2b_b();
    const int5 oc = {is[0] * 128, 0, 0, 0, 0};
    v_bf16_st_tnsr(oc, out, as_bfloat128(lid2));
}
