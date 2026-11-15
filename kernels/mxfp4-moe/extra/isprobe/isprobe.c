void main(tensor in, tensor out) {
    const int5 is = get_index_space_offset();
    const int5 sz = get_index_space_size();
    const int5 ie = is + sz;
    float128 stamp; stamp.v1 = (float)is[0]; stamp.v2 = (float)ie[0];
    bfloat128 b = v_convert_f32_to_bf16_all_b(stamp);
    const int5 oc = {0, 0, 0, 0, 0};
    v_bf16_st_tnsr(oc, out, b);
}
