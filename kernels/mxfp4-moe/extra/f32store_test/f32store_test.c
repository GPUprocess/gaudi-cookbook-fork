void main(tensor in, tensor out) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int5 c = {0, 0, 0, 0, 0};
    const bfloat128 v = v_bf16_ld_tnsr_b(c, in);
    const float128 vf = v_convert_bf16_to_f32_all_b(v);
    for (int tk = begin[0]; tk < end[0]; ++tk) {
        const int t = tk / 8;
        const int k = tk % 8;
        int5 o1 = {0, k, t, 0, 0};
        v_f32_st_tnsr_partial(o1, out, vf.v1, 0, 0);
        int5 o2 = {64, k, t, 0, 0};
        v_f32_st_tnsr_partial(o2, out, vf.v2, 0, 0);
    }
}
