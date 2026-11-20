// CS2: constant acc -> reduce -> convert -> store at the v11-decoded positions.
void main(tensor w13, tensor w13_scales, tensor x, tensor topk_ids, tensor raw) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int I2 = get_dim_size(w13, 1);
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int col = ip % I2;
        const int k = (ip / I2) % 8;
        const int t = ip / (I2 * 8);
        float128 acc; acc.v1 = 1.0; acc.v2 = 1.0;
        const float64 total = v_f32_reduce_add(acc.v1 + acc.v2);
        float128 tv; tv.v1 = total; tv.v2 = 0;
        const bfloat128 tb = v_convert_f32_to_bf16_all_b(tv, SW_LINEAR);
        int5 oc = {col, k, t, 0, 0};
        v_bf16_st_tnsr_partial(oc, raw, tb, 0, 0);
    }
}
