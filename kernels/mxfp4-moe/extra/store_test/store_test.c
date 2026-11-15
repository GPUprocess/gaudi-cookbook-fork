// store_test — the pure store-addressing test: ONE load, the stores to the 4 (t,k) targets.
// in:  BF16 [1, 256]: the pattern arange(1000..1255) — the identifiable
// out: BF16 [4, 8, 512]: the raw-shaped; the raw[t][k] block nb=0 = the in dump per (t,k)
void main(tensor in, tensor out) {
    const int5 is = get_index_space_offset();
    const int5 c = {is[0] * 128, 0, 0, 0, 0};
    const bfloat128 v = v_bf16_ld_tnsr_b(c, in);
    for (int t = 0; t < 4; ++t)
        for (int k = 0; k < 8; ++k) {
            int5 oc = {is[0] * 128, k, t, 0, 0};
            v_bf16_st_tnsr(oc, out, v);
        }
}
