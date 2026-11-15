// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm v9 — Q21/RA-002 step 2, pass 1: routing-aware multi-expert
// block-FP8 GEMV for the GLM-5.3-Flash MoE decode path (Gaudi 2 TPC).
//
//   raw[T, K, 2I] = W13[e_k, 2I, H] · x[T, H] / s13[e_k]
//
//   (the SiLU + the gate/up split are done in PyTorch post-processing — the TPC
//    has no the scalar exp/div; the vectorized SiLU in PyTorch is faster)
//
//   in0: w13        E4M3 [E_LOCAL, 2I, H]   the stacked expert weights, the
//        STANDARD H-order (the no repack)
//   in1: w13_scales F32  [E_LOCAL, 2I/128, H/128]  the STANDARD order
//   in2: x          BF16 [T, H]             the input activations
//   in3: topk_ids   int32 [T, K]            the global expert ids per token
//   out0: raw       BF16 [T, K, 2I]         the raw GEMV output (the pre-activation)
//
// Index space: T × (2I/128) points (2-D).
//   - dim0: n-block (128 output columns of the 2I dim per point)
//   - dim1: token index
// Each point: for the token t, the n-block nb (the 128 output columns):
//   for each of the K=8 routed experts: the 128-row GEMV over H
//   → the per-expert 128-lane result → the vector store per expert.
// The activation x[t] staged in VLM once (the shared across the K experts).
//
// THE v5 FIX (the v2/v3 numeric-fail root cause, the corpus-confirmed): the
// plain v_convert_f8_to_bf16_b down-converts the 256 fp8 into a SINGLE
// bfloat128 register — only 128 of the 256 values survive (the half-term dot).
// The ALL_LANES form v_convert_f8_to_bf16_all_b converts ALL 256 values into
// a bfloat256 register PAIR: .v1 = the span's first 128 values (the h-block
// 2p), .v2 = the second 128 (the h-block 2p+1) — the exact split matching the
// two 128-value scale blocks and the two staged x chunks per 256-span. The
// STANDARD weight order, one 256-f8 load per span (the half the v2 loads).
//
// THE v8 FIX (the probe-ladder-guided): the x loads, the fp8 loads, the
// ALL_LANES convert, and the bf16 stores are ALL verified EXACT in
// isolation (the xdump/f8dump probes, the max err 0.0). The remaining
// difference from the proven kernels: MY 2-D index space (is[0]=n-block,
// is[1]=token) vs the PROVEN 1-D everywhere (the mxk8, the mxk_deq). The
// v8: the 1-D index space — ip = is[0], nb = ip % (2I/128), t = ip /
// (2I/128) — the mxk8-style software decomposition; the store: back to the
// FULL vector per (t, n-block) with the pair-packed rows (the 64 pairs x
// the 2 dual groups per the 128-lane vector).

#ifndef MOE_W13_V6_TOPK
#define MOE_W13_V6_TOPK 8
#endif

void main(
    tensor w13,         // FP8 [E_LOCAL, 2I, H] — the STANDARD H-order
    tensor w13_scales,  // F32 [E_LOCAL, 2I/128, H/128]
    tensor x,           // BF16 [T, H]
    tensor topk_ids,    // int32 [T, K]
    tensor raw          // BF16 [T, K, 2I] — the raw GEMV output
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int H = get_dim_size(x, 1);       // 4096 (the TPC dim0 of x)
    const int T = get_dim_size(x, 0);       // the tokens (the TPC dim1 of x)
    const int E_LOCAL = get_dim_size(w13, 2); // the TPC dim2 of w13
    const int HP = H / 256;                 // the 256-value spans: 16
    const int NB = get_dim_size(w13, 1) / 128; // the n-blocks: 2I/128

    // The 1-D index space (the v9): the ip -> the (t, nb, m) decode:
    // the ip = ((t * NB) + nb) * 64 + m — the one (token, col-pair) per the point
    for (int ip = is[0]; ip < ie[0]; ++ip) {
    const int m = ip % 64;
    const int nb = (ip / 64) % NB;
    const int t = ip / (64 * NB);
    if (t >= T) continue;

    const ushort128 lid = read_lane_id_2b_b();

    // Read the K=8 routed expert ids for this token
    int expert_ids[MOE_W13_V6_TOPK];
    for (int k = 0; k < MOE_W13_V6_TOPK; ++k) {
        int5 idc = {k, t, 0, 0, 0};
        expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
    }

    // The expert loop: OUTER (the per-expert vector store)
    for (int k = 0; k < MOE_W13_V6_TOPK; ++k) {
        const int e = expert_ids[k];  // the global expert id = the local (the single-rank)

        // The row pair: the col_a/col_b from the decoded m (the one pair per the point)
        {
            const int col_a = nb * 128 + 2 * m;
            const int col_b = col_a + 1;

            float128 acc_a; acc_a.v1 = 0; acc_a.v2 = 0;
            float128 acc_b; acc_b.v1 = 0; acc_b.v2 = 0;

            // The dot over H in the 256-value spans (the dequant + the MAC)
            for (int p = 0; p < HP; ++p) {
                int5 sc = {p * 2, col_a / 128, e, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));
                sc[0] += 1;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));

                // ONE 256-f8 load serves BOTH rows (the adjacent columns):
                // the coord: the col_a; the col_b = the col_a + 1: the NEXT load
                int5 coord_a = {p * 256, col_a, e, 0, 0};
                const minifloat256 pk_a = v_f8_ld_tnsr_b(coord_a, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat256 w_a = v_convert_f8_to_bf16_all_b(pk_a);
                int5 coord_b = {p * 256, col_b, e, 0, 0};
                const minifloat256 pk_b = v_f8_ld_tnsr_b(coord_b, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat256 w_b = v_convert_f8_to_bf16_all_b(pk_b);

                int5 xc1 = {p * 256, t, 0, 0, 0};
                int5 xc2 = {p * 256 + 128, t, 0, 0, 0};
                const bfloat128 x1 = v_bf16_ld_tnsr_b(xc1, x);
                const bfloat128 x2 = v_bf16_ld_tnsr_b(xc2, x);

                acc_a = v_bf16_mac_acc32_b(w_a.v1 * mult, x1, acc_a);
                acc_a = v_bf16_mac_acc32_b(w_a.v2 * mult2, x2, acc_a);
                acc_b = v_bf16_mac_acc32_b(w_b.v1 * mult, x1, acc_b);
                acc_b = v_bf16_mac_acc32_b(w_b.v2 * mult2, x2, acc_b);
            }

            // The reduces: the two row sums
            const float64 s_a = v_f32_reduce_add(acc_a.v1 + acc_a.v2);
            const float64 s_b = v_f32_reduce_add(acc_b.v1 + acc_b.v2);

            // The store: the SCALAR partial stores (the deepseek mhc pattern:
            // one partial store per output column, the count=0, the reduce
            // result directly — no 128-lane vector construction)
            const bfloat128 sa_b = v_convert_f32_to_bf16_all_b((float128){s_a, s_a}, SW_LINEAR);
            const bfloat128 sb_b = v_convert_f32_to_bf16_all_b((float128){s_b, s_b}, SW_LINEAR);
            int5 yc_a = {col_a, k, t, 0, 0};
            int5 yc_b = {col_b, k, t, 0, 0};
            v_bf16_st_tnsr_partial(yc_a, raw, sa_b, 3, 0);
            v_bf16_st_tnsr_partial(yc_b, raw, sb_b, 3, 0);
        }
    }
    }
}
