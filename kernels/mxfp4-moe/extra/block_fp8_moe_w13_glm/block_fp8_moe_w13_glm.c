// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm v6 — Q21/RA-002 step 2, pass 1: routing-aware multi-expert
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
// THE v6 FIX (the one-hot-probe-guided): the VLM staging of x is REMOVED —
// the one-hot probes proved every convert/scale/MAC position correct while
// the full random-x test fails, isolating the defect to the shared-VLM x
// staging under multi-TP point concurrency. The x rows are now loaded
// DIRECTLY from the tensor per MAC pair: the added HBM x traffic is
// negligible against the fp8 weight reads, and the kernel becomes
// stateless.

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

    const int nb = is[0];   // the n-block index (0..2I/128-1)
    const int t = is[1];    // the token index
    if (t >= T) return;

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

        bfloat128 yv = (bfloat128)0.0f;  // the 128-lane output for the (t, k, nb)

        // The row loop: the 128 output columns in this n-block
        for (int row = 0; row < 128; ++row) {
            const int col_global = nb * 128 + row;  // the column in the 2I dim

            float128 acc;
            acc.v1 = 0; acc.v2 = 0;

            // The dot over H in the 256-value spans (the dequant + the MAC)
            for (int p = 0; p < HP; ++p) {
                // The block scales: the h-blocks 2p (the span's first 128) and 2p+1 (the second)
                int5 sc = {p * 2, col_global / 128, e, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));
                sc[0] += 1;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));

                // One 256-f8 load, ALL_LANES convert: .v1 = the first 128 values
                // (the h-block 2p), .v2 = the second 128 (the h-block 2p+1)
                int5 coord = {p * 256, col_global, e, 0, 0};
                const minifloat256 pk = v_f8_ld_tnsr_b(coord, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat256 wall = v_convert_f8_to_bf16_all_b(pk);

                // The x loads: DIRECT from the tensor (the no VLM — the v6)
                int5 xc1 = {p * 256, t, 0, 0, 0};
                int5 xc2 = {p * 256 + 128, t, 0, 0, 0};
                const bfloat128 x1 = v_bf16_ld_tnsr_b(xc1, x);
                const bfloat128 x2 = v_bf16_ld_tnsr_b(xc2, x);

                // The aligned MACs
                acc = v_bf16_mac_acc32_b(wall.v1 * mult, x1, acc);
                acc = v_bf16_mac_acc32_b(wall.v2 * mult2, x2, acc);
            }

            // The reduce-add → the lane insert for the row
            const float64 s = v_f32_reduce_add(acc.v1 + acc.v2);
            float128 s128;
            s128.v1 = s; s128.v2 = s;
            const bool128 la = v_u16_cmp_eq_b(lid, row, 0, (bool128){0}, 1, 0);
            yv = v_bf16_mov_vb(v_convert_f32_to_bf16_all_b(s128), 0, yv, la, 0);
        }

        // Store: raw[t, k, nb*128 .. nb*128+127] — the 128-lane vector ✓
        // The raw TPC dims: {2I, K, T} — the coord: {nb*128, k, t}
        int5 yc = {nb * 128, k, t, 0, 0};
        v_bf16_st_tnsr(yc, raw, yv);
    }
}
