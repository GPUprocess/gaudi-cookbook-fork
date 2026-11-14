// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm v3 — Q21/RA-002 step 2, pass 1: routing-aware multi-expert
// block-FP8 GEMV for the GLM-5.3-Flash MoE decode path (Gaudi 2 TPC).
//
//   raw[T, K, 2I] = W13[e_k, 2I, H] · x[T, H] / s13[e_k]
//
//   (the SiLU + the gate/up split are done in PyTorch post-processing — the TPC
//    has no the scalar exp/div; the vectorized SiLU in PyTorch is faster)
//
//   in0: w13        E4M3 [E_LOCAL, 2I, H]   the stacked expert weights in the
//        INTERLEAVED-CHUNK-PAIR H-order: the span [p*256 .. p*256+256) holds
//        the original 128-h-chunks (2p, 2p+1) interleaved — the position
//        p*256+2j = the chunk 2p's element j, the position p*256+2j+1 = the
//        chunk (2p+1)'s element j. (One-time reorder at the weight prep.)
//   in1: w13_scales F32  [E_LOCAL, 2I/128, H/128]  (the STANDARD order: the
//        scale [.., 2p] belongs to the evens, [.., 2p+1] to the odds)
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
// THE v3 ALIGNMENT FIX (the v2 numeric-fail root cause): the f8 ld_tnsr with
// SW_UNPCK_8_TO_16 loads the 256 fp8 as the PACKED PAIRS (2 per 32-bit cell)
// and v_convert_f8_to_bf16_b keeps that layout — the 256 bf16 in the packed
// pairs. MACing that against a 1-per-cell staged x vector drops the odd
// h-values (the half-term dot). The v3: the interleaved weight order puts the
// h-chunk 2p in the EVEN cells and the h-chunk 2p+1 in the ODD cells of each
// converted pair, and v_bf16_mov_dual_group_unpack_b with SW_UNPACK_EVEN_LANES
// / SW_UNPACK_ODD_LANES splits the pairs into the 1-per-cell vectors aligned
// with the staged x chunks 2p and 2p+1. One 256-f8 load per chunk pair (the
// half the loads of the v2), no predication (the glue guarantees H % 256 == 0).

#ifndef MOE_W13_V3_KCH
#define MOE_W13_V3_KCH 32  // the VLM x-chunks: H/128 <= 32 covers H = 4096
#endif
#ifndef MOE_W13_V3_TOPK
#define MOE_W13_V3_TOPK 8
#endif

__local__ bfloat128 v8_x[MOE_W13_V3_KCH];  // the activation row x[t] staged in VLM

void main(
    tensor w13,         // FP8 [E_LOCAL, 2I, H] — the INTERLEAVED-CHUNK-PAIR H-order
    tensor w13_scales,  // F32 [E_LOCAL, 2I/128, H/128] — the STANDARD order
    tensor x,           // BF16 [T, H]
    tensor topk_ids,    // int32 [T, K]
    tensor raw          // BF16 [T, K, 2I] — the raw GEMV output
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int H = get_dim_size(x, 1);       // 4096 (the TPC dim0 of x)
    const int T = get_dim_size(x, 0);       // the tokens (the TPC dim1 of x)
    const int E_LOCAL = get_dim_size(w13, 2); // the TPC dim2 of w13
    const int KB = H / 128;                 // the VLM x-chunks: 32
    const int HP = H / 256;                 // the interleaved chunk pairs: 16

    const int nb = is[0];   // the n-block index (0..2I/128-1)
    const int t = is[1];    // the token index
    if (t >= T) return;

    // Stage the activation row x[t] in VLM (the shared across the K experts)
    for (int c = 0; c < KB; ++c) {
        int5 xc = {c * 128, t, 0, 0, 0};
        v8_x[c] = v_bf16_ld_tnsr_b(xc, x);
    }

    const ushort128 lid = read_lane_id_2b_b();

    // Read the K=8 routed expert ids for this token
    int expert_ids[MOE_W13_V3_TOPK];
    for (int k = 0; k < MOE_W13_V3_TOPK; ++k) {
        int5 idc = {k, t, 0, 0, 0};
        expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
    }

    // The expert loop: OUTER (the per-expert vector store)
    for (int k = 0; k < MOE_W13_V3_TOPK; ++k) {
        const int e = expert_ids[k];  // the global expert id = the local (the single-rank)

        bfloat128 yv = (bfloat128)0.0f;  // the 128-lane output for the (t, k, nb)

        // The row loop: the 128 output columns in this n-block
        for (int row = 0; row < 128; ++row) {
            const int col_global = nb * 128 + row;  // the column in the 2I dim

            float128 acc;
            acc.v1 = 0; acc.v2 = 0;

            // The dot over H in the 256-span chunk pairs (the dequant + the MAC)
            for (int p = 0; p < HP; ++p) {
                // The block scales: the original h-chunks 2p (the evens) and 2p+1 (the odds)
                int5 sc = {p * 2, col_global / 128, e, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));
                sc[0] += 1;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));

                // One interleaved 256-f8 load: the evens = the h-chunk 2p, the odds = the 2p+1
                int5 coord = {p * 256, col_global, e, 0, 0};
                const minifloat256 pk = v_f8_ld_tnsr_b(coord, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat128 wboth = v_convert_f8_to_bf16_b(pk);
                const bfloat128 wE = v_bf16_mov_dual_group_unpack_b(wboth, 0xFFFFFFFF, 0, 0, 0, 0, SW_UNPACK_EVEN_LANES, (bfloat128)0, 1);
                const bfloat128 wO = v_bf16_mov_dual_group_unpack_b(wboth, 0xFFFFFFFF, 0, 0, 0, 0, SW_UNPACK_ODD_LANES, (bfloat128)0, 1);

                // The aligned MACs: the evens × the x chunk 2p, the odds × the x chunk 2p+1
                acc = v_bf16_mac_acc32_b(wE * mult, v8_x[p * 2], acc);
                acc = v_bf16_mac_acc32_b(wO * mult2, v8_x[p * 2 + 1], acc);
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
