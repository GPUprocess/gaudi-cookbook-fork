// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm v8dbg — Q21/RA-002 step 2, pass 1: routing-aware multi-expert
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

    // The 1-D index space (the mxk8 style): the ip -> the (t, nb) decode
    for (int ip = is[0]; ip < ie[0]; ++ip) {
    const int t = ip / NB;   // the token index
    const int nb = ip % NB;  // the n-block index (0..2I/128-1)
    if (t >= T) continue;
    // DEBUG: the x dump with the THREE addressing variants side by side:
    // the block A (the cols 0..127): the dim1 = 0 (the row 0 fixed)
    // the block B (the 128..255): the dim1 = t (the runtime variable)
    // the block C (the 256..383): the dim1 = 3 (the constant 3)
    // the block D (the 384..511): the x dump via the dim0-only walk
    int5 xa = {nb * 128, 0, 0, 0, 0};
    const bfloat128 va = v_bf16_ld_tnsr_b(xa, x);
    int5 xb = {nb * 128, t, 0, 0, 0};
    const bfloat128 vb = v_bf16_ld_tnsr_b(xb, x);
    int5 xc = {nb * 128, 3, 0, 0, 0};
    const bfloat128 vc = v_bf16_ld_tnsr_b(xc, x);
    for (int k = 0; k < 8; ++k) {
        int5 ya = {nb * 128, k, t, 0, 0};
        v_bf16_st_tnsr(ya, raw, va);
        int5 yb = {128 + nb * 128, k, t, 0, 0};
        v_bf16_st_tnsr(yb, raw, vb);
        int5 yc = {256 + nb * 128, k, t, 0, 0};
        v_bf16_st_tnsr(yc, raw, vc);
    }
}
}
