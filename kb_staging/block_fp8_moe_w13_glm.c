// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w13_glm — Q21/RA-002 step 2, pass 1: routing-aware multi-expert
// block-FP8 dequant + GEMV + SiLU for the GLM-5.3-Flash MoE decode path
// (Gaudi 2 TPC, tpc-clang -march=gaudi2).
//
//   act[T, K, I] = SiLU(W13[e_k, 2I, H] · x[T, H] / s13[e_k])
//
//   For each token t and each of its K=8 routed experts e_k:
//     1. Load the FP8 weight rows for expert e_k (the global id → the local index)
//     2. Dequant: the BF16 = the FP8 × the block scale (the 128×128 blocks)
//     3. GEMV: the dot with the activation row x[t]
//     4. SiLU: act = x · sigmoid(x) applied to the first I columns
//        (the w13 output is [2I]: the gate [:I] and the up [I:])
//
//   in0: w13        E4M3 [E_LOCAL, 2I, H]   the stacked expert weights
//   in1: w13_scales F32  [E_LOCAL, 2I/128, H/128]
//   in2: x          BF16 [T, H]             the input activations
//   in3: topk_ids   int32 [T, K]            the global expert ids per token
//   out0: act       BF16 [T, K, I]          the SiLU-activated intermediate
//
// Index space: (T) × (I/128) points.
//   - dim0: i-block (128 intermediate columns per point)
//   - dim1: token index
// Each point: for the token t, computes the 128 intermediate columns for
// ALL K=8 routed experts. The expert loop: the inner (the sequential);
// the i-block: the outer (the parallel across the TPC instances).
//
// The TPC coords: the dim0 = the innermost, the dim1 = the next, the dim2 = the expert.
// The w13 layout: [E, 2I, H] — the TPC dims: {H, 2I, E} (the innermost = H).

#ifndef MOE_W13_KCH
#define MOE_W13_KCH 32  // the VLM x-chunks: H/128 <= 32 covers H = 4096
#endif
#ifndef MOE_W13_TOPK
#define MOE_W13_TOPK 8
#endif
#ifndef MOE_W13_I_BLOCK
#define MOE_W13_I_BLOCK 128
#endif

__local__ bfloat128 v8_x[MOE_W13_KCH];  // the activation row x[t] staged in VLM

void main(
    tensor w13,         // FP8 [E_LOCAL, 2I, H]
    tensor w13_scales,  // F32 [E_LOCAL, 2I/128, H/128]
    tensor x,           // BF16 [T, H]
    tensor topk_ids,    // int32 [T, K]
    tensor act          // BF16 [T, K, I] — the output
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int H = get_dim_size(x, 1);      // 4096
    const int T = get_dim_size(x, 0);      // the tokens
    const int I = get_dim_size(act, 2);    // 2048 (the moe intermediate)
    const int E_LOCAL = get_dim_size(w13, 2);  // the TPC dim2 = the expert
    const int HB = H / 128;                // the h-blocks: 32
    const int IB = I / 128;                // the i-blocks: 16
    const int KB2 = H / 256;               // the 256-value K-chunks: 16

    const int ib = is[0];   // the i-block index (0..IB-1)
    const int t = is[1];    // the token index
    if (t >= T) return;

    // Stage the activation row x[t] in VLM (the shared across the i-blocks
    // within the same TPC instance — the VLM is per-instance, so each point
    // stages its own copy; the x row = 4096 × 2 = 8 KB)
    for (int c = 0; c < HB; ++c) {
        int5 xc = {c * 128, t, 0, 0, 0};
        v8_x[c] = v_bf16_ld_tnsr_b(xc, x);
    }

    const ushort128 lid = read_lane_id_2b_b();

    // Read the K=8 routed expert ids for this token
    int expert_ids[MOE_W13_TOPK];
    for (int k = 0; k < MOE_W13_TOPK; ++k) {
        int5 idc = {k, t, 0, 0, 0};
        expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
    }

    // For each of the 128 intermediate columns in this i-block:
    // compute the dot for ALL K=8 experts, then the SiLU per expert.
    // The i-block covers the columns ib*128 .. ib*128+127 of the 2I dimension.
    // The gate columns: [:I]; the up columns: [I:2I].
    // The SiLU: applied to the gate columns only.

    for (int col = 0; col < 128; ++col) {
        const int col_global = ib * 128 + col;  // the column in the 2I dimension
        const bool is_gate = col_global < I;    // the gate vs the up

        // For each expert: the dot product over H
        // The results: the 8 values (one per expert) for this column
        float expert_dots[MOE_W13_TOPK];

        for (int k = 0; k < MOE_W13_TOPK; ++k) {
            const int e = expert_ids[k];  // the global expert id
            // The local expert index: the single-rank (the EP: the offset-adjusted)
            // For now: e is the global id; the local = e (the single-rank benchmark)
            const int e_local = e;

            float128 acc;
            acc.v1 = 0; acc.v2 = 0;

            // The dot over H in the 256-value chunks (the dequant + the MAC)
            for (int bk = 0; bk < KB2; ++bk) {
                // The block scale: w13_scales[e_local, col_global/128, bk*2]
                int5 sc = {bk * 2, col_global / 128, e_local, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales));
                sc[0] += 1;
                const bool second = bk * 256 + 128 < H;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w13_scales), 0, 0, second);

                // The FP8 weight load: w13[e_local, col_global, bk*256 .. bk*256+127]
                // The TPC coords: {H_offset, 2I_offset, E_offset} (the dim0 = the innermost = H)
                int5 coord = {bk * 256, col_global, e_local, 0, 0};
                const minifloat256 p1 = v_f8_ld_tnsr_b(coord, w13, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat128 v1 = v_convert_f8_to_bf16_b(p1) * mult;
                acc = v_bf16_mac_acc32_b(v1, v8_x[bk * 2], acc);

                coord[0] += 128;
                const minifloat256 p2 = v_f8_ld_tnsr_b(coord, w13, SW_UNPACK | SW_UNPCK_8_TO_16, 0, second);
                const bfloat128 v2 = v_convert_f8_to_bf16_b(p2) * mult2;
                acc = v_bf16_mac_acc32_b(v2, v8_x[bk * 2 + 1], acc);
            }

            expert_dots[k] = v_f32_reduce_add(acc.v1 + acc.v2);
        }

        // The SiLU + the store: for each expert, the activated value for this column
        // The act layout: [T, K, I] — the TPC dims: {I, K, T} (the dim0 = the innermost = I)
        // The column: the gate (the col_global < I) → the SiLU; the up → the raw
        for (int k = 0; k < MOE_W13_TOPK; ++k) {
            float val = expert_dots[k];
            float activated;
            if (is_gate) {
                // SiLU: x · sigmoid(x)
                activated = val / (1.0f + exp(-val));
            } else {
                activated = val;
            }
            // The store: act[t, k, col_global] = the activated (the BF16)
            // For the gate columns: the col_global (0..I-1); for the up: the col_global - I
            const int store_col = is_gate ? col_global : col_global - I;
            int5 ac = {store_col, k, t, 0, 0};
            // The scalar store (the one value per lane); the vector store: the K values
            // per column → the K scalar stores or the vectorized if the K values fit
            // in a vector lane. For the correctness-first: the scalar store.
            s_bf16_st_g(gen_addr(ac, act), (bfloat16)activated);
        }
    }
}
