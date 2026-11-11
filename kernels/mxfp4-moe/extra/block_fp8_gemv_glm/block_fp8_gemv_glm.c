// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_gemv_glm — Q21/RA-002 step 1: single-expert block-FP8 GEMV core
// for the GLM-5.3-Flash MoE decode path (Gaudi 2 TPC, tpc-clang -march=gaudi2).
//
//   y[N] = W[N,K] · x[K]
//
//   in0: W      E4M3 [N, K]   one expert's w13 or w2 slice, N % 128 == 0, K % 256 == 0
//   in1: scales F32  [N/128, K/128]   one FP32 scale per 128×128 block
//   in2: x      bf16 [K]      one activation row (bs=1 decode)
//   out0: y     bf16 [N]
//
// Index space: N/128 points; one point computes the 128 output rows of one
// n-block with the full-K dot per row. torch's [N,K] row-major maps to the
// TPC coords as coord[0]=k (innermost), coord[1]=n — the same convention the
// 1Cat dequant kernel uses. The activation is staged in VLM once per point;
// per row the 256-value chunks load/convert/MAC exactly like the 1Cat
// reference dequant kernel, accumulating fp32 across K, then one
// v_f32_reduce_add per row and a lane-inserted vector store per n-block.
//
// This is the correctness-first variant: one row per pass, no software
// pipelining. The perf follow-up (the 4-row pipelined passes à la mxk8) is
// staged separately; this kernel is the numerical reference for it.

#ifndef GEMV_KCH
#define GEMV_KCH 32  // VLM x-chunks: K/128 <= 32 covers K <= 4096 (w13 and w2)
#endif

__local__ bfloat128 v8_gx[GEMV_KCH];

void main(tensor w, tensor scales, tensor x, tensor y) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int K = get_dim_size(w, 0);   // TPC dim0 = innermost = K
    const int N = get_dim_size(w, 1);
    const int KB = K / 128;
    const int KB2 = K / 256;

    // stage the activation row in VLM
    for (int c = 0; c < KB; ++c) {
        int5 xc = {c * 128, 0, 0, 0, 0};
        v8_gx[c] = v_bf16_ld_tnsr_b(xc, x);
    }

    const ushort128 lid = read_lane_id_2b_b();

    for (int nb = is[0]; nb < ie[0]; ++nb) {
        bfloat128 yv = (bfloat128)0.0f;
        for (int row = 0; row < 128; ++row) {
            float128 acc;
            acc.v1 = 0;
            acc.v2 = 0;
            for (int bk = 0; bk < KB2; ++bk) {
                int5 sc = {bk * 2, nb, 0, 0, 0};
                const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, scales));
                sc[0] += 1;
                const bool second = bk * 256 + 128 < K;
                const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, scales), 0, 0, second);
                int5 coord = {bk * 256, nb * 128 + row, 0, 0, 0};
                const minifloat256 p1 = v_f8_ld_tnsr_b(coord, w, SW_UNPACK | SW_UNPCK_8_TO_16);
                const bfloat128 v1 = v_convert_f8_to_bf16_b(p1) * mult;
                acc = v_bf16_mac_acc32_b(v1, v8_gx[bk * 2], acc);
                coord[0] += 128;  // [fix] advance to the second 128-value half before the paired load
                const minifloat256 p2 = v_f8_ld_tnsr_b(coord, w, SW_UNPACK | SW_UNPCK_8_TO_16, 0, second);
                const bfloat128 v2 = v_convert_f8_to_bf16_b(p2) * mult2;
                acc = v_bf16_mac_acc32_b(v2, v8_gx[bk * 2 + 1], acc);
            }
            const float64 s = v_f32_reduce_add(acc.v1 + acc.v2);
            // insert the row's sum into lane `row` of the output vector
            float128 s128;
            s128.v1 = s;
            s128.v2 = s;
            const bool128 la = v_u16_cmp_eq_b(lid, row, 0, (bool128){0}, 1, 0);
            yv = v_bf16_mov_vb(v_convert_f32_to_bf16_all_b(s128), 0, yv, la, 0);
        }
        int5 yc = {nb * 128, 0, 0, 0, 0};
        v_bf16_st_tnsr(yc, y, yv);
    }
}
