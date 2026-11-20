// SPDX-License-Identifier: Apache-2.0
//
// block_fp8_moe_w2_glm v6 — Q21/RA-002 step 2, pass 2 (the CS3-verified coverage fix).
//
//   y[T, H] = Σ_k (W2[e_k, H, I] · act_preweighted[T, K, I])
//
// THE COVERAGE FIX (Q21-E31): tpcPoints() caps the index space at maxAvailableTpc (24);
// the kernel loops over its work SLICE with npts passed as a scalar param (the mxk8 idiom).
// The previous v5 decoded is[0]/is[1] as (hb, t) in a rank-1 space -> is[1] = 0 always ->
// only token 0's first 24 h-blocks were ever processed.
//
//   in0: w2           E4M3 [E_LOCAL, H, I]   TPC dims {I, H, E}
//   in1: w2_scales    F32  [E_LOCAL, H/128, I/128]  TPC dims {I/128, H/128, E}
//   in2: act          BF16 [T, K, I]          TPC dims {I, K, T}  (the PRE-WEIGHTED)
//   in3: topk_ids     int32 [T, K]            TPC dims {K, T}
//   out0: y           BF16 [T, H]             TPC dims {H, T}
#ifndef MOE_W2_V5_TOPK
#define MOE_W2_V5_TOPK 8
#endif
void main(
    tensor w2,
    tensor w2_scales,
    tensor act,
    tensor topk_ids,
    tensor y,
    int npts
) {
    const int5 is = get_index_space_offset();
    const int5 ie = is + get_index_space_size();
    const int I = get_dim_size(act, 0);
    const int T = get_dim_size(act, 2);
    const int H = get_dim_size(y, 0);
    const int IP = I / 256;
    const int nhb = H / 128;
    const int W = T * nhb;
    const ushort128 lid = read_lane_id_2b_b();
    for (int ip = is[0]; ip < ie[0]; ++ip) {
        const int w0 = ip * W / npts;
        const int w1 = (ip + 1) * W / npts;
        for (int wp = w0; wp < w1; ++wp) {
            const int t = wp / nhb;
            const int hb = wp % nhb;
            int expert_ids[MOE_W2_V5_TOPK];
            for (int k = 0; k < MOE_W2_V5_TOPK; ++k) {
                int5 idc = {k, t, 0, 0, 0};
                expert_ids[k] = s_i32_ld_g(gen_addr(idc, topk_ids));
            }
            bfloat128 yv = (bfloat128)0.0f;
            for (int row = 0; row < 128; ++row) {
                const int row_global = hb * 128 + row;
                float128 acc;
                acc.v1 = 0; acc.v2 = 0;
                for (int k = 0; k < MOE_W2_V5_TOPK; ++k) {
                    const int e = expert_ids[k];
                    if (e < 0 || e >= get_dim_size(w2, 2)) continue;  // the non-local pair: no contribution
                    for (int p = 0; p < IP; ++p) {
                        int5 sc = {p * 2, row_global / 128, e, 0, 0};
                        const bfloat128 mult = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales));
                        sc[0] += 1;
                        const bfloat128 mult2 = (bfloat128)s_f32_ld_g(gen_addr(sc, w2_scales));
                        int5 ac1 = {p * 256, k, t, 0, 0};
                        int5 ac2 = {p * 256 + 128, k, t, 0, 0};
                        const bfloat128 a1 = v_bf16_ld_tnsr_b(ac1, act);
                        const bfloat128 a2 = v_bf16_ld_tnsr_b(ac2, act);
                        int5 coord = {p * 256, row_global, e, 0, 0};
                        const minifloat256 pk = v_f8_ld_tnsr_b(coord, w2, SW_UNPACK | SW_UNPCK_8_TO_16);
                        const bfloat256 wall = v_convert_f8_to_bf16_all_b(pk);
                        acc = v_bf16_mac_acc32_b(wall.v1 * mult, a1, acc);
                        acc = v_bf16_mac_acc32_b(wall.v2 * mult2, a2, acc);
                    }
                }
                const float64 s = v_f32_reduce_add(acc.v1 + acc.v2);
                float128 s128;
                s128.v1 = s; s128.v2 = s;
                const bool128 la = v_u16_cmp_eq_b(lid, row, 0, (bool128){0}, 1, 0);
                yv = v_bf16_mov_vb(v_convert_f32_to_bf16_all_b(s128), 0, yv, la, 0);
            }
            int5 yc = {hb * 128, t, 0, 0, 0};
            v_bf16_st_tnsr(yc, y, yv);
        }
    }
}
