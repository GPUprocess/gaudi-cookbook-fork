// SPDX-License-Identifier: Apache-2.0
//
// Kernel block_fp8_dequant_glm (GLM-5.3-Flash Q21): E4M3 block-FP8 weights -> BF16.
//   in  weight E4M3 [N, K]  (128x128 blocks, our served checkpoint layout)
//   in  scale   F32 [N/128, K/128]  (one FP32 scale per 128x128 block)
//   out bf16 [N, K], out[n, k] = W[n, k]
// Adapted from 1Cat-vLLM-Gaudi block_fp8_dequant_gaudi2.c (Apache-2.0) to the
// mxfp4-moe framework's 1-D index-space convention: the glue sets index space
// [0, npts) with one point per TPC; each point owns a contiguous slice of the
// (block_n, block_k) pairs and decomposes them by div/mod. The scale is rounded
// to BF16 BEFORE multiplying (the checkpoint contract); the load pipe expands
// the packed E4M3 bytes straight into the conversion lanes (v_f8_ld_tnsr_b
// SW_UNPACK + v_convert_f8_to_bf16_b).
//
// Grid assumptions (checked by the glue): K % 256 == 0 (whole 256-byte rows),
// N % 128 == 0.
void main(tensor weight, tensor scale, tensor output, int npts)
{
    const int pid = get_index_space_offset()[0];
    const int sn = get_dim_size(scale, 0);  // block rows: N / 128
    const int sk = get_dim_size(scale, 1);  // block col pairs: K / 256
    const int pairs = sn * sk;
    const int width = get_dim_size(weight, 0);
    const int per = (pairs + npts - 1) / npts;
    for (int blk = pid * per; blk < (pid + 1) * per && blk < pairs; ++blk) {
        const int block_n = blk / sk;
        const int block_k = blk % sk;
        int5 scale_coord = {block_k * 2, block_n, 0, 0, 0};
        const bfloat128 multiplier = (bf16)s_f32_ld_g(gen_addr(scale_coord, scale));
        scale_coord[0] += 1;
        const bool second = block_k * 256 + 128 < width;
        const bfloat128 multiplier2 = (bf16)s_f32_ld_g(gen_addr(scale_coord, scale), 0, 0, second);
        #pragma loop_unroll(4) taken
        for (int row = 0; row < 128; ++row) {
            int5 coord = {block_k * 256, block_n * 128 + row, 0, 0, 0};
            const minifloat256 packed = v_f8_ld_tnsr_b(coord, weight, SW_UNPACK | SW_UNPCK_8_TO_16);
            const bfloat128 values = v_convert_f8_to_bf16_b(packed);
            v_bf16_st_tnsr(coord, output, values * multiplier);
            coord[0] += 128;
            const minifloat256 packed2 = v_f8_ld_tnsr_b(coord, weight, SW_UNPACK | SW_UNPCK_8_TO_16, 0, second);
            const bfloat128 values2 = v_convert_f8_to_bf16_b(packed2);
            v_bf16_st_tnsr(coord, output, values2 * multiplier2, 0, second);
        }
    }
}
