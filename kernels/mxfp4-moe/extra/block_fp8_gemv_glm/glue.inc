// glue.inc for block_fp8_gemv_glm (Q21 step 1, GLM-5.3-Flash decode MoE).
// Included into libmxk_tpc.so by build.sh --extra-kernel after the shared
// helpers; DECL_ELF(block_fp8_gemv_glm) is declared by the generated
// mxk_extra.inc. Defines inst_block_fp8_gemv_glm.
//
// Contract:
//   in 0: weight E4M3 (DATA_F8_143) [N, K], K % 256 == 0, N % 128 == 0, K <= 4096
//   in 1: scale   F32               [N/128, K/128]  (FP32 per 128x128 block)
//   in 2: x       BF16              [K]             (one activation row)
//   out 0: y      BF16              [N]
// Index space: 1-D, N/128 points, one point = one n-block (128 output rows,
// full-K dot per row). The kernel derives everything from the tensor dims;
// no scalar params.
static GlueCodeReturn inst_block_fp8_gemv_glm(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[3] = {DATA_F8_143, DATA_F32, DATA_BF16};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[1].geometry.maxSizes;
    const uint64_t* xs = in->inputTensors[2].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    fprintf(stderr, "[gemv-glue] dims: w=%u s=%u x=%u | w sizes=%lu,%lu,%lu s=%lu,%lu x=%lu out=%lu,%lu\n",
            in->inputTensors[0].geometry.dims, in->inputTensors[1].geometry.dims, in->inputTensors[2].geometry.dims,
            (unsigned long)ws[0], (unsigned long)ws[1], (unsigned long)ws[2],
            (unsigned long)ss[0], (unsigned long)ss[1], (unsigned long)xs[0],
            (unsigned long)os[0], (unsigned long)os[1]);
    if (in->inputTensors[0].geometry.dims != 2 || in->inputTensors[1].geometry.dims != 2 ||
        in->inputTensors[2].geometry.dims != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t K = ws[0], N = ws[1];  // the geometry is innermost-first: ws[0]=K, ws[1]=N
    if (!N || !K || K % 256 || N % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (K > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;  // the VLM staging cap
    if (ss[0] != K / 128 || ss[1] != N / 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;  // the scales [K/128, N/128] in TPC order
    if (xs[0] != K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (os[0] != N) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t npts = N / 128;
    uint32_t p = tpcPoints(in, npts);
    setIndexSpace(in, out, p);
    out->kernel.paramsNr = 0;
    return copyElf(out, ELF(block_fp8_gemv_glm));
}
