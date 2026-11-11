// glue.inc for block_fp8_dequant_glm (GLM-5.3-Flash Q21).
// Included into libmxk_tpc.so by build.sh --extra-kernel after the shared
// helpers; DECL_ELF(block_fp8_dequant_glm) is declared by the generated
// mxk_extra.inc. Defines inst_block_fp8_dequant_glm.
//
// Contract (mirrors the 1Cat host glue, adapted to the framework helpers):
//   in 0: weight E4M3 (DATA_F8_143) [N, K], K % 256 == 0, N % 128 == 0
//   in 1: scale   F32          [N/128, K/128]  (FP32 per 128x128 block)
//   out 0: bf16               [N, K]
// Index space: 1-D, `pairs` = (N/128) * (K/256) units, one point per TPC;
// the kernel decomposes (block_n, block_k) from its point id.
static GlueCodeReturn inst_block_fp8_dequant_glm(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[2] = {DATA_F8_143, DATA_F32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[1].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    if (in->inputTensors[0].geometry.dims != 2 || in->inputTensors[1].geometry.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t K = ws[0], N = ws[1];  // the geometry is innermost-first (the square-w13 case masked this)
    if (!N || !K || K % 256 || N % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (ss[0] != K / 128 || ss[1] != N / 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;  // the TPC order
    if (os[0] != K || os[1] != N) return GLUE_INCOMPATIBLE_INPUT_SIZE;  // the output geometry is TPC order too
    const uint64_t pairs = (N / 128) * (K / 256);
    // One index point per TPC; the kernel slices the pairs by point id. The
    // point's access patterns are all-required (each point writes its own
    // disjoint block set; the scale is read whole).
    uint32_t npts = tpcPoints(in, pairs);
    // Framework convention (as inst_mxk_deq): indexSpaceGeometry[0] = npts
    // points, all-required access patterns, and the kernel receives npts as a
    // scalar param to slice the (block_n, block_k) pairs by point id.
    setIndexSpace(in, out, npts);
    out->kernel.scalarParams[0] = (int32_t)npts;
    out->kernel.paramsNr = 1;
    return copyElf(out, ELF(block_fp8_dequant_glm));
}
