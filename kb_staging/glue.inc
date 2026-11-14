// glue.inc for block_fp8_moe_w2_glm (Q21 step 2, pass 2, GLM-5.3-Flash MoE).
// Included into libmxk_tpc.so by build.sh --extra-kernel after the shared
// helpers; DECL_ELF(block_fp8_moe_w2_glm) is declared by the generated
// mxk_extra.inc. Defines inst_block_fp8_moe_w2_glm.
//
// Contract:
//   in 0: w2           E4M3 (DATA_F8_143) [E_LOCAL, H, I] — the TPC dims: {I, H, E}
//   in 1: w2_scales    F32               [E_LOCAL, H/128, I/128] — the TPC dims: {I/128, H/128, E}
//   in 2: act          BF16 (DATA_BF16)  [T, K, I]             — the TPC dims: {I, K, T}
//   in 3: topk_ids     int32             [T, K]                — the TPC dims: {K, T}
//   in 4: topk_weights F32               [T, K]                — the TPC dims: {K, T}
//   out 0: y           BF16              [T, H]                — the TPC dims: {H, T}
// Index space: 2-D, T × H/128 points, one point = one (token, h-block):
// the 128 output h-columns accumulated over ALL K=8 routed experts.
static GlueCodeReturn inst_block_fp8_moe_w2_glm(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[5] = {DATA_F8_143, DATA_F32, DATA_BF16, DATA_I32, DATA_F32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[1].geometry.maxSizes;
    const uint64_t* as = in->inputTensors[2].geometry.maxSizes;
    const uint64_t* ids = in->inputTensors[3].geometry.maxSizes;
    const uint64_t* ows = in->inputTensors[4].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    // The TPC geometry is innermost-first: ws = {I, H, E}, ss = {I/128, H/128, E}, as = {I, K, T}, ids = {K, T}, ows = {K, T}
    if (in->inputTensors[0].geometry.dims != 3 || in->inputTensors[1].geometry.dims != 3 ||
        in->inputTensors[2].geometry.dims != 3 || in->inputTensors[3].geometry.dims != 2 ||
        in->inputTensors[4].geometry.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t I = ws[0], H = ws[1], E_LOCAL = ws[2];
    const uint64_t T = as[2];
    const uint64_t K = ids[0];
    if (!H || !I || !E_LOCAL || !T || !K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (I % 256 || H % 128 || I > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (ss[0] != I / 128 || ss[1] != H / 128 || ss[2] != E_LOCAL) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (as[0] != I || as[1] != K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (K != 8) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (os[0] != H || os[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    // The index space: T × H/128 points (the 2-D: the dim0 = the h-block, the dim1 = the token)
    const uint64_t npts_h = H / 128;
    const uint64_t npts_t = T;
    uint32_t p = tpcPoints(in, npts_h * npts_t);
    setIndexSpace(in, out, p);
    out->kernel.paramsNr = 0;
    return copyElf(out, ELF(block_fp8_moe_w2_glm));
}
