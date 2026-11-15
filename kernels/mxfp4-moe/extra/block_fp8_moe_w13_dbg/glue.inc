// glue.inc for block_fp8_moe_w13_dbg (Q21 step 2, pass 1, GLM-5.3-Flash MoE).
// Included into libmxk_tpc.so by build.sh --extra-kernel after the shared
// helpers; DECL_ELF(block_fp8_moe_w13_dbg) is declared by the generated
// mxk_extra.inc. Defines inst_block_fp8_moe_w13_dbg.
//
// Contract:
//   in 0: w13        E4M3 (DATA_F8_143) [E_LOCAL, 2I, H] — the TPC dims: {H, 2I, E}
//        The H-order: STANDARD (the v5: the no repack; the ALL_LANES convert
//        keeps all the values: .v1 = the first 128 of each 256-span).
//   in 1: w13_scales F32               [E_LOCAL, 2I/128, H/128] — the TPC dims: {H/128, 2I/128, E}
//   in 2: x          BF16 (DATA_BF16)  [T, H]             — the TPC dims: {H, T}
//   in 3: topk_ids   int32             [T, K]             — the TPC dims: {K, T}
//   out 0: raw       BF16              [T, K, 2I]         — the TPC dims: {2I, K, T}
// Index space: 2-D, T × 2I/128 points, one point = one (token, n-block):
// the 128 raw output columns (of the 2I dim) for ALL K=8 routed experts.
// The SiLU + the gate/up product + the routing pre-weighting happen in PyTorch
// between this pass and block_fp8_moe_w2_glm (the TPC has no the scalar exp).
static GlueCodeReturn inst_block_fp8_moe_w13_dbg(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[4] = {DATA_F8_143, DATA_F32, DATA_BF16, DATA_I32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[1].geometry.maxSizes;
    const uint64_t* xs = in->inputTensors[2].geometry.maxSizes;
    const uint64_t* ids = in->inputTensors[3].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    // The TPC geometry is innermost-first: ws = {H, 2I, E}, ss = {H/128, 2I/128, E}, xs = {H, T}, ids = {K, T}
    if (in->inputTensors[0].geometry.dims != 3 || in->inputTensors[1].geometry.dims != 3 ||
        in->inputTensors[2].geometry.dims != 2 || in->inputTensors[3].geometry.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t H = ws[0], I2 = ws[1], E_LOCAL = ws[2];
    const uint64_t T = xs[1];
    const uint64_t K = ids[0];
    const uint64_t I = I2 / 2;
    if (!H || !I || !E_LOCAL || !T || !K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (H % 256 || I2 % 128 || H > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (ss[0] != H / 128 || ss[1] != I2 / 128 || ss[2] != E_LOCAL) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (xs[0] != H) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (K != 8) return GLUE_INCOMPATIBLE_INPUT_SIZE;  // the topk 8 for GLM-5.3-Flash
    if (os[0] != I2 || os[1] != K || os[2] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    // The index space: T × 2I/128 points (the 2-D: the dim0 = the n-block, the dim1 = the token)
    const uint64_t npts_i = I2 / 128;
    const uint64_t npts_t = T;
    uint32_t p = tpcPoints(in, npts_i * npts_t);
    setIndexSpace(in, out, p);
    out->kernel.paramsNr = 0;
    return copyElf(out, ELF(block_fp8_moe_w13_dbg));
}
