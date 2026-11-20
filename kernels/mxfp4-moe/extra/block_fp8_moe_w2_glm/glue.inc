static GlueCodeReturn inst_block_fp8_moe_w2_glm(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[4] = {DATA_F8_143, DATA_F32, DATA_BF16, DATA_I32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[1].geometry.maxSizes;
    const uint64_t* as = in->inputTensors[2].geometry.maxSizes;
    const uint64_t* ids = in->inputTensors[3].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    if (in->inputTensors[0].geometry.dims != 3 || in->inputTensors[1].geometry.dims != 2 ||
        in->inputTensors[2].geometry.dims != 3 || in->inputTensors[3].geometry.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t I = ws[0], H = ws[1], E_LOCAL = ws[2];
    const uint64_t T = as[2];
    const uint64_t K = ids[0];
    if (!H || !I || !E_LOCAL || !T || !K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (I % 256 || H % 128 || I > 8192) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    // THE PER-ROW SCALES: [E_LOCAL, H] -> the TPC dims {H, E}
    if (ss[0] != H || ss[1] != E_LOCAL) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (as[0] != I || as[1] != K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (os[0] != H || os[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint32_t npts = tpcPoints(in, T * (H / 128));
    setIndexSpace(in, out, npts);
    out->kernel.scalarParams[0] = npts;
    out->kernel.paramsNr = 1;
    return copyElf(out, ELF(block_fp8_moe_w2_glm));
}
