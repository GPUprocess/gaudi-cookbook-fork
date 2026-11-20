static GlueCodeReturn inst_cs3(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[4] = {DATA_F8_143, DATA_F32, DATA_BF16, DATA_I32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* xs = in->inputTensors[2].geometry.maxSizes;
    const uint64_t* ids = in->inputTensors[3].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    if (in->inputTensors[0].geometry.dims != 3 || in->inputTensors[2].geometry.dims != 2 ||
        in->inputTensors[3].geometry.dims != 2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t H = ws[0], I2 = ws[1];
    const uint64_t T = xs[1];
    const uint64_t K = ids[0];
    if (os[0] != I2 || os[1] != K || os[2] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint32_t npts = tpcPoints(in, T * K * I2);
    setIndexSpace(in, out, npts);
    out->kernel.scalarParams[0] = npts;
    out->kernel.paramsNr = 1;
    return copyElf(out, ELF(cs3));
}
