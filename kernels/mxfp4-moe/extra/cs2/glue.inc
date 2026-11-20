static GlueCodeReturn inst_cs2(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[4] = {DATA_F8_143, DATA_F32, DATA_BF16, DATA_I32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    if (in->inputTensors[0].geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t I2 = ws[1];
    if (os[0] != I2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    uint32_t p = tpcPoints(in, 1 << 20);
    setIndexSpace(in, out, p);
    out->kernel.paramsNr = 0;
    return copyElf(out, ELF(cs2));
}
