static GlueCodeReturn inst_f8dump_probe(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[1] = {DATA_F8_143};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* is_ = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    if (in->inputTensors[0].geometry.dims != 3 || in->outputTensors[0].geometry.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (is_[0] != 512 || is_[1] != 512 || is_[2] != 8 || os[0] != 256 || os[1] != 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    uint32_t p = tpcPoints(in, 2);
    setIndexSpace(in, out, p);
    out->kernel.paramsNr = 0;
    return copyElf(out, ELF(f8dump_probe));
}
