// op.cpp for block_fp8_dequant_glm (GLM-5.3-Flash Q21).
// Registers torch.ops.custom_op.block_fp8_dequant_glm(weight, scale) -> bf16 [N, K]
// via the framework's lazy-mode legacy custom-op API (hpu_custom_op.h), the same
// pattern as mxk_op_lazy.cpp.
#include "hpu_custom_op.h"

#include <torch/extension.h>

#include <mutex>
#include <vector>

using namespace habana::custom_op;
// [note] the torch schema is NOT declared here on purpose: the legacy bridge
// registers this op by name through REGISTER_CUSTOM_OP_ATTRIBUTES, and a
// TORCH_LIBRARY def of the same name collides with the plugin"s static
// registrations when both load in one process (d2d_memcpy double-register).

static std::vector<InputDesc> dequantInputs() {
    // [weight E4M3 tensor, scale F32 tensor]
    std::vector<InputDesc> in;
    in.push_back(InputDesc{input_type::TENSOR, 0u});
    in.push_back(InputDesc{input_type::TENSOR, 1u});
    return in;
}

static std::vector<OutputDesc> dequantOut() {
    // bf16, same shape as the weight
    return {OutputDesc{0, c10::ScalarType::BFloat16,
                       [](const at::Stack& st) {
                           return std::vector<int64_t>{st[0].toTensor().size(0), st[0].toTensor().size(1)};
                       }}};
}

at::Tensor blockFp8Dequant(const at::Tensor& weight, const at::Tensor& scale) {
    static std::once_flag f;
    std::call_once(f, []() {
        auto desc = dequantInputs();
        auto outs = dequantOut();
        // No node params: the kernel reads dims from the tensors and its point
        // id from the index space.
        auto pf = [](const at::Stack&, size_t& size) {
            HPU_PARAMS_STUB(int32_t);
            *params = 0;
            size = 0;
            return params;
        };
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::block_fp8_dequant_glm",
                                      "block_fp8_dequant_glm", desc, outs, pf);
    });
    std::vector<c10::IValue> in{weight, scale};
    auto d = HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::block_fp8_dequant_glm");
    return d.execute(in)[0];
}

