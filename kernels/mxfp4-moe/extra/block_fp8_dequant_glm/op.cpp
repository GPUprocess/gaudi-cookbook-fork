// SPDX-License-Identifier: Apache-2.0
// block_fp8_dequant_glm - PT2 custom-op with LAZY registration.
// The registration is deferred to the first call (not at .so load) to avoid
// the registrar collision: the backend .so registers its built-in kernels at
// import; our .so's load-time registration re-triggers the registry init and
// collides (EXP-Q21). The lazy registration runs AFTER the backend is fully
// initialized, inside the serving process's execution context.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "habanalabs/hpu_custom_op_pt2.h"

namespace {

constexpr const char* schema = "custom_op::block_fp8_dequant_glm";
constexpr const char* guid = "block_fp8_dequant_glm";

void validate(const at::Tensor& weight, const at::Tensor& scale) {
    TORCH_CHECK((weight.scalar_type() == at::kFloat8_e4m3fnuz || weight.scalar_type() == at::kFloat8_e4m3fn) &&
                scale.scalar_type() == at::kFloat && weight.dim() == 2 &&
                scale.dim() == 2 && weight.size(0) % 128 == 0 &&
                weight.size(1) % 256 == 0 && weight.is_contiguous() &&
                scale.is_contiguous() && weight.device() == scale.device(),
                "block FP8 dequant requires contiguous E4M3 [N,K] and F32 scales");
}

std::once_flag reg_flag;
bool reg_ok = false;

void lazy_register() {
    std::call_once(reg_flag, []() {
        habana::custom_op::registerUserCustomOp(
            schema, guid,
            [](const at::Stack& stack) {
                auto w = stack.at(0).toTensor();
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {w.size(0), w.size(1)};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}

at::Tensor run(const at::Tensor& weight, const at::Tensor& scale) {
    validate(weight, scale);
    lazy_register();
    TORCH_CHECK(reg_ok, "block_fp8_dequant_glm registration failed");
    TORCH_CHECK(weight.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{weight, scale};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    // Schema only — the HPU impl is routed through the PT2 descriptor at call time.
    m.def("block_fp8_dequant_glm(Tensor weight, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("block_fp8_dequant_glm",
           [](const at::Tensor& w, const at::Tensor& s) -> at::Tensor {
               TORCH_CHECK(false, "block_fp8_dequant_glm requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("block_fp8_dequant_glm",
           [](const at::Tensor& w, const at::Tensor& s) -> at::Tensor {
               return run(w, s);
           });
}
