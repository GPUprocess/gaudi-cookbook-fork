// SPDX-License-Identifier: Apache-2.0
// block_fp8_gemv_glm - PT2 custom-op with LAZY registration (Q21 step 1).
// Same lazy-registration pattern as block_fp8_dequant_glm: the registration
// defers to the first call so it runs after the backend is fully initialized
// inside the serving process (the B7 load-order recipe).
//
//   y[N] = W[N,K] · x[K]  — one expert's block-FP8 GEMV (bs=1 decode core)
//
// The routing-aware fused MoE op (w13 + SiLU + w2 + the weighted sum over
// topk experts) is the step-2 integration on top of this core.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "habanalabs/hpu_custom_op_pt2.h"

namespace {

constexpr const char* schema = "custom_op::block_fp8_gemv_glm";
constexpr const char* guid = "block_fp8_gemv_glm";

void validate(const at::Tensor& w, const at::Tensor& scales, const at::Tensor& x) {
    TORCH_CHECK((w.scalar_type() == at::kFloat8_e4m3fnuz || w.scalar_type() == at::kFloat8_e4m3fn) &&
                scales.scalar_type() == at::kFloat &&
                x.scalar_type() == at::kBFloat16 &&
                w.dim() == 2 && scales.dim() == 2 && x.dim() == 1 &&
                w.size(0) % 128 == 0 && w.size(1) % 256 == 0 && w.size(1) <= 4096 &&
                scales.size(0) == w.size(0) / 128 && scales.size(1) == w.size(1) / 128 &&
                x.size(0) == w.size(1) &&
                w.is_contiguous() && scales.is_contiguous() && x.is_contiguous() &&
                w.device() == scales.device() && w.device() == x.device(),
                "block_fp8_gemv_glm requires contiguous E4M3 [N,K], F32 [N/128,K/128] scales and BF16 [K] x");
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
                output.shape = {w.size(0)};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}

at::Tensor run(const at::Tensor& w, const at::Tensor& scales, const at::Tensor& x) {
    validate(w, scales, x);
    lazy_register();
    TORCH_CHECK(reg_ok, "block_fp8_gemv_glm registration failed");
    TORCH_CHECK(w.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{w, scales, x};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("block_fp8_gemv_glm(Tensor w, Tensor scales, Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("block_fp8_gemv_glm",
           [](const at::Tensor& w, const at::Tensor& s, const at::Tensor& x) -> at::Tensor {
               TORCH_CHECK(false, "block_fp8_gemv_glm requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("block_fp8_gemv_glm",
           [](const at::Tensor& w, const at::Tensor& s, const at::Tensor& x) -> at::Tensor {
               return run(w, s, x);
           });
}
