// SPDX-License-Identifier: Apache-2.0
// block_fp8_moe_w2_glm - PT2 custom-op with LAZY registration (Q21 step 2, pass 2).
// Same lazy-registration pattern as the pass 1: the registration defers to the
// first call so it runs after the backend is fully initialized inside the
// serving process.
//
//   y[T,H] = Σ_k w_k · (W2[e_k,H,I] · act[T,K,I] / s2[e_k])
//
//   The pass 2 of the two-pass fused MoE: for each token and each of its K=8
//   routed experts, the dequant (the FP8 → the BF16 with the block scales)
//   + the GEMV over I using the act from the pass 1 + the weighted accumulate
//   into the output y [T, H]. Consumes the act produced by the pass 1.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "habanalabs/hpu_custom_op.h"

namespace {

constexpr const char* schema = "custom_op::block_fp8_moe_w2_glm";
constexpr const char* guid = "block_fp8_moe_w2_glm";

void validate(const at::Tensor& w2, const at::Tensor& w2_scales,
              const at::Tensor& act, const at::Tensor& topk_ids,
              const at::Tensor& topk_weights) {
    TORCH_CHECK((w2.scalar_type() == at::kFloat8_e4m3fnuz || w2.scalar_type() == at::kFloat8_e4m3fn) &&
                w2_scales.scalar_type() == at::kFloat &&
                act.scalar_type() == at::kBFloat16 &&
                topk_ids.scalar_type() == at::kInt &&
                topk_weights.scalar_type() == at::kFloat &&
                w2.dim() == 3 && w2_scales.dim() == 3 && act.dim() == 3 &&
                topk_ids.dim() == 2 && topk_weights.dim() == 2 &&
                w2.size(1) % 128 == 0 && w2.size(2) % 256 == 0 &&
                w2_scales.size(0) == w2.size(0) &&
                w2_scales.size(1) == w2.size(1) / 128 &&
                w2_scales.size(2) == w2.size(2) / 128 &&
                act.size(2) == w2.size(2) &&
                act.size(1) == topk_ids.size(1) &&
                topk_ids.size(0) == act.size(0) &&
                w2.is_contiguous() && w2_scales.is_contiguous() &&
                act.is_contiguous() && topk_ids.is_contiguous() && topk_weights.is_contiguous(),
                "block_fp8_moe_w2_glm requires contiguous E4M3 [E,H,I], F32 [E,H/128,I/128] scales, BF16 [T,K,I] act, int32 [T,K] topk_ids, F32 [T,K] topk_weights");
}

std::once_flag reg_flag;
bool reg_ok = false;

void lazy_register() {
    std::call_once(reg_flag, []() {
        habana::custom_op::registerUserCustomOp(
            schema, guid,
            [](const at::Stack& stack) {
                auto act = stack.at(2).toTensor();
                auto w2 = stack.at(0).toTensor();
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {act.size(0), w2.size(1)};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}

at::Tensor run(const at::Tensor& w2, const at::Tensor& w2_scales,
               const at::Tensor& act, const at::Tensor& topk_ids,
               const at::Tensor& topk_weights) {
    validate(w2, w2_scales, act, topk_ids, topk_weights);
    lazy_register();
    TORCH_CHECK(reg_ok, "block_fp8_moe_w2_glm registration failed");
    TORCH_CHECK(w2.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{w2, w2_scales, act, topk_ids, topk_weights};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("block_fp8_moe_w2_glm(Tensor w2, Tensor w2_scales, Tensor act, Tensor topk_ids, Tensor topk_weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("block_fp8_moe_w2_glm",
           [](const at::Tensor& w2, const at::Tensor& w2_scales, const at::Tensor& act,
              const at::Tensor& topk_ids, const at::Tensor& topk_weights) -> at::Tensor {
               TORCH_CHECK(false, "block_fp8_moe_w2_glm requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("block_fp8_moe_w2_glm",
           [](const at::Tensor& w2, const at::Tensor& w2_scales, const at::Tensor& act,
              const at::Tensor& topk_ids, const at::Tensor& topk_weights) -> at::Tensor {
               return run(w2, w2_scales, act, topk_ids, topk_weights);
           });
}
