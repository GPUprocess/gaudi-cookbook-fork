// SPDX-License-Identifier: Apache-2.0
// block_fp8_moe_w13_glm - PT2 custom-op with LAZY registration (Q21 step 2, pass 1).
// Same lazy-registration pattern as block_fp8_gemv_glm (the step 1): the
// registration defers to the first call so it runs after the backend is fully
// initialized inside the serving process (the B7 load-order recipe).
//
//   act[T, K, I] = SiLU(W13[e_k, 2I, H] · x[T, H] / s13[e_k])
//
//   The pass 1 of the two-pass fused MoE: for each token and each of its K=8
//   routed experts, the dequant (the FP8 → the BF16 with the block scales)
//   + the GEMV over H + the SiLU on the gate columns. The output act [T, K, I]
//   is consumed by the pass 2 (block_fp8_moe_w2_glm).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "habanalabs/hpu_custom_op.h"

namespace {

constexpr const char* schema = "custom_op::block_fp8_moe_w13_glm";
constexpr const char* guid = "block_fp8_moe_w13_glm";

void validate(const at::Tensor& w13, const at::Tensor& w13_scales,
              const at::Tensor& x, const at::Tensor& topk_ids) {
    TORCH_CHECK((w13.scalar_type() == at::kFloat8_e4m3fnuz || w13.scalar_type() == at::kFloat8_e4m3fn) &&
                w13_scales.scalar_type() == at::kFloat &&
                x.scalar_type() == at::kBFloat16 &&
                topk_ids.scalar_type() == at::kInt &&
                w13.dim() == 3 && w13_scales.dim() == 3 && x.dim() == 2 && topk_ids.dim() == 2 &&
                w13.size(1) % 128 == 0 && w13.size(2) % 256 == 0 &&
                w13_scales.size(0) == w13.size(0) &&
                w13_scales.size(1) == w13.size(1) / 128 &&
                w13_scales.size(2) == w13.size(2) / 128 &&
                x.size(1) == w13.size(2) &&
                topk_ids.size(0) == x.size(0) &&
                w13.is_contiguous() && w13_scales.is_contiguous() &&
                x.is_contiguous() && topk_ids.is_contiguous() &&
                w13.device() == w13_scales.device() && w13.device() == x.device(),
                "block_fp8_moe_w13_glm requires contiguous E4M3 [E,2I,H], F32 [E,2I/128,H/128] scales, BF16 [T,H] x, int32 [T,K] topk_ids");
}

std::once_flag reg_flag;
bool reg_ok = false;

void lazy_register() {
    std::call_once(reg_flag, []() {
        habana::custom_op::registerUserCustomOp(
            schema, guid,
            [](const at::Stack& stack) {
                auto x = stack.at(2).toTensor();
                auto topk_ids = stack.at(3).toTensor();
                auto w13 = stack.at(0).toTensor();
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {x.size(0), topk_ids.size(1), w13.size(1) / 2};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}

at::Tensor run(const at::Tensor& w13, const at::Tensor& w13_scales,
               const at::Tensor& x, const at::Tensor& topk_ids) {
    validate(w13, w13_scales, x, topk_ids);
    lazy_register();
    TORCH_CHECK(reg_ok, "block_fp8_moe_w13_glm registration failed");
    TORCH_CHECK(w13.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    const int64_t T = x.size(0);
    const int64_t K = topk_ids.size(1);
    const int64_t I = w13.size(1) / 2;
    auto act = at::empty({T, K, I}, x.options().dtype(at::kBFloat16));
    std::vector<c10::IValue> inputs{w13, w13_scales, x, topk_ids, act};
    descriptor.execute(inputs);
    return act;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("block_fp8_moe_w13_glm(Tensor w13, Tensor w13_scales, Tensor x, Tensor topk_ids, Tensor act) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("block_fp8_moe_w13_glm",
           [](const at::Tensor& w13, const at::Tensor& w13_scales, const at::Tensor& x,
              const at::Tensor& topk_ids, const at::Tensor& act) -> at::Tensor {
               TORCH_CHECK(false, "block_fp8_moe_w13_glm requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("block_fp8_moe_w13_glm",
           [](const at::Tensor& w13, const at::Tensor& w13_scales, const at::Tensor& x,
              const at::Tensor& topk_ids, const at::Tensor& act) -> at::Tensor {
               // The act: pre-allocated [T, K, I] BF16, filled by the kernel
               validate(w13, w13_scales, x, topk_ids);
               lazy_register();
               TORCH_CHECK(reg_ok, "block_fp8_moe_w13_glm registration failed");
               auto descriptor =
                   habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
               std::vector<c10::IValue> inputs{w13, w13_scales, x, topk_ids, act};
               descriptor.execute(inputs);
               return act;
           });
}
