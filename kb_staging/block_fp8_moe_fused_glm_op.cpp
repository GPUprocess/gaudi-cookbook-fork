// SPDX-License-Identifier: Apache-2.0
// block_fp8_moe_fused_glm - PT2 custom-op with LAZY registration (Q21 step 2).
// Same lazy-registration pattern as block_fp8_gemv_glm (the step 1) and
// block_fp8_dequant_glm: the registration defers to the first call so it runs
// after the backend is fully initialized inside the serving process (the B7
// load-order recipe).
//
//   y[T,H] = Σ_k w_k · (W2[e_k,H,I] · SiLU(W13[e_k,2I,H] · x[T,H] /s13) /s2)
//
//   The two-pass fused MoE: the pass 1 (the w13 dequant + GEMV + SiLU → the act
//   [T,K,I]) then the pass 2 (the w2 dequant + GEMV + the weighted accumulate
//   → the y [T,H]). Replaces the stock MoE dispatch chain (~18,200 recipe
//   nodes) with 2 TPC kernel launches per layer.
//
//   The TPC kernels: block_fp8_moe_w13_glm.c (the pass 1) and
//   block_fp8_moe_w2_glm.c (the pass 2).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "habanalabs/hpu_custom_op_pt2.h"

namespace {

// ---- The pass 1: the w13 dequant + GEMV + SiLU ----
constexpr const char* schema_w13 = "custom_op::block_fp8_moe_w13_glm";
constexpr const char* guid_w13 = "block_fp8_moe_w13_glm";

// ---- The pass 2: the w2 dequant + GEMV + the weighted accumulate ----
constexpr const char* schema_w2 = "custom_op::block_fp8_moe_w2_glm";
constexpr const char* guid_w2 = "block_fp8_moe_w2_glm";

void validate_common(const at::Tensor& x, const at::Tensor& topk_ids,
                     const at::Tensor& topk_weights) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                topk_ids.scalar_type() == at::kInt &&
                topk_weights.scalar_type() == at::kFloat &&
                x.dim() == 2 && topk_ids.dim() == 2 && topk_weights.dim() == 2 &&
                topk_ids.size(0) == x.size(0) &&
                topk_weights.size(0) == x.size(0) &&
                topk_ids.size(1) == topk_weights.size(1) &&
                x.is_contiguous() && topk_ids.is_contiguous() && topk_weights.is_contiguous() &&
                x.device() == topk_ids.device() && x.device() == topk_weights.device(),
                "block_fp8_moe_fused_glm requires contiguous BF16 [T,H] x, int32 [T,K] topk_ids, F32 [T,K] topk_weights");
}

void validate_w13(const at::Tensor& w13, const at::Tensor& w13_scales,
                  const at::Tensor& x, const at::Tensor& topk_ids) {
    TORCH_CHECK((w13.scalar_type() == at::kFloat8_e4m3fnuz || w13.scalar_type() == at::kFloat8_e4m3fn) &&
                w13_scales.scalar_type() == at::kFloat &&
                w13.dim() == 3 && w13_scales.dim() == 3 &&
                w13.size(1) % 128 == 0 && w13.size(2) % 256 == 0 &&
                w13_scales.size(0) == w13.size(0) &&
                w13_scales.size(1) == w13.size(1) / 128 &&
                w13_scales.size(2) == w13.size(2) / 128 &&
                w13.is_contiguous() && w13_scales.is_contiguous(),
                "block_fp8_moe_w13_glm requires contiguous E4M3 [E,2I,H] and F32 [E,2I/128,H/128] scales");
}

void validate_w2(const at::Tensor& w2, const at::Tensor& w2_scales,
                 const at::Tensor& act, const at::Tensor& topk_ids) {
    TORCH_CHECK((w2.scalar_type() == at::kFloat8_e4m3fnuz || w2.scalar_type() == at::kFloat8_e4m3fn) &&
                w2_scales.scalar_type() == at::kFloat &&
                w2.dim() == 3 && w2_scales.dim() == 3 &&
                w2.size(1) % 128 == 0 && w2.size(2) % 256 == 0 &&
                w2_scales.size(0) == w2.size(0) &&
                w2_scales.size(1) == w2.size(1) / 128 &&
                w2_scales.size(2) == w2.size(2) / 128 &&
                w2.is_contiguous() && w2_scales.is_contiguous() &&
                act.size(2) == w2.size(2),
                "block_fp8_moe_w2_glm requires contiguous E4M3 [E,H,I] and F32 [E,H/128,I/128] scales, BF16 [T,K,I] act");
}

std::once_flag reg_flag_w13;
std::once_flag reg_flag_w2;
bool reg_ok_w13 = false;
bool reg_ok_w2 = false;

void lazy_register_w13() {
    std::call_once(reg_flag_w13, []() {
        habana::custom_op::registerUserCustomOp(
            schema_w13, guid_w13,
            [](const at::Stack& stack) {
                auto act = stack.at(4).toTensor();
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {act.size(0), act.size(1), act.size(2)};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok_w13 = true;
    });
}

void lazy_register_w2() {
    std::call_once(reg_flag_w2, []() {
        habana::custom_op::registerUserCustomOp(
            schema_w2, guid_w2,
            [](const at::Stack& stack) {
                auto w2 = stack.at(0).toTensor();
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {w2.size(0), w2.size(1)};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok_w2 = true;
    });
}

at::Tensor run(const at::Tensor& x, const at::Tensor& w13, const at::Tensor& w13_scales,
               const at::Tensor& w2, const at::Tensor& w2_scales,
               const at::Tensor& topk_ids, const at::Tensor& topk_weights,
               int64_t expert_offset) {
    validate_common(x, topk_ids, topk_weights);
    validate_w13(w13, w13_scales, x, topk_ids);
    validate_w2(w2, w2_scales, x, topk_ids);
    lazy_register_w13();
    lazy_register_w2();
    TORCH_CHECK(reg_ok_w13 && reg_ok_w2, "the Q21 fused MoE registration failed");
    TORCH_CHECK(x.device().type() == at::kHPU);

    const int64_t T = x.size(0);
    const int64_t K = topk_ids.size(1);
    const int64_t I = w2.size(2);

    auto descriptor_w13 =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema_w13);
    auto descriptor_w2 =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema_w2);

    // ---- The pass 1: the w13 dequant + GEMV + SiLU → the act [T, K, I] ----
    auto act = at::empty({T, K, I}, x.options().dtype(at::kBFloat16));
    {
        std::vector<c10::IValue> inputs{w13, w13_scales, x, topk_ids, act};
        descriptor_w13.execute(inputs);
    }

    // ---- The pass 2: the w2 dequant + GEMV + the weighted accumulate → the y [T, H] ----
    auto y = at::empty({T, x.size(1)}, x.options().dtype(at::kBFloat16));
    {
        std::vector<c10::IValue> inputs{w2, w2_scales, act, topk_ids, topk_weights, y};
        descriptor_w2.execute(inputs);
    }

    return y;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("block_fp8_moe_w13_glm(Tensor w13, Tensor w13_scales, Tensor x, Tensor topk_ids, Tensor act) -> Tensor");
    m.def("block_fp8_moe_w2_glm(Tensor w2, Tensor w2_scales, Tensor act, Tensor topk_ids, Tensor topk_weights, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("block_fp8_moe_w13_glm",
           [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
              const at::Tensor&, const at::Tensor&) -> at::Tensor {
               TORCH_CHECK(false, "block_fp8_moe_w13_glm requires HPU tensors");
               return at::Tensor();
           });
    m.impl("block_fp8_moe_w2_glm",
           [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
              const at::Tensor&, const at::Tensor&, const at::Tensor&) -> at::Tensor {
               TORCH_CHECK(false, "block_fp8_moe_w2_glm requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("block_fp8_moe_w13_glm",
           [](const at::Tensor& w13, const at::Tensor& w13_scales, const at::Tensor& x,
              const at::Tensor& topk_ids, const at::Tensor& act) -> at::Tensor {
               validate_w13(w13, w13_scales, x, topk_ids);
               lazy_register_w13();
               TORCH_CHECK(reg_ok_w13, "the w13 registration failed");
               auto descriptor =
                   habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema_w13);
               std::vector<c10::IValue> inputs{w13, w13_scales, x, topk_ids, act};
               descriptor.execute(inputs);
               return act;
           });
    m.impl("block_fp8_moe_w2_glm",
           [](const at::Tensor& w2, const at::Tensor& w2_scales, const at::Tensor& act,
              const at::Tensor& topk_ids, const at::Tensor& topk_weights, const at::Tensor& y) -> at::Tensor {
               validate_w2(w2, w2_scales, act, topk_ids);
               lazy_register_w2();
               TORCH_CHECK(reg_ok_w2, "the w2 registration failed");
               auto descriptor =
                   habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema_w2);
               std::vector<c10::IValue> inputs{w2, w2_scales, act, topk_ids, topk_weights, y};
               descriptor.execute(inputs);
               return y;
           });
}

// ---- The fused entry point (the two-pass orchestration) ----

at::Tensor block_fp8_moe_fused_glm(
    const at::Tensor& x, const at::Tensor& w13, const at::Tensor& w13_scales,
    const at::Tensor& w2, const at::Tensor& w2_scales,
    const at::Tensor& topk_ids, const at::Tensor& topk_weights,
    int64_t expert_offset) {
    return run(x, w13, w13_scales, w2, w2_scales, topk_ids, topk_weights, expert_offset);
}
