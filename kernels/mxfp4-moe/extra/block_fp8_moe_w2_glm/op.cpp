// SPDX-License-Identifier: Apache-2.0
// block_fp8_moe_w2_glm — THE LAZY-NATIVE REGISTRATION
// (REGISTER_CUSTOM_OP_ATTRIBUTES, hpu_custom_op.h): the op is a LAZY GRAPH NODE —
// the inputs stay lazy (the serving graph's internal activations flow natively).
// The invocation: torch.ops.custom_op.block_fp8_moe_w2_glm(...) — the schema only.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op.h"

namespace {

using habana::custom_op::InputDesc;
using habana::custom_op::input_type;
using habana::custom_op::OutputDesc;

// the w2 output: [T, H]
std::vector<int64_t> y_shape(const at::Stack& st) {
    auto act = st.at(2).toTensor();  // the act_q [T, K, I]: the T from the dim 0
    auto w = st.at(0).toTensor();
    return {act.size(0), w.size(1)};
}

habana::custom_op::HabanaCustomOpDescriptor make_desc() {
    std::vector<InputDesc> inputs{
        InputDesc{input_type::TENSOR, 0}, InputDesc{input_type::TENSOR, 1},
        InputDesc{input_type::TENSOR, 2}, InputDesc{input_type::TENSOR, 3},
        InputDesc{input_type::TENSOR, 4}};
    std::vector<OutputDesc> outputs{
        OutputDesc{0, c10::ScalarType::BFloat16, y_shape}};
    habana::custom_op::NodeDesc node_desc{"block_fp8_moe_w2_glm", "custom_op::block_fp8_moe_w2_glm", nullptr};
    return habana::custom_op::HabanaCustomOpDescriptor{node_desc, inputs, outputs};
}
static habana::custom_op::HabanaCustomOpDescriptor g_desc = make_desc();

struct LazyReg {
    LazyReg() {
        habana::custom_op::registerKernel(g_desc);
    }
};
static LazyReg _lazy_reg;

} // namespace anon_q21

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("block_fp8_moe_w2_glm(Tensor w2, Tensor w2_scales, Tensor act_q, Tensor act_scales, Tensor topk_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("block_fp8_moe_w2_glm",
           [](const at::Tensor& w2, const at::Tensor& w2_scales,
              const at::Tensor& act_q, const at::Tensor& act_scales, const at::Tensor& topk_ids) -> at::Tensor {
               // THE LAZY-NODE ENQUEUE: the lazy-registered op's descriptor execute
               // handles the lazy inputs natively (no materialization).
               auto d = g_desc;
               std::vector<c10::IValue> inputs{w2, w2_scales, act_q, act_scales, topk_ids};
               return d.execute(inputs)[0];
           });
}
