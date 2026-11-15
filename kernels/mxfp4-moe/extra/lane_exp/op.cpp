// SPDX-License-Identifier: Apache-2.0
// lane_exp - the PT2 custom-op with the LAZY registration (the lane-id dump).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* schema = "custom_op::lane_exp";
constexpr const char* guid = "lane_exp";

void validate(const at::Tensor& src) {
    TORCH_CHECK(src.scalar_type() == at::kBFloat16 &&
                src.dim() == 2 && src.size(0) == 2 && src.size(1) == 512 && src.is_contiguous(),
                "lane_exp requires contiguous BF16 [2,512]");
}

std::once_flag reg_flag;
bool reg_ok = false;

void lazy_register() {
    std::call_once(reg_flag, []() {
        habana::custom_op::registerUserCustomOp(
            schema, guid,
            [](const at::Stack& stack) {
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {1, 256};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}

at::Tensor run(const at::Tensor& src) {
    validate(src);
    lazy_register();
    TORCH_CHECK(reg_ok, "lane_exp registration failed");
    TORCH_CHECK(src.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{src};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("lane_exp(Tensor src) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("lane_exp",
           [](const at::Tensor& src) -> at::Tensor {
               TORCH_CHECK(false, "lane_exp requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("lane_exp",
           [](const at::Tensor& src) -> at::Tensor {
               return run(src);
           });
}
