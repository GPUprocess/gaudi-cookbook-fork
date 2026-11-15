// SPDX-License-Identifier: Apache-2.0
// macprobe2 - the PT2 custom-op with the LAZY registration (the lane-id dump).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* schema = "custom_op::macprobe2";
constexpr const char* guid = "macprobe2";

void validate(const at::Tensor& src) {
    TORCH_CHECK(src.scalar_type() == at::kBFloat16 &&
                src.dim() == 2 && src.size(0) == 4 && src.size(1) == 512 && src.is_contiguous(),
                "macprobe2 requires contiguous BF16 [4,512]");
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
    TORCH_CHECK(reg_ok, "macprobe2 registration failed");
    TORCH_CHECK(src.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{src};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("macprobe2(Tensor src) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("macprobe2",
           [](const at::Tensor& src) -> at::Tensor {
               TORCH_CHECK(false, "macprobe2 requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("macprobe2",
           [](const at::Tensor& src) -> at::Tensor {
               return run(src);
           });
}
