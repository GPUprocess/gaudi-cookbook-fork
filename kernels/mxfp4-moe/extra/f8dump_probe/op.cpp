// SPDX-License-Identifier: Apache-2.0
// f8dump_probe - the PT2 custom-op with the LAZY registration (the lane-id dump).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* schema = "custom_op::f8dump_probe";
constexpr const char* guid = "f8dump_probe";

void validate(const at::Tensor& src) {
    TORCH_CHECK(src.scalar_type() == at::kFloat8_e4m3fn &&
                src.dim() == 3 && src.size(0) == 8 && src.size(1) == 512 && src.size(2) == 512 && src.is_contiguous(),
                "f8dump_probe requires contiguous BF16 [1,256]");
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
    TORCH_CHECK(reg_ok, "f8dump_probe registration failed");
    TORCH_CHECK(src.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{src};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("f8dump_probe(Tensor src) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("f8dump_probe",
           [](const at::Tensor& src) -> at::Tensor {
               TORCH_CHECK(false, "f8dump_probe requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("f8dump_probe",
           [](const at::Tensor& src) -> at::Tensor {
               return run(src);
           });
}
