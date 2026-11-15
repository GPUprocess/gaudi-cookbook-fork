// SPDX-License-Identifier: Apache-2.0
// store_test - the PT2 custom-op with the LAZY registration (the lane-id dump).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* schema = "custom_op::store_test";
constexpr const char* guid = "store_test";

void validate(const at::Tensor& src) {
    TORCH_CHECK(src.scalar_type() == at::kBFloat16 &&
                src.dim() == 2 && src.size(0) == 1 && src.size(1) == 256 && src.is_contiguous(),
                "store_test requires contiguous BF16 [1,256]");
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
                output.shape = {(int64_t)4, (int64_t)8, (int64_t)512};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}

at::Tensor run(const at::Tensor& src) {
    validate(src);
    lazy_register();
    TORCH_CHECK(reg_ok, "store_test registration failed");
    TORCH_CHECK(src.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{src};
    return descriptor.execute(inputs)[0];
}

} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("store_test(Tensor src) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("store_test",
           [](const at::Tensor& src) -> at::Tensor {
               TORCH_CHECK(false, "store_test requires HPU tensors");
               return at::Tensor();
           });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("store_test",
           [](const at::Tensor& src) -> at::Tensor {
               return run(src);
           });
}
