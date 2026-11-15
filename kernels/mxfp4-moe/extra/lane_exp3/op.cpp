#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"
namespace {
constexpr const char* schema = "custom_op::lane_exp3";
constexpr const char* guid = "lane_exp3";
std::once_flag reg_flag;
bool reg_ok = false;
void lazy_register() {
    std::call_once(reg_flag, []() {
        habana::custom_op::registerUserCustomOp(
            schema, guid,
            [](const at::Stack& stack) {
                habana::PartialOutputMetaData output;
                output.dtype = at::kBFloat16;
                output.shape = {4, 8, 512};
                return habana::PartialOutputMetaDataVector{output};
            },
            nullptr);
        reg_ok = true;
    });
}
at::Tensor run(const at::Tensor& w, const at::Tensor& ws, const at::Tensor& x, const at::Tensor& ids) {
    lazy_register();
    auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{w, ws, x, ids};
    return d.execute(inputs)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("lane_exp3(Tensor w, Tensor ws, Tensor x, Tensor ids) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
    m.impl("lane_exp3", [](const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) -> at::Tensor {
        TORCH_CHECK(false, "lane_exp3 requires HPU");
        return at::Tensor();
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("lane_exp3", [](const at::Tensor& w, const at::Tensor& ws, const at::Tensor& x, const at::Tensor& ids) -> at::Tensor {
        return run(w, ws, x, ids);
    });
}
