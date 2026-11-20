// CS probe op - TEMPLATE (schema/guid substituted per probe).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {
constexpr const char* schema = "custom_op::cs1";
constexpr const char* guid = "cs1";

void validate(const at::Tensor& w13, const at::Tensor& w13_scales,
              const at::Tensor& x, const at::Tensor& topk_ids) {
    TORCH_CHECK((w13.scalar_type() == at::kFloat8_e4m3fn) &&
                w13_scales.scalar_type() == at::kFloat &&
                x.scalar_type() == at::kBFloat16 &&
                topk_ids.scalar_type() == at::kInt &&
                w13.dim() == 3 && x.dim() == 2 && topk_ids.dim() == 2 &&
                x.is_contiguous() && topk_ids.is_contiguous(),
                "cs1 requires E4M3 [E,2I,H], F32 scales, BF16 [T,H] x, int32 [T,K] ids");
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
                output.shape = {x.size(0), topk_ids.size(1), w13.size(1)};
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
    TORCH_CHECK(reg_ok, "cs1 registration failed");
    TORCH_CHECK(w13.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{w13, w13_scales, x, topk_ids};
    return descriptor.execute(inputs)[0];
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("cs1(Tensor w13, Tensor w13_scales, Tensor x, Tensor topk_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("cs1",
           [](const at::Tensor& w13, const at::Tensor& w13_scales, const at::Tensor& x,
              const at::Tensor& topk_ids) -> at::Tensor {
               return run(w13, w13_scales, x, topk_ids);
           });
}
