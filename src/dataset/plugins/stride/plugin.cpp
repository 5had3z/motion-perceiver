#include "plugin.hpp"

#include <dali/core/static_switch.h>
#include <dali/kernels/common/utils.h>

#define SLICE_TYPES (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, bool)
#define STRIDE_SUPPORTED_NDIMS (1, 2, 3, 4, 5)

namespace sliceop
{
using ConstTensor = dali::ConstSampleView<dali::CPUBackend>;
using Tensor = dali::SampleView<dali::CPUBackend>;

template <typename OutputType, typename InputType, int DimsLeft>
void StrideRecurse(const InputType* input, OutputType* output, const int64_t* inStride, const int64_t* outStride,
    const int64_t* inShape, const int64_t* outShape, std::integral_constant<int, DimsLeft>)
{
    static_assert(DimsLeft > 0, "Dims less than 0");
    if constexpr (DimsLeft > 1)
    {
        for (int64_t inIdx = 0, outIdx = 0; inIdx < *inShape && outIdx < *outShape; inIdx++, outIdx++)
        {
            StrideRecurse(input, output, inStride + 1, outStride + 1, inShape + 1, outShape + 1,
                std::integral_constant<int, DimsLeft - 1>());
            output += *outStride;
            input += *inStride;
        }
    }
    else
    {
        for (int64_t inIdx = 0, outIdx = 0; inIdx < *inShape && outIdx < *outShape;
             inIdx += *inStride, outIdx += *outStride)
        {
            output[outIdx] = static_cast<OutputType>(input[inIdx]);
        }
    }
}

void StrideImpl(ConstTensor in, Tensor out, int64_t dim, int64_t stride) noexcept
{

    const auto& inShape = in.shape();
    const auto nDim = inShape.sample_dim();
    auto inStrides = dali::kernels::GetStrides(inShape);

    const auto& outShape = out.shape();
    auto outStrides = dali::kernels::GetStrides(outShape);

    inStrides[dim] *= stride; // Take larger steps at this level

    // clang-format off
    TYPE_SWITCH(in.type(), dali::type2id, T, SLICE_TYPES, (
        VALUE_SWITCH(nDim, Dims, STRIDE_SUPPORTED_NDIMS, (
            StrideRecurse<T, T, Dims>(in.data<T>(), out.mutable_data<T>(),
                inStrides.data(), outStrides.data(), inShape.data(), outShape.data(),
                std::integral_constant<int, Dims>());
        ), DALI_FAIL(dali::make_string("Unsupported Ndims: ", nDim)));
    ), DALI_FAIL(dali::make_string("Unsupported Type: ", in.type())));
    // clang-format on

    // auto logshape = [](dali::TensorShape<-1> s)
    // {
    //     std::stringstream ss;
    //     for (auto&& e : s)
    //     {
    //         ss << e << ", ";
    //     }
    //     return ss.str();
    // };
    // std::cout << "strides in: " << logshape(inStrides) << " out: " << logshape(outStrides)
    //           << "shape in: " << logshape(inShape) << " out: " << logshape(outShape) << std::endl;
}

template <>
void StrideSliceOp<dali::CPUBackend>::RunImpl(dali::Workspace& ws)
{
    auto& tPool = ws.GetThreadPool();

    DALI_ENFORCE(ws.NumInput() == 1, "Should have singlular input");

    const auto& input = ws.Input<dali::CPUBackend>(0);
    auto& output = ws.Output<dali::CPUBackend>(0);
    for (int sIdx = 0; sIdx < ws.GetInputBatchSize(0); ++sIdx)
    {
        tPool.AddWork([&, sIdx](int) { StrideImpl(input[sIdx], output[sIdx], mDimIdx, mStride); });
    }

    tPool.RunAll();
}

} // namespace sliceop

DALI_REGISTER_OPERATOR(StrideSlice, sliceop::StrideSliceOp<dali::CPUBackend>, dali::CPU);

DALI_SCHEMA(StrideSlice)
    .DocStr("Slice along axis with stride")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .AddArg("axis", "axis to slice", dali::DALIDataType::DALI_INT64)
    .AddArg("stride", "stride of slicing", dali::DALIDataType::DALI_INT64);