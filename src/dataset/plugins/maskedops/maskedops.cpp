#include "maskedops.hpp"

#include <algorithm>
#include <execution>
#include <functional>
#include <limits>
#include <span>
#include <utility>

/**
 * @brief Waymo Motion format is a contigous [n_instances, timestep] so as a contigous
 *
 */

namespace maskedop
{

using ConstTensor = dali::ConstSampleView<dali::CPUBackend>;
using Tensor = dali::SampleView<dali::CPUBackend>;

[[nodiscard]] int64_t nElemSampleView(ConstTensor tensor)
{
    const auto tensorShape = tensor.shape();
    // Product of all the dimensions is the full size of the tensor
    return std::reduce(std::execution::unseq, tensorShape.begin(), tensorShape.end(), 1LL, std::multiplies<int64_t>());
}

/**
 * @brief Currently without std::views::zip, I can't create a view of the input based on mask values
 *        Once that is added to the standard, I can do something like zip | filter | minmax_element
 *        The mask value is true if the input is valid.
 */
std::pair<float, float> findMinMax(ConstTensor inputTensor, ConstTensor maskTensor)
{
    auto minMax = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());

    const auto maxElem = nElemSampleView(inputTensor);
    assert(maxElem == nElemSampleView(maskTensor) && "input and mask have different element counts");

    for (int64_t cIdx = 0; cIdx < maxElem; cIdx++)
    {
        if (maskTensor.data<int>()[cIdx])
        {
            const auto sample = inputTensor.data<float>()[cIdx];
            if (minMax.first > sample)
            {
                minMax.first = sample;
            }
            if (minMax.second < sample)
            {
                minMax.second = sample;
            }
        }
    }

    assert(minMax.first != std::numeric_limits<float>::max() && minMax.second != std::numeric_limits<float>::lowest()
        && "No valid data to calculate min/max");

    return minMax;
}

void normalize(ConstTensor inputTensor, ConstTensor maskTensor, Tensor outputTensor, const float normalize)
{
    // Find min max of the sequence
    const auto [min, max] = findMinMax(inputTensor, maskTensor);

    // Calculate the center of the map
    const auto center = (max + min) / 2;

    std::span inputView(inputTensor.data<float>(), static_cast<std::size_t>(nElemSampleView(inputTensor)));
    std::transform(std::execution::unseq, inputView.begin(), inputView.end(), outputTensor.mutable_data<float>(),
        [=](float input) { return (input - center) / normalize; });
}

template <>
void Normalize<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& input = ws.Input<::dali::CPUBackend>(0);
    const auto& mask = ws.Input<::dali::CPUBackend>(1);
    auto& output = ws.Output<::dali::CPUBackend>(0);

    auto& tPool = ws.GetThreadPool();
    const auto& inShape = input.shape();

    for (int sampleId = 0; sampleId < inShape.num_samples(); sampleId++)
    {
        tPool.AddWork(
            [&, sampleId](int) { normalize(input[sampleId], mask[sampleId], output[sampleId], mNormalizeScalar); });
    }
    tPool.RunAll();
}

void median(ConstTensor inputTensor, ConstTensor maskTensor, Tensor outputTensor)
{
    const auto [min, max] = findMinMax(inputTensor, maskTensor);

    // Calculate the center of the map
    outputTensor.mutable_data<float>()[0] = (max + min) / 2;
}

template <>
void Median<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& input = ws.Input<::dali::CPUBackend>(0);
    const auto& mask = ws.Input<::dali::CPUBackend>(1);
    auto& output = ws.Output<::dali::CPUBackend>(0);

    auto& tPool = ws.GetThreadPool();
    const auto& inShape = input.shape();

    for (int sampleId = 0; sampleId < inShape.num_samples(); sampleId++)
    {
        tPool.AddWork([&, sampleId](int) { median(input[sampleId], mask[sampleId], output[sampleId]); });
    }
    tPool.RunAll();
}

} // namespace maskedop

DALI_REGISTER_OPERATOR(MaskedNormalize, ::maskedop::Normalize<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(MaskedNormalize)
    .DocStr("Centers data and divides by normalisation value w/ ignore mask")
    .NumInput(2)
    .NumOutput(1)
    .AddArg("normalize_value", "Input is divided by this value to normalize", ::dali::DALIDataType::DALI_FLOAT);

DALI_REGISTER_OPERATOR(MaskedMedian, ::maskedop::Median<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(MaskedMedian).DocStr("Finds Median of data /w ignore mask").NumInput(2).NumOutput(1);
