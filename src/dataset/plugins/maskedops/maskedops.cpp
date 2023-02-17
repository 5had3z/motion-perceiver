#include "maskedops.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

/**
 * @brief Waymo Motion format is a contigous [n_instances, timestep] so as a contigous
 *
 */

namespace maskedop
{

[[nodiscard]] std::size_t nElemSampleView(::dali::ConstSampleView<::dali::CPUBackend> tensor)
{
    const auto tensorShape = tensor.shape();
    // Product of all the dimensions is the full size of the tensor
    return std::accumulate(tensorShape.begin(), tensorShape.end(), 1, std::multiplies<int>());
}

/**
 * @brief Currently without std::views::zip, I can't create a view of the input based on mask values
 *        Once that is added to the standard, I can do something like zip | filter | minmax_element
 *        The mask value is true if the input is valid.
 */
std::pair<float, float> findMinMax(
    ::dali::ConstSampleView<::dali::CPUBackend> inputTensor, ::dali::ConstSampleView<::dali::CPUBackend> maskTensor)
{
    auto minMax = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());

    const auto maxElem = nElemSampleView(inputTensor);
    assert(maxElem == nElemSampleView(maskTensor) && "input and mask have different element counts");

    for (std::size_t cIdx = 0; cIdx < maxElem; cIdx++)
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

void normalize(::dali::ConstSampleView<::dali::CPUBackend> inputTensor,
    ::dali::ConstSampleView<::dali::CPUBackend> maskTensor, ::dali::SampleView<::dali::CPUBackend> outputTensor,
    const float normalize)
{
    // Find min max of the sequence
    const auto [min, max] = findMinMax(inputTensor, maskTensor);

    // Calculate the center of the map
    const auto center = (max + min) / 2;

    const auto maxElem = nElemSampleView(inputTensor);

    // Can maybe use std::span and generic algorithm for this
    for (std::size_t cIdx = 0; cIdx < maxElem; cIdx++)
    {
        // Subtract the center of the map and divide by normalization factor
        outputTensor.mutable_data<float>()[cIdx] = (inputTensor.data<float>()[cIdx] - center) / normalize;
    }
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
        tPool.AddWork([&, sampleId](int thread_id)
            { normalize(input[sampleId], mask[sampleId], output[sampleId], mNormalizeScalar); });
    }
    tPool.RunAll();
}

void median(::dali::ConstSampleView<::dali::CPUBackend> inputTensor,
    ::dali::ConstSampleView<::dali::CPUBackend> maskTensor, ::dali::SampleView<::dali::CPUBackend> outputTensor)
{
    const auto [min, max] = findMinMax(inputTensor, maskTensor);

    // Calculate the center of the map
    *outputTensor.mutable_data<float>() = (max + min) / 2;
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
        tPool.AddWork([&, sampleId](int thread_id) { median(input[sampleId], mask[sampleId], output[sampleId]); });
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
