#include "maskfrompoly.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

namespace occupancyop
{

using ConstDaliTensor = ::dali::ConstSampleView<::dali::CPUBackend>;
using DaliTensor = ::dali::SampleView<::dali::CPUBackend>;

/**
 * @brief sizeof(DALIDataType)
 *
 * @param d DALIDataType to query sizeof in bytes
 * @return std::size_t
 */
std::size_t daliType2size(dali::DALIDataType d)
{
    switch (d)
    {
    case dali::DALIDataType::DALI_INT16: return sizeof(int16_t);
    case dali::DALIDataType::DALI_INT32: return sizeof(int32_t);
    case dali::DALIDataType::DALI_INT64: return sizeof(int64_t);
    case dali::DALIDataType::DALI_FLOAT16: return sizeof(half);
    case dali::DALIDataType::DALI_FLOAT: return sizeof(float);
    case dali::DALIDataType::DALI_FLOAT64: return sizeof(double);
    default: throw std::runtime_error("Unidentified DALI type" + std::to_string(d));
    }
}

/**
 * @brief converts pose in pixel space to a polygon (ul, ur, bl, br)
 *
 * @param x
 * @param y
 * @param t
 * @param w
 * @param l
 * @return std::vector<cv::Point2i> polygon points
 */
std::vector<cv::Point2i> poseToPoly(float x, float y, float t, float w, float l) noexcept
{
    // clang-format off
    // 2x2 rotation matrix
    const cv::Matx22f rot = {
        std::cos(t), -std::sin(t),
        std::sin(t), std::cos(t)
    };
    l /= 2;
    w /= 2;
    // 2x4 bbox corners at 0,0
    const cv::Matx<float, 2, 4> points = {
        -l, l, l, -l,
        w, w, -w, -w
    };
    // clang-format on

    // Rotate points around 0,0
    const cv::Matx<float, 2, 4> rotated_box = rot * points;

    std::vector<cv::Point2i> polyPoints;
    polyPoints.reserve(4);
    for (auto col = 0; col < rotated_box.cols; ++col)
    {
        polyPoints.emplace_back(rotated_box(0, col) + x, rotated_box(1, col) + y);
    }
    return polyPoints;
}

/**
 * @brief Mask out future vehicles if they aren't present in the past or current frame(s)
 *        Tensor dimensions are [instance, time], assuming we have the entire mask tensor
 *        the current frame is index 11, so we check "any of" valid until 11.
 * @param maskTensor
 * @return std::vector<int>
 */
std::vector<int> maskInvalidFuture(ConstDaliTensor maskTensor) noexcept
{
    std::vector<int> newMask;
    const auto outputDims = maskTensor.shape();
    const auto outputSize
        = std::reduce(std::execution::unseq, outputDims.begin(), outputDims.end(), 1L, std::multiplies<int64_t>());
    newMask.reserve(outputSize);

    // Iterate over each instance in the tensor
    for (int64_t inst_id = 0; inst_id < outputDims.shape[0]; ++inst_id)
    {
        const auto instStart = maskTensor.data<int>() + inst_id * outputDims.shape[1];

        // Check if the target is valid at any point until "current"
        const auto observeable
            = std::any_of(std::execution::unseq, instStart, instStart + 11, [](auto elem) { return elem > 0; });

        // Get iterator to start of the instance's mask
        if (observeable)
        {
            const auto instEnd = instStart + outputDims.shape[1];
            std::copy(std::execution::unseq, instStart, instEnd, std::back_inserter(newMask));
        }
        else
        {
            // Append time dimension of zeroes
            newMask.resize(newMask.size() + outputDims.shape[1], 0);
        }
    }

    return newMask;
}

/**
 * @brief Create a Heatmap Image at a time interval
 *
 * @param xTensor
 * @param yTensor
 * @param tTensor
 * @param wTensor
 * @param lTensor
 * @param maskTensor
 * @param outputTensor Class, Timestep, Height, Width
 * @param timeIdx
 */
void createHeatmapImage(ConstDaliTensor xTensor, ConstDaliTensor yTensor, ConstDaliTensor tTensor,
    ConstDaliTensor wTensor, ConstDaliTensor lTensor, ConstDaliTensor maskTensor, ConstDaliTensor classTensor,
    std::size_t inTimeIdx, DaliTensor outputTensor, std::size_t outTimeIdx, double roiScale,
    bool separateClasses) noexcept
{
    const auto outputDims = outputTensor.shape();
    const auto outputStride = outputDims[2] * outputDims[3] * daliType2size(outputTensor.type());
    const auto timeStride = outputStride * outputDims[1];

    // Create vector of cv::Mat views of class image tensors
    std::vector<cv::Mat> heatmapImages;
    for (auto idx = 0; idx < outputDims[0]; ++idx)
    {
        heatmapImages.emplace_back(outputDims[2], outputDims[3], CV_32F,
            outputTensor.raw_mutable_data() + idx * timeStride + outputStride * outTimeIdx);
        heatmapImages.back().setTo(0);
    }

    const auto instanceDim = xTensor.shape()[0];
    const auto timeDim = xTensor.shape()[1];

    for (int64_t instanceId = 0; instanceId < instanceDim; ++instanceId)
    {
        const auto cIdx = instanceId * timeDim + inTimeIdx;
        // If valid sample
        if (maskTensor.data<int>()[cIdx])
        {
            const int classIdx = classTensor.data<float>()[cIdx] - 1;
            if (classIdx < 0 || classIdx > 2)
                continue; // skip class that isn't vehicle, pedestrian, cyclist

            // Scaling factor from normalized coordinates to pixel coordinates
            const float xScale = outputDims[3] / 2.f;
            const float yScale = outputDims[2] / 2.f;

            // Calulate bbox in image coordinates
            // transform absolute position from -1,1 to 0,1 before pixel scaling factor
            const float x = (xTensor.data<float>()[cIdx] / roiScale + 1.f) * xScale;
            const float y = (yTensor.data<float>()[cIdx] / roiScale + 1.f) * yScale;
            const float angle = tTensor.data<float>()[cIdx];
            const float width = wTensor.data<float>()[cIdx] / roiScale * xScale;
            const float length = lTensor.data<float>()[cIdx] / roiScale * yScale;

            // Find polyPoints
            auto polyPoints = poseToPoly(x, y, angle, width, length);

            // Apply to image
            cv::fillConvexPoly(heatmapImages[separateClasses ? classIdx : 0], polyPoints, cv::Scalar(1));
        }
    }
}

/**
 * @brief Run over multiple time indexes
 *
 */
void createHeatMapImageMulti(ConstDaliTensor xTensor, ConstDaliTensor yTensor, ConstDaliTensor tTensor,
    ConstDaliTensor wTensor, ConstDaliTensor lTensor, ConstDaliTensor maskTensor, ConstDaliTensor classTensor,
    std::vector<int64_t> timeIdxs, DaliTensor outputTensor, bool filterFuture, double roiScale,
    bool separateClasses) noexcept
{
    const std::vector<int> futureMask = filterFuture ? maskInvalidFuture(maskTensor) : std::vector<int>();

    auto mask_ = filterFuture ? ConstDaliTensor(futureMask.data(), maskTensor.shape()) : maskTensor;

    for (std::size_t outputIdx = 0; outputIdx < timeIdxs.size(); ++outputIdx)
    {
        createHeatmapImage(xTensor, yTensor, tTensor, wTensor, lTensor, mask_, classTensor, timeIdxs[outputIdx],
            outputTensor, outputIdx, roiScale, separateClasses);
    }
}

std::vector<int64_t> generateTimeIdxs(
    std::vector<int64_t>& constTime, std::uniform_int_distribution<> randDist, int64_t randCount) noexcept
{
    // Use set since we're operating on a few elements and hash cost would be relatively high.
    std::set<int64_t> timeIdxs;
    for (auto&& tidx : constTime)
    {
        timeIdxs.emplace(tidx);
    }

    std::random_device dev;
    std::mt19937 randGen(dev());
    // random sort vector and sample would be guaranteed O(N) vs this which is O(inf)
    while (timeIdxs.size() < randCount + constTime.size())
    {
        timeIdxs.emplace(randDist(randGen));
    }

    return {timeIdxs.begin(), timeIdxs.end()};
}

template <>
void OccupancyMaskGenerator<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& xTensor = ws.Input<::dali::CPUBackend>(0);
    const auto& yTensor = ws.Input<::dali::CPUBackend>(1);
    const auto& tTensor = ws.Input<::dali::CPUBackend>(2);
    const auto& wTensor = ws.Input<::dali::CPUBackend>(3);
    const auto& lTensor = ws.Input<::dali::CPUBackend>(4);
    const auto& maskTensor = ws.Input<::dali::CPUBackend>(5);
    const auto& classTensor = ws.Input<::dali::CPUBackend>(6);

    auto& outputTensor = ws.Output<::dali::CPUBackend>(0);
    auto& outputTimeIdx = ws.Output<::dali::CPUBackend>(1);
    auto outputTimeIdxType = outputTimeIdx.type_info();

    auto& tPool = ws.GetThreadPool();
    const auto& inShape = xTensor.shape();

    const auto timeIdx = generateTimeIdxs(mConstTimeIndex, mRandIdx, mRandIdxCount);

    for (int sampleId = 0; sampleId < inShape.num_samples(); ++sampleId)
    {
        tPool.AddWork(
            [&, sampleId](int thread_id)
            {
                createHeatMapImageMulti(xTensor[sampleId], yTensor[sampleId], tTensor[sampleId], wTensor[sampleId],
                    lTensor[sampleId], maskTensor[sampleId], classTensor[sampleId], timeIdx, outputTensor[sampleId],
                    mFilterFuture, mROIScale, mSeparateClasses);
                outputTimeIdxType.Copy<::dali::CPUBackend, ::dali::CPUBackend>(
                    outputTimeIdx.raw_mutable_tensor(sampleId), timeIdx.data(), timeIdx.size(), 0);
            });
    }
    tPool.RunAll();
}
} // namespace occupancyop

DALI_REGISTER_OPERATOR(OccupancyMask, ::occupancyop::OccupancyMaskGenerator<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(OccupancyMask)
    .DocStr(
        "Generates occupancy mask from x,y,t,l,w at different time points where the inputs x,y already normalised "
        "between [-1,1] from masked normalise step and l,w are normalized with the same scaling factor, also returns "
        "the time index of the sample")
    .NumInput(7)
    .NumOutput(2)
    .AddArg("size", "Size of the output image, the image is always square", ::dali::DALIDataType::DALI_INT64)
    .AddArg("roi", "Scale to fraction of ROI observeable, centered", ::dali::DALIDataType::DALI_FLOAT)
    .AddArg("const_time_idx", "Time indicies to always sample from", ::dali::DALIDataType::DALI_INT_VEC)
    .AddArg("separate_classes", "Separate classes occupancy into different channels", ::dali::DALIDataType::DALI_BOOL)
    .AddArg("filter_future", "Filter ids that don't appear in past or current frame", ::dali::DALIDataType::DALI_BOOL)
    .AddOptionalArg("min_random_idx", "Minimum random time index to sample", 0)
    .AddOptionalArg("max_random_idx", "Maximum random time index to sample", 0)
    .AddOptionalArg(
        "n_random_idx", "Number of random time idx to sample from with replacement", static_cast<int64_t>(0));
