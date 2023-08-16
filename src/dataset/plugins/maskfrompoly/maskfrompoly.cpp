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
    l /= 2.f;
    w /= 2.f;
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
        constexpr auto roundOffset = 0.5f;
        polyPoints.emplace_back(rotated_box(0, col) + x + roundOffset, rotated_box(1, col) + y + roundOffset);
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

struct stateData
{
    float x;
    float y;
    float yaw;
    float vx;
    float vy;
    float vyaw;
    float w;
    float l;
    float cls;
};

void renderBBox(const void* tensorData, int64_t index, std::vector<cv::Mat> rasterImage,
    const dali::TensorShape<-1>& outputDims, float roiScale, bool separateClasses) noexcept
{
    const auto data = static_cast<const stateData*>(tensorData)[index];

    const int classIdx = static_cast<int>(data.cls) - 1;
    if (classIdx < 0 || classIdx > 2)
        return; // skip class that isn't vehicle, pedestrian, cyclist

    // Scaling factor from normalized coordinates to pixel coordinates
    const float xScale = outputDims[3] / 2.f;
    const float yScale = outputDims[2] / 2.f;

    // Calulate bbox in image coordinates
    // transform absolute position from -1,1 to 0,1 before pixel scaling factor
    const float x = (data.x / roiScale + 1.f) * xScale;
    const float y = (data.y / roiScale + 1.f) * yScale;
    const float width = data.w / roiScale * xScale;
    const float length = data.l / roiScale * yScale;

    // Find polyPoints
    auto polyPoints = poseToPoly(x, y, data.yaw * M_PI, width, length);

    // Apply to image
    cv::fillConvexPoly(rasterImage[separateClasses ? classIdx : 0], polyPoints, cv::Scalar(1));
}

struct pedestrainData
{
    float x;
    float y;
    float t;
    float vx;
    float vy;
    float vt;
};

void renderCircle(const void* tensorData, int64_t index, cv::Mat rasterImage, const dali::TensorShape<-1>& outputDims,
    float roiScale, float circleRad) noexcept
{
    const auto data = static_cast<const pedestrainData*>(tensorData)[index];

    // Scaling factor from normalized coordinates to pixel coordinates
    const float xScale = outputDims[3] / 2.f;
    const float yScale = outputDims[2] / 2.f;

    // Calulate bbox in image coordinates
    // transform absolute position from -1,1 to 0,1 before pixel scaling factor
    const auto x = static_cast<int>((data.x / roiScale + 1.f) * xScale);
    const auto y = static_cast<int>((data.y / roiScale + 1.f) * yScale);

    cv::circle(rasterImage, cv::Point2i{x, y}, circleRad * xScale, cv::Scalar{1}, -1);
}

/**
 * @brief Create a Heatmap Image at a time interval
 *
 * @param dataTensor
 * @param maskTensor
 * @param outputTensor Class, Timestep, Height, Width
 * @param timeIdx
 */
void createHeatmapImage(ConstDaliTensor dataTensor, ConstDaliTensor maskTensor, std::size_t inTimeIdx,
    DaliTensor outputTensor, std::size_t outTimeIdx, float roiScale, float circleRad, bool separateClasses) noexcept
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

    const auto instanceDim = dataTensor.shape()[0];
    const auto timeDim = dataTensor.shape()[1];
    const bool isBboxTensor = dataTensor.shape()[2] == 9;

    for (int64_t instanceId = 0; instanceId < instanceDim; ++instanceId)
    {
        const auto cIdx = instanceId * timeDim + inTimeIdx;
        // If valid sample
        if (maskTensor.data<int>()[cIdx])
        {
            if (isBboxTensor)
            {
                renderBBox(dataTensor.raw_data(), cIdx, heatmapImages, outputDims, roiScale, separateClasses);
            }
            else
            {
                renderCircle(dataTensor.raw_data(), cIdx, heatmapImages[0], outputDims, roiScale, circleRad);
            }
        }
    }
}

/**
 * @brief Run over multiple time indexes
 */
void createHeatMapImageMulti(ConstDaliTensor stateTensor, ConstDaliTensor maskTensor, ConstDaliTensor timeIdxs,
    DaliTensor outputTensor, float roiScale, float circleRad, bool separateClasses, bool filterFuture) noexcept
{
    const std::vector<int> futureMask = filterFuture ? maskInvalidFuture(maskTensor) : std::vector<int>();

    auto mask_ = filterFuture ? ConstDaliTensor(futureMask.data(), maskTensor.shape()) : maskTensor;

    for (int64_t outputIdx = 0; outputIdx < timeIdxs.shape()[0]; ++outputIdx)
    {
        createHeatmapImage(stateTensor, mask_, timeIdxs.data<int>()[outputIdx], outputTensor, outputIdx, roiScale,
            circleRad, separateClasses);
    }
}

template <>
void OccupancyMaskGenerator<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& stateTensor = ws.Input<::dali::CPUBackend>(0);
    const auto& maskTensor = ws.Input<::dali::CPUBackend>(1);
    const auto& timeTensor = ws.Input<::dali::CPUBackend>(2);

    auto& outputTensor = ws.Output<::dali::CPUBackend>(0);
    auto& tPool = ws.GetThreadPool();

    for (int sampleId = 0; sampleId < ws.GetRequestedBatchSize(0); ++sampleId)
    {
        tPool.AddWork(
            [&, sampleId](int)
            {
                createHeatMapImageMulti(stateTensor[sampleId], maskTensor[sampleId], timeTensor[sampleId],
                    outputTensor[sampleId], mROIScale, mCircleRadPx, mSeparateClasses, mFilterFuture);
            });
    }
    tPool.RunAll();
}

/**
 * @brief Create a Heatmap Image at a time interval
 *
 * @param dataTensor
 * @param maskTensor
 * @param outputTensor Class{x,y}, Timestep, Height, Width
 * @param timeIdx
 */
void createFlowImage(ConstDaliTensor dataTensor, ConstDaliTensor maskTensor, std::size_t inTimeIdx,
    DaliTensor outputTensor, std::size_t outTimeIdx, float roiScale, bool separateClasses) noexcept
{
    const auto outputDims = outputTensor.shape();
    const auto outputStride = outputDims[2] * outputDims[3] * daliType2size(outputTensor.type());
    const auto timeStride = outputStride * outputDims[1];

    // Create vector of cv::Mat views of class image tensors
    std::vector<cv::Mat> flowImages;
    flowImages.reserve(outputDims[0]);
    for (auto idx = 0; idx < outputDims[0]; ++idx)
    {
        flowImages.emplace_back(outputDims[2], outputDims[3], CV_32F,
            outputTensor.raw_mutable_data() + idx * timeStride + outputStride * outTimeIdx);
        flowImages.back().setTo(0);
    }

    const auto instanceDim = dataTensor.shape()[0];
    const auto timeDim = dataTensor.shape()[1];

    for (int64_t instanceId = 0; instanceId < instanceDim; ++instanceId)
    {
        const auto cIdx = instanceId * timeDim + inTimeIdx;
        // If valid sample
        if (maskTensor.data<int>()[cIdx])
        {
            const auto data = static_cast<const stateData*>(dataTensor.raw_data())[cIdx];

            const int classIdx = static_cast<int>(data.cls) - 1;
            if (classIdx < 0 || classIdx > 2)
                continue; // skip class that isn't vehicle, pedestrian, cyclist

            // Scaling factor from normalized coordinates to pixel coordinates
            const float xScale = outputDims[3] / 2.f;
            const float yScale = outputDims[2] / 2.f;

            // Calulate bbox in image coordinates
            // transform absolute position from -1,1 to 0,1 before pixel scaling factor
            const float x = (data.x / roiScale + 1.f) * xScale;
            const float y = (data.y / roiScale + 1.f) * yScale;
            const float width = data.w / roiScale * xScale;
            const float length = data.l / roiScale * yScale;

            // Find polyPoints
            auto polyPoints = poseToPoly(x, y, data.yaw * M_PI, width, length);

            // Apply to image
            cv::fillConvexPoly(flowImages[separateClasses ? 2 * classIdx : 0], polyPoints, cv::Scalar(data.vx));
            cv::fillConvexPoly(flowImages[separateClasses ? 2 * classIdx + 1 : 1], polyPoints, cv::Scalar(data.vy));
        }
    }
}

/**
 * @brief Run over multiple time indexes
 */
void createFlowImageMulti(ConstDaliTensor stateTensor, ConstDaliTensor maskTensor, ConstDaliTensor timeIdxs,
    DaliTensor outputTensor, bool filterFuture, float roiScale, bool separateClasses) noexcept
{
    const std::vector<int> futureMask = filterFuture ? maskInvalidFuture(maskTensor) : std::vector<int>();

    auto mask_ = filterFuture ? ConstDaliTensor(futureMask.data(), maskTensor.shape()) : maskTensor;

    for (int64_t outputIdx = 0; outputIdx < timeIdxs.shape()[0]; ++outputIdx)
    {
        createFlowImage(
            stateTensor, mask_, timeIdxs.data<int>()[outputIdx], outputTensor, outputIdx, roiScale, separateClasses);
    }
}

template <>
void FlowMaskGenerator<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& stateTensor = ws.Input<::dali::CPUBackend>(0);
    const auto& maskTensor = ws.Input<::dali::CPUBackend>(1);
    const auto& timeTensor = ws.Input<::dali::CPUBackend>(2);

    auto& outputTensor = ws.Output<::dali::CPUBackend>(0);
    auto& tPool = ws.GetThreadPool();

    for (int sampleId = 0; sampleId < ws.GetRequestedBatchSize(0); ++sampleId)
    {
        tPool.AddWork(
            [&, sampleId](int)
            {
                createFlowImageMulti(stateTensor[sampleId], maskTensor[sampleId], timeTensor[sampleId],
                    outputTensor[sampleId], mFilterFuture, mROIScale, mSeparateClasses);
            });
    }
    tPool.RunAll();
}

} // namespace occupancyop

DALI_REGISTER_OPERATOR(OccupancyMask, ::occupancyop::OccupancyMaskGenerator<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(OccupancyMask)
    .DocStr(
        "Generates occupancy mask from x,y,t,l,w at different time points where the inputs x,y,t are already "
        "normalised between [-1,1] from masked normalise step and l,w are normalized with the same scaling factor, "
        "also returns the time index of the sample")
    .NumInput(3)
    .NumOutput(1)
    .AddArg("size", "Size of the output image, the image is always square", ::dali::DALIDataType::DALI_INT64)
    .AddArg("roi", "Scale to fraction of ROI observeable, centered", ::dali::DALIDataType::DALI_FLOAT)
    .AddArg("circle_radius", "Size of the occupancy radius if running on pedestrian dataset",
        ::dali::DALIDataType::DALI_FLOAT)
    .AddArg("separate_classes", "Separate classes occupancy into different channels", ::dali::DALIDataType::DALI_BOOL)
    .AddArg("filter_future", "Filter ids that don't appear in past or current frame", ::dali::DALIDataType::DALI_BOOL);

DALI_REGISTER_OPERATOR(FlowMask, ::occupancyop::FlowMaskGenerator<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(FlowMask)
    .DocStr(
        "Generates occupancy mask from x,y,t,l,w at different time points where the inputs x,y,t are already "
        "normalised between [-1,1] from masked normalise step and l,w are normalized with the same scaling factor, "
        "also returns the time index of the sample")
    .NumInput(3)
    .NumOutput(1)
    .AddArg("size", "Size of the output image, the image is always square", ::dali::DALIDataType::DALI_INT64)
    .AddArg("roi", "Scale to fraction of ROI observeable, centered", ::dali::DALIDataType::DALI_FLOAT)
    .AddArg("separate_classes", "Separate classes occupancy into different channels", ::dali::DALIDataType::DALI_BOOL)
    .AddArg("filter_future", "Filter ids that don't appear in past or current frame", ::dali::DALIDataType::DALI_BOOL);
