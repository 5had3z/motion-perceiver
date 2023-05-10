#include "roadgraph.hpp"

#include <type_traits>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

namespace roadgraphop
{

using ConstDaliTensor = ::dali::ConstSampleView<::dali::CPUBackend>;
using DaliTensor = ::dali::SampleView<::dali::CPUBackend>;
using RoadMarkingType = RoadGraphImage<::dali::CPUBackend>::MarkingTypes;

enum class FeatureType
{
    LANECENTER_FREEWAY = 1,
    LANECENTER_STREET = 2,
    LANECENTER_BIKE = 3,
    ROADLINE_BROKENSINGLEWHITE = 6,
    ROADLINE_SOLIDSINGLEWHITE = 7,
    ROADLINE_SOLIDDOUBLEWHITE = 8,
    ROADLINE_BROKENSINGLEYELLOW = 9,
    ROADLINE_BROKENDOUBLEYELLOW = 10,
    ROADLINE_SOLIDSINGLEYELLOW = 11,
    ROADLINE_SOLIDDOUBLEYELLOW = 12,
    ROADLINE_PASSINGDOUBLEYELLOW = 13,
    ROADEDGE_BOUNDARY = 15,
    ROADEDGE_MEDIAN = 16,
    STOPSIGN = 17,
    CROSSWALK = 18,
    SPEEDBUMP = 19
};

enum class FeatureCategory
{
    LANECENTER,
    ROADLINE,
    ROADEDGE,
    OTHER
};

bool isRoadLineType(FeatureType x)
{
    switch (x)
    {
    case FeatureType::ROADLINE_BROKENSINGLEWHITE:
    case FeatureType::ROADLINE_SOLIDSINGLEWHITE:
    case FeatureType::ROADLINE_SOLIDDOUBLEWHITE:
    case FeatureType::ROADLINE_BROKENSINGLEYELLOW:
    case FeatureType::ROADLINE_BROKENDOUBLEYELLOW:
    case FeatureType::ROADLINE_SOLIDSINGLEYELLOW:
    case FeatureType::ROADLINE_SOLIDDOUBLEYELLOW:
    case FeatureType::ROADLINE_PASSINGDOUBLEYELLOW: return true;
    default: return false;
    }
}

bool isLaneCenterType(FeatureType x)
{
    switch (x)
    {
    case FeatureType::LANECENTER_FREEWAY:
    case FeatureType::LANECENTER_STREET:
    case FeatureType::LANECENTER_BIKE: return true;
    default: return false;
    }
}

bool isRoadEdgeType(FeatureType x)
{
    switch (x)
    {
    case FeatureType::ROADEDGE_BOUNDARY:
    case FeatureType::ROADEDGE_MEDIAN: return true;
    default: return false;
    }
}

bool isOtherType(FeatureType x)
{
    switch (x)
    {
    case FeatureType::STOPSIGN:
    case FeatureType::CROSSWALK:
    case FeatureType::SPEEDBUMP: return true;
    default: return false;
    }
}

/**
 * @brief Roadgraph index is a valid lanecenter point
 * @param[in] valid int64_t roadgraph_samples/valid
 * @param[in] type int64_t roadgraph_samples/type
 * @return bool is valid
 */
inline bool isValidLC(int64_t valid, int64_t type)
{
    return valid && isLaneCenterType(static_cast<FeatureType>(type));
}

/**
 * @brief Roadgraph index is a valid lanecenter point
 * @param[in] valid int64_t roadgraph_samples/valid
 * @param[in] type int64_t roadgraph_samples/type
 * @return bool is valid
 */
inline bool isValidRL(int64_t valid, int64_t type)
{
    return valid && isRoadLineType(static_cast<FeatureType>(type));
}

void createRoadGraphImage(ConstDaliTensor xyzTensor, ConstDaliTensor typeTensor, ConstDaliTensor idTensor,
    ConstDaliTensor validTensor, ConstDaliTensor tfTensor, float normalisaiton, int32_t markingFlags,
    DaliTensor imageTensor)
{
    const auto outputDims = imageTensor.shape();
    cv::Mat heatmapImage(outputDims[1], outputDims[2], CV_32F, imageTensor.raw_mutable_data());
    heatmapImage.setTo(0);

    const auto maxIdx = xyzTensor.shape()[0];

    const float* positionPtr = xyzTensor.data<float>();
    const int64_t* typePtr = typeTensor.data<int64_t>();
    const int64_t* idPtr = idTensor.data<int64_t>();
    const int64_t* validPtr = validTensor.data<int64_t>();

    const cv::Matx23f transform(tfTensor.data<float>());

    auto transform_point = [&](cv::Matx31f point)
    {
        // Perform coordinate transform
        cv::Matx21f pointTf = transform * point;

        // Perfrom Normalization to [0, 2]
        pointTf = pointTf / normalisaiton + cv::Matx21f(1.f, 1.f);

        // Scale to image coordinates
        pointTf = pointTf.mul(cv::Matx21f(outputDims[2] / 2.f, outputDims[1] / 2.f));

        // Add 0.5 to counteract floor rounding for float->int in pixel space
        return cv::Point2i{static_cast<int>(pointTf.val[0] + 0.5), static_cast<int>(pointTf.val[1] + 0.5)};
    };

    for (int64_t idx = 1; idx < maxIdx; ++idx)
    {
        const auto featureType = static_cast<FeatureType>(typePtr[idx]);
        bool predicate = validPtr[idx - 1] && validPtr[idx];
        predicate &= idPtr[idx - 1] == idPtr[idx];
        predicate &= ~(markingFlags & RoadMarkingType::LANECENTER) || isLaneCenterType(featureType);
        predicate &= ~(markingFlags & RoadMarkingType::ROADLINE) || isRoadLineType(featureType);

        if (!predicate)
        {
            continue;
        }

        const float* fromXYZ = positionPtr + 3 * (idx - 1);
        const float* toXYZ = positionPtr + 3 * idx;

        const auto start = transform_point({fromXYZ[0], fromXYZ[1], 1});
        const auto end = transform_point({toXYZ[0], toXYZ[1], 1});

        cv::line(heatmapImage, start, end, cv::Scalar(1), 1);
    }
}

template <>
void RoadGraphImage<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& xyzTensor = ws.Input<::dali::CPUBackend>(0);
    const auto& typeTensor = ws.Input<::dali::CPUBackend>(1);
    const auto& idTensor = ws.Input<::dali::CPUBackend>(2);
    const auto& validTensor = ws.Input<::dali::CPUBackend>(3);
    const auto& tfTensor = ws.Input<::dali::CPUBackend>(4);

    auto& outImageTensor = ws.Output<::dali::CPUBackend>(0);

    auto& tPool = ws.GetThreadPool();
    const auto& inShape = xyzTensor.shape();

    for (int sampleId = 0; sampleId < inShape.num_samples(); ++sampleId)
    {
        tPool.AddWork(
            [&, sampleId](int)
            {
                createRoadGraphImage(xyzTensor[sampleId], typeTensor[sampleId], idTensor[sampleId],
                    validTensor[sampleId], tfTensor[sampleId], mNormalizeFactor, mMarkingFlags,
                    outImageTensor[sampleId]);
            });
    }
    tPool.RunAll();
}

} // namespace roadgraphop

DALI_REGISTER_OPERATOR(RoadgraphImage, ::roadgraphop::RoadGraphImage<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(RoadgraphImage)
    .DocStr("Generates tensor that represents road features from waymo format")
    .NumInput(5)
    .NumOutput(1)
    .AddOptionalArg("lane_center", "add lane centers to image", false)
    .AddOptionalArg("lane_markings", "add lane markings to image", false)
    .AddArg("size", "size of the output image", ::dali::DALIDataType::DALI_INT64)
    .AddArg("normalize_value", "Normalisation factor for x,y coords", ::dali::DALIDataType::DALI_FLOAT);
