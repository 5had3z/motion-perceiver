#include "roadgraph.hpp"

#include <type_traits>
#include <unordered_map>

#include <eigen3/Eigen/Dense>
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

constexpr inline unsigned int factorial(unsigned int f)
{
    auto ret = 1;
    while (f > 0)
    {
        ret *= f;
        --f;
    }
    return ret;
}

inline float nchoosek(int n, int k)
{
    return factorial(n) / (factorial(n - k) * factorial(k));
}

template <std::size_t degree>
std::array<std::pair<float, float>, degree> bezierFit(std::vector<std::pair<float, float>> points)
{
    Eigen::Matrix<float, -1, degree> G(points.size());
    auto t = Eigen::VectorXf::LinSpaced(points.size(), 0, 1);

    for (std::size_t i = 0; i < degree; ++i)
    {
        G.row(i) = nchoosek(degree, i) * (1 - t.array()).array().pow(degree - i).array() * t.pow(i).array();
    }

    Eigen::Matrix<float, -1, degree> D(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        D(i, 0) = points[i].first;
        D(i, 1) = points[i].second;
    }

    auto GtD = G.transpose() * D;
    auto GtG = G.transpose() * G;
    auto B = GtG.ldlt().solve(GtD);

    std::array<std::pair<float, float>, degree> ret;
    for (std::size_t i = 0; i < degree; ++i)
    {
        ret[i].first = B(i, 0);
        ret[i].second = B(i, 1);
    }

    return ret;
}

/**
 * @brief Currently just handles lanecenter
 */
void createRoadGraphFeatures(ConstDaliTensor xyzTensor, ConstDaliTensor typeTensor, ConstDaliTensor idTensor,
    ConstDaliTensor validTensor, float center_x, float center_y, float normalisaiton, DaliTensor dataTensor,
    DaliTensor maskTensor)
{
    std::unordered_map<int64_t, std::vector<std::pair<float, float>>> roadpoints;
    const auto maxIdx = xyzTensor.shape()[0];

    const float* positionPtr = xyzTensor.data<float>();
    const int64_t* typePtr = typeTensor.data<int64_t>();
    const int64_t* idPtr = typeTensor.data<int64_t>();
    const int64_t* validPtr = validTensor.data<int64_t>();

    for (std::size_t idx = 0; idx < maxIdx; ++idx)
    {
        if (!validPtr[idx])
            continue;

        if (!isLaneCenterType(static_cast<FeatureType>(typePtr[idx])))
            continue;

        const int64_t id = idPtr[idx];
        auto it = roadpoints.find(id);
        if (it == roadpoints.end())
        {
            roadpoints[id] = std::vector<std::pair<float, float>>();
            roadpoints[id].reserve(64);
        }
        roadpoints[id].emplace_back(positionPtr[3 * idx + 0], positionPtr[3 * idx + 1]);
    }

    const auto sum_elements = std::accumulate(
        roadpoints.begin(), roadpoints.end(), 0, [](const auto& acc, const auto& e) { return acc + e.second.size(); });

    if (sum_elements > 0)
    {
        std::cout << "number of splines found: " << std::to_string(roadpoints.size()) << " with average num elements "
                  << std::to_string(sum_elements / roadpoints.size()) << std::endl;
    }
    else
    {
        std::cout << "NO SPLINES FOUND IN SAMPLE" << std::endl;
    }
}

template <>
void RoadGraphTokens<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws)
{
    const auto& xyzTensor = ws.Input<::dali::CPUBackend>(0);
    const auto& typeTensor = ws.Input<::dali::CPUBackend>(1);
    const auto& idTensor = ws.Input<::dali::CPUBackend>(2);
    const auto& validTensor = ws.Input<::dali::CPUBackend>(3);
    const auto& centerXTensor = ws.Input<::dali::CPUBackend>(4);
    const auto& centerYTensor = ws.Input<::dali::CPUBackend>(5);

    auto& outDataTensor = ws.Output<::dali::CPUBackend>(0);
    auto& outMaskTensor = ws.Output<::dali::CPUBackend>(1);

    auto& tPool = ws.GetThreadPool();
    const auto& inShape = xyzTensor.shape();

    for (int sampleId = 0; sampleId < inShape.num_samples(); ++sampleId)
    {
        tPool.AddWork(
            [&, sampleId](int thread_id)
            {
                const float center_x = centerXTensor[sampleId].data<float>()[0];
                const float center_y = centerYTensor[sampleId].data<float>()[0];
                createRoadGraphFeatures(xyzTensor[sampleId], typeTensor[sampleId], idTensor[sampleId],
                    validTensor[sampleId], center_x, center_y, mNormalizeFactor, outDataTensor[sampleId],
                    outMaskTensor[sampleId]);
            });
    }
    tPool.RunAll();
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
    ConstDaliTensor validTensor, float center_x, float center_y, float rotation, float normalisaiton,
    int32_t markingFlags, bool waymoEvalFrame, DaliTensor imageTensor)
{
    const auto outputDims = imageTensor.shape();
    cv::Mat heatmapImage(outputDims[1], outputDims[2], CV_32F, imageTensor.raw_mutable_data());
    heatmapImage.setTo(0);

    const auto maxIdx = xyzTensor.shape()[0];

    const float* positionPtr = xyzTensor.data<float>();
    const int64_t* typePtr = typeTensor.data<int64_t>();
    const int64_t* idPtr = idTensor.data<int64_t>();
    const int64_t* validPtr = validTensor.data<int64_t>();

    auto transform_point = [&](float x, float y)
    {
        // Translate the point into the origin frame
        x -= center_x;
        y -= center_y;

        // Rotate the point around origin
        float rot_x = std::cos(rotation) * x - std::sin(rotation) * y;
        float rot_y = std::sin(rotation) * x + std::cos(rotation) * y;

        // normalize to image
        int norm_x = (rot_x / normalisaiton + 1.f) * outputDims[2] / 2.f;

        // transform to waymo frame if required
        // constexpr float frame_offset = 3.f / 8.f; // for 384
        // constexpr float frame_offset = 1.f / 4.f; // for 512
        constexpr float frame_offset = 1.f / 2.f; // for 256
        int norm_y = waymoEvalFrame ? (-rot_y / normalisaiton + 1.f + frame_offset) * outputDims[1] / 2.f
                                    : (rot_y / normalisaiton + 1.f) * outputDims[1] / 2.f;

        return cv::Point2i{norm_x, norm_y};
    };

    for (std::size_t idx = 1; idx < maxIdx; ++idx)
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

        const cv::Point2i start = transform_point(fromXYZ[0], fromXYZ[1]);
        const cv::Point2i end = transform_point(toXYZ[0], toXYZ[1]);

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
    const auto& centerXTensor = ws.Input<::dali::CPUBackend>(4);
    const auto& centerYTensor = ws.Input<::dali::CPUBackend>(5);
    const auto& centerRTensor = ws.Input<::dali::CPUBackend>(6);

    auto& outImageTensor = ws.Output<::dali::CPUBackend>(0);

    auto& tPool = ws.GetThreadPool();
    const auto& inShape = xyzTensor.shape();

    for (int sampleId = 0; sampleId < inShape.num_samples(); ++sampleId)
    {
        tPool.AddWork(
            [&, sampleId](int thread_id)
            {
                const float center_x = centerXTensor[sampleId].data<float>()[0];
                const float center_y = centerYTensor[sampleId].data<float>()[0];
                const float rotation = centerRTensor[sampleId].data<float>()[0];
                createRoadGraphImage(xyzTensor[sampleId], typeTensor[sampleId], idTensor[sampleId],
                    validTensor[sampleId], center_x, center_y, rotation, mNormalizeFactor, mMarkingFlags,
                    mWaymoEvalFrame, outImageTensor[sampleId]);
            });
    }
    tPool.RunAll();
}

} // namespace roadgraphop

DALI_REGISTER_OPERATOR(RoadgraphTokens, ::roadgraphop::RoadGraphTokens<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(RoadgraphTokens)
    .DocStr("Generates tensor that represents road features from waymo format")
    .NumInput(7)
    .NumOutput(2)
    .AddArg("max_features", "Maximum number of roadgraph features", ::dali::DALIDataType::DALI_INT64)
    .AddArg("n_samples", "Number of features that parameterrise the roadgraph", ::dali::DALIDataType::DALI_INT64)
    .AddArg("normalize_value", "Normalisation factor for x,y coords", ::dali::DALIDataType::DALI_FLOAT)
    .AddArg("lane_center", "output lane center features", ::dali::DALIDataType::DALI_BOOL);

DALI_REGISTER_OPERATOR(RoadgraphImage, ::roadgraphop::RoadGraphImage<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(RoadgraphImage)
    .DocStr("Generates tensor that represents road features from waymo format")
    .NumInput(7)
    .NumOutput(1)
    .AddOptionalArg("lane_center", "add lane centers to image", false)
    .AddOptionalArg("lane_markings", "add lane markings to image", false)
    .AddOptionalArg("waymo_eval_frame", "add offsets to frame for waymo eval", false)
    .AddArg("size", "size of the output image", ::dali::DALIDataType::DALI_INT64)
    .AddArg("normalize_value", "Normalisation factor for x,y coords", ::dali::DALIDataType::DALI_FLOAT);
