#pragma once

#include <vector>

#include <dali/pipeline/operator/operator.h>

namespace roadgraphop
{

/**
 * @brief Input is tuple of waymo tensors xyz, type, id, valid
 *        Output is tuple features, valid
 */
template <typename Backend>
class RoadGraphTokens : public ::dali::Operator<Backend>
{
private:
    bool mLaneCenter = false;
    bool mLaneMarkings = false;
    std::size_t mMaxFeatures = 0;
    std::size_t mNumSamples = 0;
    float mNormalizeFactor = 0;

public:
    inline explicit RoadGraphTokens(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
        , mLaneCenter(spec.GetArgument<bool>("lane_center"))
        , mLaneMarkings(spec.GetArgument<bool>("lane_markings"))
        , mMaxFeatures(spec.GetArgument<int64_t>("max_features"))
        , mNumSamples(spec.GetArgument<int64_t>("n_samples"))
        , mNormalizeFactor(spec.GetArgument<float>("normalize_value"))
    {
    }

    virtual inline ~RoadGraphTokens() = default;

    // Remove copy/move constructor/assignment
    RoadGraphTokens(const RoadGraphTokens&) = delete;
    RoadGraphTokens& operator=(const RoadGraphTokens&) = delete;
    RoadGraphTokens(RoadGraphTokens&&) = delete;
    RoadGraphTokens& operator=(RoadGraphTokens&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc, const ::dali::Workspace& ws) override
    {
        const auto& input = ws.Input<Backend>(0);
        const auto n_samples = input.num_samples();

        // Two outputs: data and mask
        output_desc.resize(2);

        // data output
        const dali::TensorShape<2> dataShape(mMaxFeatures, mNumSamples);
        output_desc[0].type = dali::DALIDataType::DALI_FLOAT;
        output_desc[0].shape.resize(n_samples, dataShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[0].shape.set_tensor_shape(i, dataShape);
        }

        // mask output
        const dali::TensorShape<1> maskShape(mMaxFeatures);
        output_desc[1].type = dali::DALIDataType::DALI_INT32;
        output_desc[1].shape.resize(n_samples, maskShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[1].shape.set_tensor_shape(i, maskShape);
        }

        bool valid = true;
        valid &= input.type() == ::dali::DALIDataType::DALI_FLOAT;
        valid &= ws.template Input<Backend>(1).type() == ::dali::DALIDataType::DALI_INT64;
        return valid;
    }

    void RunImpl(::dali::Workspace& ws) override;
};

/**
 * @brief Input is tuple of waymo tensors xyz, type, id, valid
 *        Output is roadgraph image
 */
template <typename Backend>
class RoadGraphImage : public ::dali::Operator<Backend>
{
private:
    bool mWaymoEvalFrame = false;
    int32_t mMarkingFlags = 0;
    float mNormalizeFactor = 0;
    std::size_t mImageSize = 0;

public:
    enum MarkingTypes
    {
        LANECENTER = 0,
        ROADLINE = 1,
        ROADEDGE = 2,
        OTHER = 3,
    };

    inline explicit RoadGraphImage(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
        , mWaymoEvalFrame(spec.GetArgument<bool>("waymo_eval_frame"))
        , mImageSize(spec.GetArgument<int64_t>("size"))
        , mNormalizeFactor(spec.GetArgument<float>("normalize_value"))
    {
        mMarkingFlags |= spec.GetArgument<bool>("lane_center") << MarkingTypes::LANECENTER;
        mMarkingFlags |= spec.GetArgument<bool>("lane_markings") << MarkingTypes::ROADLINE;
    }

    virtual inline ~RoadGraphImage() = default;

    // Remove copy/move constructor/assignment
    RoadGraphImage(const RoadGraphImage&) = delete;
    RoadGraphImage& operator=(const RoadGraphImage&) = delete;
    RoadGraphImage(RoadGraphImage&&) = delete;
    RoadGraphImage& operator=(RoadGraphImage&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc, const ::dali::Workspace& ws) override
    {
        const auto& input = ws.Input<Backend>(0);
        const auto n_samples = input.num_samples();

        DALI_ENFORCE(mMarkingFlags != 0, "No roadgraph features have been selected to overlay");

        // Two outputs: data and mask
        output_desc.resize(1);

        // image output
        const dali::TensorShape<3> imageShape(1, mImageSize, mImageSize);
        output_desc[0].type = dali::DALIDataType::DALI_FLOAT;
        output_desc[0].shape.resize(n_samples, imageShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[0].shape.set_tensor_shape(i, imageShape);
        }

        bool valid = true;
        valid &= input.type() == ::dali::DALIDataType::DALI_FLOAT;
        valid &= ws.template Input<Backend>(1).type() == ::dali::DALIDataType::DALI_INT64;
        return valid;
    }

    void RunImpl(::dali::Workspace& ws) override;
};

} // namespace roadgraphop
