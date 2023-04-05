#pragma once

#include <random>
#include <vector>

#include <dali/pipeline/operator/operator.h>

namespace occupancyop
{

/**
 * @brief Clamps a dimension between -1, 1, finds the min/max value to center the data and then normalizes by a given
 * value. This algorithm uses an additonal tensor to ignore padding values in the data tensor.
 */
template <typename Backend>
class OccupancyMaskGenerator : public ::dali::Operator<Backend>
{
private:
    bool mSeparateClasses{false};
    bool mFilterFuture{false};
    float mROIScale{1.0};
    int64_t mMaskSize{0};
    int64_t mRandIdxCount{0};
    std::uniform_int_distribution<> mRandIdx;
    std::vector<int64_t> mConstTimeIndex;

public:
    inline explicit OccupancyMaskGenerator(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
        , mSeparateClasses{spec.GetArgument<bool>("separate_classes")}
        , mFilterFuture{spec.GetArgument<bool>("filter_future")}
        , mROIScale{spec.GetArgument<float>("roi")}
        , mMaskSize{spec.GetArgument<int64_t>("size")}
        , mRandIdxCount{spec.GetArgument<int64_t>("n_random_idx")}
        , mRandIdx{spec.GetArgument<int>("min_random_idx"), spec.GetArgument<int>("max_random_idx")}
        , mConstTimeIndex{spec.GetRepeatedArgument<int64_t>("const_time_idx")}
    {
    }

    virtual inline ~OccupancyMaskGenerator() = default;

    // Remove copy/move constructor/assignment
    OccupancyMaskGenerator(const OccupancyMaskGenerator&) = delete;
    OccupancyMaskGenerator& operator=(const OccupancyMaskGenerator&) = delete;
    OccupancyMaskGenerator(OccupancyMaskGenerator&&) = delete;
    OccupancyMaskGenerator& operator=(OccupancyMaskGenerator&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc, const ::dali::Workspace& ws) override
    {
        const auto& input = ws.Input<Backend>(0);
        const auto n_samples = input.num_samples();

        DALI_ENFORCE(static_cast<std::size_t>(mRandIdx.max() - mRandIdx.min()) >= mRandIdxCount,
            "Number of random time idxs to yield is greater than the min-max range");

        DALI_ENFORCE(mROIScale > 0.0 && mROIScale <= 1.0, "invalid roi, 0 < roi <= 1");

        // Class, Timestep, Height, Width
        dali::TensorShape<4> maskShape(
            mSeparateClasses ? 3 : 1, mConstTimeIndex.size() + mRandIdxCount, mMaskSize, mMaskSize);

        // Two outputs, heatmap and time index
        output_desc.resize(2);

        // Heatmap output
        output_desc[0].type = dali::DALIDataType::DALI_FLOAT;
        output_desc[0].shape.resize(n_samples, maskShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[0].shape.set_tensor_shape(i, maskShape);
        }

        // Time index output
        dali::TensorShape<1> timeShape(mConstTimeIndex.size() + mRandIdxCount);
        output_desc[1].type = dali::DALIDataType::DALI_INT64;
        output_desc[1].shape.resize(n_samples, timeShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[1].shape.set_tensor_shape(i, timeShape);
        }

        // Not handling validation yet
        return true;
    }

    void RunImpl(::dali::Workspace& ws) override;
};

} // namespace occupancyop
