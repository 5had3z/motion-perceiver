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

public:
    inline explicit OccupancyMaskGenerator(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
        , mSeparateClasses{spec.GetArgument<bool>("separate_classes")}
        , mFilterFuture{spec.GetArgument<bool>("filter_future")}
        , mROIScale{spec.GetArgument<float>("roi")}
        , mMaskSize{spec.GetArgument<int64_t>("size")}
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
        DALI_ENFORCE(mROIScale > 0.0 && mROIScale <= 1.0, "invalid roi, 0 < roi <= 1");

        const auto nTimestamp = ws.GetInputShape(2)[0][0];
        // Class, Timestep, Height, Width
        dali::TensorShape<4> maskShape(mSeparateClasses ? 3 : 1, nTimestamp, mMaskSize, mMaskSize);

        // Two outputs, heatmap and time index
        output_desc.resize(1);

        const auto n_samples = ws.GetRequestedBatchSize(0);

        // Heatmap output
        output_desc[0].type = dali::DALIDataType::DALI_FLOAT;
        output_desc[0].shape.resize(n_samples, maskShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[0].shape.set_tensor_shape(i, maskShape);
        }

        // Not handling validation yet
        return true;
    }

    void RunImpl(::dali::Workspace& ws) override;
};

template <typename Backend>
class FlowMaskGenerator : public ::dali::Operator<Backend>
{
};

} // namespace occupancyop
