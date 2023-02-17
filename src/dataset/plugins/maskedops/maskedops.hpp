#pragma once

#include <dali/pipeline/operator/operator.h>

namespace maskedop
{

/**
 * @brief Clamps a dimension between -1, 1, finds the min/max value to center the data and then normalizes by a given
 * value. This algorithm uses an additonal tensor to ignore padding values in the data tensor.
 */
template <typename Backend>
class Normalize : public ::dali::Operator<Backend>
{
private:
    float mNormalizeScalar = 0;

public:
    inline explicit Normalize(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
        , mNormalizeScalar(spec.GetArgument<float>("normalize_value"))
    {
    }

    virtual inline ~Normalize() = default;

    // Remove copy/move constructor/assignment
    Normalize(const Normalize&) = delete;
    Normalize& operator=(const Normalize&) = delete;
    Normalize(Normalize&&) = delete;
    Normalize& operator=(Normalize&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc, const ::dali::Workspace& ws) override
    {
        const auto& input = ws.Input<Backend>(0);
        const auto& mask = ws.Input<Backend>(1);
        output_desc.resize(1);
        output_desc[0] = {input.shape(), input.type()};

        // Only Input Types Supported due to assumptions for static casting
        bool valid = true;
        valid &= input.type() == ::dali::DALIDataType::DALI_FLOAT;
        valid &= mask.type() == ::dali::DALIDataType::DALI_INT32;
        return valid;
    }

    void RunImpl(::dali::Workspace& ws) override;
};

/**
 * @brief Clamps a dimension between -1, 1, finds the min/max value to center the data and then normalizes by a given
 * value. This algorithm uses an additonal tensor to ignore padding values in the data tensor.
 */
template <typename Backend>
class Median : public ::dali::Operator<Backend>
{
public:
    inline explicit Median(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
    {
    }

    virtual inline ~Median() = default;

    // Remove copy/move constructor/assignment
    Median(const Median&) = delete;
    Median& operator=(const Median&) = delete;
    Median(Median&&) = delete;
    Median& operator=(Median&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc, const ::dali::Workspace& ws) override
    {
        const auto& input = ws.Input<Backend>(0);
        const auto& mask = ws.Input<Backend>(1);
        const auto n_samples = input.num_samples();

        // Single Scalar Output
        output_desc.resize(1);
        output_desc[0].type = dali::DALIDataType::DALI_FLOAT;
        output_desc[0].shape.resize(n_samples, 1);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[0].shape.set_tensor_shape(i, dali::TensorShape<1>(1));
        }

        // Only Input Types Supported due to assumptions for static casting
        bool valid = true;
        valid &= input.type() == ::dali::DALIDataType::DALI_FLOAT;
        valid &= mask.type() == ::dali::DALIDataType::DALI_INT32;
        return valid;
    }

    void RunImpl(::dali::Workspace& ws) override;
};

} // namespace maskedop
