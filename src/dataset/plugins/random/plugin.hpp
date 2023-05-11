#pragma once

#include <random>
#include <vector>

#include <dali/pipeline/operator/operator.h>

namespace randomop
{

template <typename Backend>
class MixedRandomGenerator : public dali::Operator<Backend>
{
private:
    int64_t mCount{0};
    int64_t mMin{0};
    int64_t mMax{0};
    std::set<int64_t> mConstValues{};

public:
    inline explicit MixedRandomGenerator(const dali::OpSpec& spec)
        : dali::Operator<Backend>(spec)
        , mCount{std::max(spec.GetArgument<int64_t>("n_random"), 0L)} // Bug in DALI won't accept zero default
        , mMin{spec.GetArgument<int64_t>("min")}
        , mMax{spec.GetArgument<int64_t>("max")}
    {
        for (auto&& elem : spec.GetRepeatedArgument<int64_t>("always_sample"))
        {
            mConstValues.emplace(elem);
        }
    }

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<dali::OutputDesc>& output_desc, const dali::Workspace& ws) override
    {
        DALI_ENFORCE((mMax - mMin) >= mCount, "Number of random idxs to yield is greater than the min-max range");

        // Only one output
        output_desc.resize(1);

        const auto n_samples = ws.GetRequestedBatchSize(0);

        // Random Sample Output
        dali::TensorShape<1> timeShape(mConstValues.size() + mCount);
        output_desc[0].type = dali::DALIDataType::DALI_INT32;
        output_desc[0].shape.resize(n_samples, timeShape.static_ndim);
        for (int i = 0; i < n_samples; ++i)
        {
            output_desc[0].shape.set_tensor_shape(i, timeShape);
        }

        // Not handling validaton atm
        return true;
    }

    void RunImpl(dali::Workspace& ws) override;
};

} // namespace randomop