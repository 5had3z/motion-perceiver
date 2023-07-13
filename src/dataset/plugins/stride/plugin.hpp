#pragma once

#include <dali/pipeline/operator/operator.h>

namespace sliceop
{

template <typename Backend>
class StrideSliceOp : public dali::Operator<Backend>
{
private:
    int64_t mDimIdx;
    int64_t mStride;

public:
    inline explicit StrideSliceOp(const dali::OpSpec& spec)
        : dali::Operator<Backend>(spec)
        , mDimIdx{spec.GetArgument<int64_t>("axis")}
        , mStride{spec.GetArgument<int64_t>("stride")}
    {
    }

    virtual inline ~StrideSliceOp() = default;

    StrideSliceOp(const StrideSliceOp&) = delete;
    StrideSliceOp& operator=(const StrideSliceOp&) = delete;
    StrideSliceOp(StrideSliceOp&&) = delete;
    StrideSliceOp& operator=(StrideSliceOp&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<dali::OutputDesc>& outputDesc, const dali::Workspace& ws) override
    {
        bool isValid{true};

        DALI_ENFORCE(mStride > 1, "Stride must be greater than 1");
        DALI_ENFORCE(ws.NumInput() == 1, "Only input of 1 supported");

        auto shapeList = ws.GetInputShape(0);
        isValid &= shapeList.ndim > mDimIdx;
        DALI_ENFORCE(shapeList.ndim > mDimIdx, "Stride dimension exceeds tensor dimension");

        for (int sIdx = 0; sIdx < shapeList.num_samples(); ++sIdx)
        {
            auto shape = shapeList.tensor_shape(sIdx);
            shape[mDimIdx] /= mStride;
            shapeList.set_tensor_shape(sIdx, shape);
        }
        outputDesc.resize(1, {shapeList, ws.GetInputDataType(0)});

        return isValid;
    }

    void RunImpl(dali::Workspace& ws) override;
};

} // namespace sliceop