#pragma once

#include <dali/pipeline/operator/operator.h>
#include <filesystem>
#include <unordered_map>

namespace sceneloader
{

template <typename Backend>
class LoadScene : public dali::Operator<Backend>
{
private:
    int64_t mImageChannels;
    int64_t mImageSize;
    std::filesystem::path mMetadataFile;
    std::filesystem::path mImageFolder;
    std::unordered_map<std::string, double> mMapInfo;

    void loadMetadata();

public:
    inline explicit LoadScene(const dali::OpSpec& spec)
        : dali::Operator<Backend>(spec)
        , mImageChannels{spec.GetArgument<int64_t>("channels")}
        , mImageSize{spec.GetArgument<int64_t>("size")}
        , mMetadataFile{spec.GetArgument<std::string>("metadata")}
        , mImageFolder{spec.GetArgument<std::string>("src")}
    {
    }

    LoadScene(const LoadScene&) = delete;
    LoadScene& operator=(const LoadScene&) = delete;
    LoadScene(LoadScene&&) = delete;
    LoadScene& operator=(LoadScene&&) = delete;

protected:
    bool CanInferOutputs() const override
    {
        return true;
    }

    bool SetupImpl(std::vector<dali::OutputDesc>& output_desc, const dali::Workspace& ws) override
    {
        DALI_ENFORCE(ws.GetInputBatchSize(0) == ws.GetInputBatchSize(1) && "Input Batch size does not match");
        DALI_ENFORCE(mImageChannels > 0 && mImageSize > 0 && "Image size and channels should be > 0");
        this->loadMetadata();

        output_desc.resize(1); // Only one output
        output_desc.back().type = dali::DALI_UINT8;
        output_desc.back().shape = dali::TensorListShape<>::make_uniform(
            ws.GetInputBatchSize(0), dali::TensorShape<3>{mImageChannels, mImageSize, mImageSize});

        return true;
    }

    void RunImpl(dali::Workspace& ws) override;
};
} // namespace sceneloader