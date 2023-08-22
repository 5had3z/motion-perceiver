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
    uint64_t mImageChannels;
    uint64_t mImageSize;
    std::filesystem::path mMetadataFile;
    std::filesystem::path mImageFolder;
    std::unordered_map<std::string, double> mMapInfo;

    void loadMetadata();

public:
    inline explicit LoadScene(const dali::OpSpec& spec)
        : dali::Operator<Backend>(spec)
        , mImageChannels{spec.GetArgument<uint64_t>("channels")}
        , mImageSize{spec.GetArgument<uint64_t>("size")}
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
        this->loadMetadata();

        output_desc.resize(1); // Only one output
        output_desc.back().type = dali::DALI_UINT8;
        const dali::TensorShape<3> imageShape(mImageChannels, mImageSize, mImageSize);
        output_desc.back().shape.make_uniform(ws.GetInputBatchSize(0), imageShape);

        return true;
    }

    void RunImpl(dali::Workspace& ws) override;
};
} // namespace sceneloader