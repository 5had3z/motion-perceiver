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
    std::filesystem::path mMetadataFile;
    std::filesystem::path mImageFolder;
    std::unordered_map<std::string, float> mMapInfo;

    void loadMetadata();

    [[nodiscard]] auto findLargestFilesize() const noexcept -> std::size_t;

public:
    inline explicit LoadScene(const dali::OpSpec& spec)
        : dali::Operator<Backend>(spec)
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

        auto maxFileSize = this->findLargestFilesize();

        output_desc.resize(2);
        output_desc[0].type = dali::DALI_UINT8;
        output_desc[0].shape
            = dali::TensorListShape<>::make_uniform(ws.GetInputBatchSize(0), dali::TensorShape<1>(maxFileSize));
        output_desc[1].type = dali::DALI_FLOAT;
        output_desc[1].shape = dali::TensorListShape<>::make_uniform(ws.GetInputBatchSize(0), dali::TensorShape<1>(1));

        return true;
    }

    void RunImpl(dali::Workspace& ws) override;
};
} // namespace sceneloader