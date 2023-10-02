#include "src.hpp"

#include <fstream>
#include <ranges>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;
namespace rv = std::ranges::views;

namespace sceneloader
{

template <>
void LoadScene<dali::CPUBackend>::loadMetadata()
{
    DALI_ENFORCE(fs::exists(mMetadataFile), "Metadata file not found");

    auto metadata = YAML::LoadFile(mMetadataFile.string());
    assert(metadata.IsMap() && "Metadata should be map of scene_id to pixel scaling factor");
    for (auto&& scene : metadata)
    {
        mMapInfo[scene.first.as<std::string>()] = scene.second.as<float>();
    }
}

template <>
std::size_t LoadScene<dali::CPUBackend>::findLargestFilesize() const noexcept
{
    auto getFileSize = [&](std::string_view sceneId)
    {
        fs::path filePath = mImageFolder / sceneId;
        filePath.replace_extension(".jpg");
        return fs::file_size(filePath);
    };
    auto fileSizes = mMapInfo | rv::keys | rv::transform(getFileSize) | rv::common;
    auto maxFileSize = *std::ranges::max_element(fileSizes);

    return maxFileSize;
}

template <>
void LoadScene<dali::CPUBackend>::RunImpl(dali::Workspace& ws)
{
    const auto& idTensor = ws.Input<dali::CPUBackend>(0);

    auto& imageTensor = ws.Output<dali::CPUBackend>(0);
    auto& scaleTensor = ws.Output<dali::CPUBackend>(1);

    for (int sampleId = 0; sampleId < idTensor.num_samples(); ++sampleId)
    {
        const std::string rawId = static_cast<const char*>(idTensor[sampleId].raw_data());
        // SceneId format should be SCENE_IDX_.., we want to just get SCENE_IDX
        auto sceneIdView = rawId | rv::split('_') | rv::take(2) | rv::join_with('_') | rv::common;
        std::string sceneId(sceneIdView.begin(), sceneIdView.end());
        *scaleTensor[sampleId].mutable_data<float>() = 1 / mMapInfo.at(sceneId); // pixels per meter
        auto filePath = mImageFolder / sceneId;
        filePath.replace_extension(".jpg");

        std::ifstream imageFile(filePath, std::ios::binary | std::ios::ate);
        std::streamsize fileSize = imageFile.tellg(); // Get filesize
        imageFile.seekg(0, std::ios::beg);            // Go back to beginning
        imageFile.read(reinterpret_cast<char*>(imageTensor.raw_mutable_tensor(sampleId)), fileSize);

        if (imageFile.fail())
        {
            throw std::runtime_error("Failure reading " + filePath.string());
        }
    }
}

} // namespace sceneloader

DALI_REGISTER_OPERATOR(LoadScene, sceneloader::LoadScene<dali::CPUBackend>, dali::CPU);

DALI_SCHEMA(LoadScene)
    .DocStr("Returns context image and pixel per meter of the image")
    .NumInput(1)
    .NumOutput(2)
    .AddArg("src", "Folder which contains images of corresponding dataset", dali::DALI_STRING)
    .AddArg("metadata", "Metadata file which contains mapping from scene id to pixel to meteres scaling factor",
        dali::DALI_STRING);
