#include "src.hpp"

#include <ranges>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

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
        mMapInfo[scene.first.as<std::string>()] = scene.second.as<double>();
    }
}

template <>
void LoadScene<dali::CPUBackend>::RunImpl(dali::Workspace& ws)
{
    const auto& idTensor = ws.Input<dali::CPUBackend>(0);
    const auto& tfTensor = ws.Input<dali::CPUBackend>(1);

    auto& imageTensor = ws.Output<dali::CPUBackend>(0);

    auto& tPool = ws.GetThreadPool();

    namespace rv = std::ranges::views;

    for (int sampleId = 0; sampleId < idTensor.num_samples(); ++sampleId)
    {
        const std::string rawId = static_cast<const char*>(idTensor[sampleId].raw_data());
        // SceneId format should be SCENE_IDX_.., we want to just get SCENE_IDX
        auto sceneId = rawId | rv::split('_') | rv::take(2) | rv::join_with('_') | rv::common;
        std::string sceneIdS(sceneId.begin(), sceneId.end());

        tPool.AddWork([&, sampleId](int) {});
    }
    tPool.RunAll();
}

} // namespace sceneloader

DALI_REGISTER_OPERATOR(LoadScene, sceneloader::LoadScene<dali::CPUBackend>, dali::CPU);

DALI_SCHEMA(LoadScene)
    .DocStr("Reads context image and applies appropriate transforms to match with scene")
    .NumInput(2)
    .NumOutput(1)
    .AddArg("src", "Folder which contains images of corresponding dataset", dali::DALI_STRING)
    .AddArg("metadata", "Metadata file which contains mapping from scene id to pixel to meteres scaling factor",
        dali::DALI_STRING)
    .AddArg("size", "size of the final image", dali::DALI_INT64)
    .AddArg("channels", "number of channels expected for output image", dali::DALI_INT64);
