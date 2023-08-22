#include "src.hpp"

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
