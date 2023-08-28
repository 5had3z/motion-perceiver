#include "plugin.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_set>

namespace randomop
{
using DaliTensor = dali::SampleView<dali::CPUBackend>;

std::vector<int32_t> generateTimeIdxs(const std::set<int64_t>& constTime, int64_t minTime, int64_t maxTime,
    int64_t randCount, int64_t stride, unsigned int seed = 0)
{
    // Create vector of candidates that aren't in the constant set according to range(min,max,stride)
    // TODO: use std::views::to() when available instead of emplace_back
    std::vector<int64_t> randomIdxs;
    for (auto&& v : std::views::iota(minTime, maxTime + 1) | std::views::stride(stride)
            | std::views::filter([&](auto i) { return constTime.count(i) == 0; }))
    {
        randomIdxs.emplace_back(v);
    }

    // Statically initialise generator with seed
    static auto gen = std::mt19937(seed);

    // Randomly Shuffle
    std::ranges::shuffle(randomIdxs, gen);

    // Create new set with constant and n randomly sampled time idxs
    // We use a set so its ordered which may simplify things downstream
    // Although this does contain some strange syntax....
    std::set<int64_t> timeIdxs = constTime;
    timeIdxs.insert(randomIdxs.begin(), std::next(randomIdxs.begin(), randCount));

    if (timeIdxs.size() != (randCount + constTime.size()))
    {
        throw std::runtime_error("Incorrect size somehow");
    }

    // Return as std::vector for contiguity
    return {timeIdxs.begin(), timeIdxs.end()};
}

template <>
void MixedRandomGenerator<dali::CPUBackend>::RunImpl(dali::Workspace& ws)
{
    auto& outputTensor = ws.Output<::dali::CPUBackend>(0);
    auto outputTensorType = outputTensor.type_info();
    auto timeIdx = generateTimeIdxs(mConstValues, mMin, mMax, mCount, mStride, mSeed);

    for (int sampleId = 0; sampleId < ws.GetRequestedBatchSize(0); ++sampleId)
    {
        outputTensorType.Copy<dali::CPUBackend, dali::CPUBackend>(
            outputTensor.raw_mutable_tensor(sampleId), timeIdx.data(), timeIdx.size(), 0);
    }
};

} // namespace randomop

DALI_REGISTER_OPERATOR(MixedRandomGenerator, randomop::MixedRandomGenerator<dali::CPUBackend>, dali::CPU);

// Bug in DALI won't accept zero as default value
DALI_SCHEMA(MixedRandomGenerator)
    .DocStr("Randomly samples from distibution and mixes with always sampled values")
    .NumInput(0)
    .NumOutput(1)
    .AddArg("always_sample", "Values to always sample", dali::DALIDataType::DALI_INT_VEC)
    .AddOptionalArg<int64_t>("min", "Minimum value of random sample", -1)
    .AddOptionalArg<int64_t>("max", "Maximum random value to sample", -1)
    .AddOptionalArg<int64_t>("stride", "Stride to randomly sample from", 1)
    .AddOptionalArg<int64_t>("n_random", "Number of random values to sample", -1);
