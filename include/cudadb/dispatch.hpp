#pragma once
#include "cudadb/types.hpp"
#include <optional>
#include <string>
#include <vector>

namespace cudadb {

struct DeviceInfo {
    int index = 0;
    std::string name;
    int computeMajor = 0, computeMinor = 0;
    int multiProcessorCount = 0;
    std::size_t totalGlobalMem = 0;
    int maxThreadsPerBlock = 0;
    std::size_t sharedMemPerBlock = 0;
};

bool cudaCompiled();   // true only in CUDA builds
bool cudaAvailable();  // compiled and at least one device (never throws for "no device")
std::vector<DeviceInfo> listDevices(); // empty when unavailable
DeviceInfo requireDevice(int index);   // throws CudaError when unusable

struct LaunchConfig {
    int blockSize = 128;
    int interleave = 1;
    int gridMultiplier = 32; // blocks per SM target for grid sizing
    std::uint64_t batchCandidates = 1048576;
    std::string source = "fallback"; // fallback|user|tune-cache|tuned
};

// GPU search over offsets [0,count) from baseDigits for one target digest.
// Returns the minimum matching offset, or nullopt. Throws CudaError if no
// usable device (no silent CPU fallback, spec section 12 for verify/search).
std::optional<std::uint64_t> gpuSearch(Algorithm alg, const Target& target, const Digest& wanted,
                                       const Bytes& alphabet, const std::vector<std::uint8_t>& baseDigits,
                                       std::uint64_t count, int device, const LaunchConfig& cfg);

// Explicit-candidate GPU digest evaluation (spec sections 7, 12).
Digest gpuDigest(Algorithm alg, const Bytes& password, const Bytes& salt, int device);

// Deterministic fallback + optional tuning (spec section 11).
// tune: "off"|"quick"|"full". Tuning runs synthetic bounded work only.
LaunchConfig resolveLaunchConfig(const std::string& tune, Algorithm alg, std::uint32_t length,
                                 std::uint32_t alphabetSize, int device,
                                 std::optional<int> userBlock, std::optional<int> userInterleave,
                                 std::optional<std::uint64_t> userBatch, double* tuneSecondsOut);

struct BenchmarkResult {
    double hashesPerSecond = 0;
    double deviceSeconds = 0;
    std::uint64_t candidates = 0;
    int blockSize = 0, interleave = 0;
    int registersPerThread = -1;
    std::size_t sharedMemBytes = 0;
    std::string kernelFamily; // generic|fast
};
BenchmarkResult runBenchmark(Algorithm alg, std::uint32_t length, const Bytes& alphabet,
                             std::uint64_t count, int repeats, int device, const LaunchConfig& cfg);

int runCli(int argc, char** argv);

} // namespace cudadb
