#include "cudadb/dispatch.hpp"
// Host-only stub used when CUDADB_ENABLE_CUDA=OFF (spec section 2).
// GPU operations fail explicitly; there is no silent CPU fallback.
namespace cudadb {
bool cudaCompiled() { return false; }
bool cudaAvailable() { return false; }
std::vector<DeviceInfo> listDevices() { return {}; }
DeviceInfo requireDevice(int) { throw CudaError("CUDA support is not compiled into this build (CUDADB_ENABLE_CUDA=OFF)"); }
std::optional<std::uint64_t> gpuSearch(Algorithm, const Target&, const Digest&, const Bytes&,
                                       const std::vector<std::uint8_t>&, std::uint64_t, int, const LaunchConfig&) {
    throw CudaError("CUDA support is not compiled into this build (CUDADB_ENABLE_CUDA=OFF)");
}
Digest gpuDigest(Algorithm, const Bytes&, const Bytes&, int) {
    throw CudaError("CUDA support is not compiled into this build (CUDADB_ENABLE_CUDA=OFF)");
}
BenchmarkResult runBenchmark(Algorithm, std::uint32_t, const Bytes&, std::uint64_t, int, int, const LaunchConfig&) {
    throw CudaError("CUDA support is not compiled into this build (CUDADB_ENABLE_CUDA=OFF)");
}
} // namespace cudadb
