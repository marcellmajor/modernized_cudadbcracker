#include "cudadb/dispatch.hpp"
#include <cuda_runtime.h>

namespace cudadb {
bool cudaCompiled() { return true; }
bool cudaAvailable() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }
std::vector<DeviceInfo> listDevices() {
    std::vector<DeviceInfo> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return out;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        DeviceInfo d;
        d.index = i; d.name = p.name; d.computeMajor = p.major; d.computeMinor = p.minor;
        d.multiProcessorCount = p.multiProcessorCount; d.totalGlobalMem = p.totalGlobalMem;
        d.maxThreadsPerBlock = p.maxThreadsPerBlock; d.sharedMemPerBlock = p.sharedMemPerBlock;
        out.push_back(d);
    }
    return out;
}
DeviceInfo requireDevice(int index) {
    auto devs = listDevices();
    if (devs.empty()) throw CudaError("no CUDA device is available");
    if (index < 0 || index >= (int)devs.size()) throw CudaError("CUDA device index out of range");
    if (cudaSetDevice(index) != cudaSuccess) throw CudaError("cudaSetDevice failed");
    return devs[(std::size_t)index];
}
} // namespace cudadb
