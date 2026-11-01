#pragma once
#include "message_pack.cuh"
#include <cstdint>

namespace cudadb_cuda {
// Host-side launch helpers implemented in kernels.cu. Each returns the
// minimum matching offset, or UINT64_MAX. gridBlocks must already be >= 1.
std::uint64_t runGeneric(const KernelParams& p, int blockSize, int gridBlocks);
std::uint64_t runFast(const KernelParams& p, int blockSize, int gridBlocks, int interleave);
void runDigest(const KernelParams& p, const std::uint8_t* password, std::uint32_t outWords[5]);
// Benchmark timing: launches the search kernel (fast when eligible, else
// generic) `repeats` times over p.batchCount candidates; returns median
// seconds per launch. Also reports kernel attributes when available.
double timeSearchLaunches(const KernelParams& p, int blockSize, int gridBlocks, int interleave,
                          int repeats, int* numRegs, std::size_t* sharedBytes, bool* usedFast);
} // namespace cudadb_cuda
