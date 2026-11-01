#pragma once
#include "message_pack.cuh"

namespace cudadb_cuda {

// Generic correctness kernel (spec 9.1): one candidate per grid-stride step,
// no shared memory, no barriers.
__global__ void genericSearchKernel(__grid_constant__ const KernelParams p, std::uint64_t* result) {
    const bool single = encodedLength(p) <= 55;
    std::uint64_t stride = std::uint64_t(blockDim.x) * gridDim.x;
    for (std::uint64_t off = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         off < p.batchCount; off += stride) {
        std::uint8_t digits[128];
        digitsPlusOffset(p, off, digits);
        std::uint8_t pw[128];
        for (int i = 0; i < p.length; ++i) pw[i] = p.alphabet[digits[i]];
        // Lengths that fit one block (Oracle <=45, MSSQL <=25) skip the
        // multi-block staging buffer even on the generic path.
        ShaState st = single ? digestSingleBlock(p, pw) : digestGeneric(p, pw);
        if (stateEquals(st, p.digest)) atomicMin((unsigned long long*)result, (unsigned long long)off);
    }
}

__global__ void digestKernel(__grid_constant__ const KernelParams p, const std::uint8_t* password, std::uint32_t* outWords) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    std::uint8_t pw[128];
    for (int i = 0; i < p.length; ++i) pw[i] = password[i];
    // Exercise the same single-block path the search uses, so verify/self-test
    // cover digestSingleBlock (Oracle <=45, MSSQL <=25), not only digestGeneric.
    ShaState st = encodedLength(p) <= 55 ? digestSingleBlock(p, pw) : digestGeneric(p, pw);
    for (int i = 0; i < 5; ++i) outWords[i] = st.h[i];
}

} // namespace cudadb_cuda
