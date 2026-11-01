#pragma once
#include "message_pack.cuh"

namespace cudadb_cuda {

// Short-message grouped kernel (spec 9.2): each thread owns one group of B
// consecutive ranks that share all digits except digit 0. Prepared words are
// private scalars (via a statically indexed local array the compiler can
// promote); the working schedule lives inside shaCompress.
// Requires: single-block message, baseDigits digit 0 == 0, batchCount covers
// whole groups (host composes partial edges with the generic kernel).
template<int ALGO, int L>
__device__ __forceinline__ void prepareGroup(const KernelParams& p, std::uint64_t group, std::uint32_t prep[16]) {
    // Buffers sized by the compile-time length L (<= 16 here) so the
    // compiler can keep them in registers instead of local memory.
    // digitsPlusOffset only writes the first L digits.
    std::uint8_t digits[16];
    digitsPlusOffset(p, group * p.alphabetSize, digits);
    std::uint8_t pw[L > 0 ? L : 1];
    for (int i = 0; i < L; ++i) pw[i] = p.alphabet[digits[i]];
    // No KernelParams copy: algorithm/length are compile-time here.
    packSingleBlockEx(p, ALGO, L, pw, prep);
}

template<int ALGO, int L, int INTERLEAVE>
__global__ void fastSearchKernel(__grid_constant__ const KernelParams p, std::uint64_t* result) {
    const std::uint64_t B = p.alphabetSize;
    const std::uint64_t groups = (p.batchCount + B - 1) / B;
    if (INTERLEAVE == 1) {
        std::uint64_t stride = std::uint64_t(blockDim.x) * gridDim.x;
        for (std::uint64_t g = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; g < groups; g += stride) {
            std::uint32_t prep[16];
            prepareGroup<ALGO, L>(p, g, prep);
            for (std::uint64_t v = 0; v < B; ++v) {
                std::uint64_t rank = g * B + v;
                if (rank >= p.batchCount) break;
                std::uint32_t w[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) w[i] = prep[i];
                w[0] = (prep[0] & 0x00FFFFFFu) | (std::uint32_t(p.alphabet[v]) << 24);
                ShaState st = shaInit();
                shaCompress(st, w);
                if (stateEquals(st, p.digest)) atomicMin((unsigned long long*)result, (unsigned long long)rank);
            }
        }
    } else if (INTERLEAVE == 4) {
        // Four groups, round-interleaved (spec 9.3 optional experiment).
        std::uint64_t quadStride = std::uint64_t(blockDim.x) * gridDim.x;
        std::uint64_t quads = (groups + 3) / 4;
        for (std::uint64_t quad = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; quad < quads; quad += quadStride) {
            std::uint64_t g[4];
            std::uint32_t prep[4][16];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                g[q] = quad * 4 + (std::uint64_t)q;
                prepareGroup<ALGO, L>(p, g[q] < groups ? g[q] : quad * 4, prep[q]);
            }
            for (std::uint64_t v = 0; v < B; ++v) {
                std::uint32_t w[4][16];
                std::uint32_t first = std::uint32_t(p.alphabet[v]) << 24;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) w[q][i] = prep[q][i];
                    w[q][0] = (prep[q][0] & 0x00FFFFFFu) | first;
                }
                ShaState s[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) s[q] = shaInit();
                shaCompressQuad(s, w);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    std::uint64_t r = g[q] * B + v;
                    if (g[q] < groups && r < p.batchCount && stateEquals(s[q], p.digest))
                        atomicMin((unsigned long long*)result, (unsigned long long)r);
                }
            }
        }
    } else {
        // Two groups with round-interleaved independent states (spec 9.3).
        std::uint64_t pairStride = std::uint64_t(blockDim.x) * gridDim.x;
        std::uint64_t pairs = (groups + 1) / 2;
        for (std::uint64_t pair = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; pair < pairs; pair += pairStride) {
            std::uint64_t g0 = pair * 2, g1 = g0 + 1;
            std::uint32_t prep0[16], prep1[16];
            prepareGroup<ALGO, L>(p, g0, prep0);
            prepareGroup<ALGO, L>(p, g1 < groups ? g1 : g0, prep1);
            for (std::uint64_t v = 0; v < B; ++v) {
                std::uint32_t w0[16], w1[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) { w0[i] = prep0[i]; w1[i] = prep1[i]; }
                std::uint32_t first = std::uint32_t(p.alphabet[v]) << 24;
                w0[0] = (prep0[0] & 0x00FFFFFFu) | first;
                w1[0] = (prep1[0] & 0x00FFFFFFu) | first;
                ShaState s0 = shaInit(), s1 = shaInit();
                shaCompressPair(s0, w0, s1, w1);
                std::uint64_t r0 = g0 * B + v;
                if (r0 < p.batchCount && stateEquals(s0, p.digest))
                    atomicMin((unsigned long long*)result, (unsigned long long)r0);
                std::uint64_t r1 = g1 * B + v;
                if (g1 < groups && r1 < p.batchCount && stateEquals(s1, p.digest))
                    atomicMin((unsigned long long*)result, (unsigned long long)r1);
            }
        }
    }
}

} // namespace cudadb_cuda
