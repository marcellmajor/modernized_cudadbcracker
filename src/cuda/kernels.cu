#include "kernel_launch.hpp"
#include "generic_kernel.cuh"
#include "fast_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cudadb_cuda {
namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
// One reusable stream for all launches in this process (spec section 10).
cudaStream_t stream() {
    static cudaStream_t s = nullptr;
    if (!s) check(cudaStreamCreate(&s), "cudaStreamCreate");
    return s;
}
struct ResultBuf {
    std::uint64_t* d = nullptr;
    ResultBuf() { check(cudaMalloc(&d, sizeof(std::uint64_t)), "cudaMalloc result"); }
    ~ResultBuf() { cudaFree(d); }
    void reset() { std::uint64_t v = UINT64_MAX; check(cudaMemcpy(d, &v, sizeof(v), cudaMemcpyHostToDevice), "init result"); }
    std::uint64_t get() { std::uint64_t v; check(cudaMemcpy(&v, d, sizeof(v), cudaMemcpyDeviceToHost), "read result"); return v; }
};

template<int ALGO, int L>
std::uint64_t launchFastFor(const KernelParams& p, int blockSize, int gridBlocks, int interleave, ResultBuf& rb) {
    if (interleave == 4) fastSearchKernel<ALGO, L, 4><<<gridBlocks, blockSize, 0, stream()>>>(p, rb.d);
    else if (interleave == 2) fastSearchKernel<ALGO, L, 2><<<gridBlocks, blockSize, 0, stream()>>>(p, rb.d);
    else fastSearchKernel<ALGO, L, 1><<<gridBlocks, blockSize, 0, stream()>>>(p, rb.d);
    check(cudaGetLastError(), "fast kernel launch");
    check(cudaStreamSynchronize(stream()), "fast kernel sync");
    return rb.get();
}
template<int L>
std::uint64_t launchFastL(const KernelParams& p, int blockSize, int gridBlocks, int interleave, ResultBuf& rb) {
    if (p.algorithm == kMs) return launchFastFor<kMs, L>(p, blockSize, gridBlocks, interleave, rb);
    return launchFastFor<kOra, L>(p, blockSize, gridBlocks, interleave, rb);
}
std::uint64_t launchFastSwitch(const KernelParams& p, int blockSize, int gridBlocks, int interleave, ResultBuf& rb) {
    switch (p.length) {
        case 1: return launchFastL<1>(p, blockSize, gridBlocks, interleave, rb);
        case 2: return launchFastL<2>(p, blockSize, gridBlocks, interleave, rb);
        case 3: return launchFastL<3>(p, blockSize, gridBlocks, interleave, rb);
        case 4: return launchFastL<4>(p, blockSize, gridBlocks, interleave, rb);
        case 5: return launchFastL<5>(p, blockSize, gridBlocks, interleave, rb);
        case 6: return launchFastL<6>(p, blockSize, gridBlocks, interleave, rb);
        case 7: return launchFastL<7>(p, blockSize, gridBlocks, interleave, rb);
        case 8: return launchFastL<8>(p, blockSize, gridBlocks, interleave, rb);
        case 9: return launchFastL<9>(p, blockSize, gridBlocks, interleave, rb);
        case 10: return launchFastL<10>(p, blockSize, gridBlocks, interleave, rb);
        case 11: return launchFastL<11>(p, blockSize, gridBlocks, interleave, rb);
        case 12: return launchFastL<12>(p, blockSize, gridBlocks, interleave, rb);
        case 13: return launchFastL<13>(p, blockSize, gridBlocks, interleave, rb);
        case 14: return launchFastL<14>(p, blockSize, gridBlocks, interleave, rb);
        case 15: return launchFastL<15>(p, blockSize, gridBlocks, interleave, rb);
        case 16: return launchFastL<16>(p, blockSize, gridBlocks, interleave, rb);
        default: throw std::runtime_error("fast kernel: unsupported length");
    }
}
bool fastEligible(const KernelParams& p, int interleave) {
    return p.length >= 1 && p.length <= 16 && p.alphabetSize > 1 &&
           (interleave == 1 || interleave == 2 || interleave == 4);
}
} // namespace

std::uint64_t runGeneric(const KernelParams& p, int blockSize, int gridBlocks) {
    ResultBuf rb; rb.reset();
    genericSearchKernel<<<gridBlocks, blockSize, 0, stream()>>>(p, rb.d);
    check(cudaGetLastError(), "generic kernel launch");
    check(cudaStreamSynchronize(stream()), "generic kernel sync");
    return rb.get();
}
std::uint64_t runFast(const KernelParams& p, int blockSize, int gridBlocks, int interleave) {
    ResultBuf rb; rb.reset();
    return launchFastSwitch(p, blockSize, gridBlocks, interleave, rb);
}
void runDigest(const KernelParams& p, const std::uint8_t* password, std::uint32_t outWords[5]) {
    std::uint8_t* dPw = nullptr; std::uint32_t* dOut = nullptr;
    check(cudaMalloc(&dPw, 128), "cudaMalloc password");
    check(cudaMalloc(&dOut, sizeof(std::uint32_t) * 5), "cudaMalloc digest");
    check(cudaMemcpy(dPw, password, p.length, cudaMemcpyHostToDevice), "copy password");
    digestKernel<<<1, 1, 0, stream()>>>(p, dPw, dOut);
    check(cudaGetLastError(), "digest kernel launch");
    check(cudaStreamSynchronize(stream()), "digest kernel sync");
    check(cudaMemcpy(outWords, dOut, sizeof(std::uint32_t) * 5, cudaMemcpyDeviceToHost), "read digest");
    cudaFree(dPw); cudaFree(dOut);
}
double timeSearchLaunches(const KernelParams& p, int blockSize, int gridBlocks, int interleave,
                          int repeats, int* numRegs, std::size_t* sharedBytes, bool* usedFast) {
    bool fast = fastEligible(p, interleave);
    if (usedFast) *usedFast = fast;
    if (numRegs) *numRegs = -1;
    if (sharedBytes) *sharedBytes = 0;
    ResultBuf rb; rb.reset();
    auto launchOnce = [&]() {
        if (fast) {
            // Attributes for one representative variant are reported by the
            // caller only as diagnostics; timing covers the real launch.
            switch (p.length) {
                case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8:
                case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16:
                    launchFastSwitch(p, blockSize, gridBlocks, interleave, rb); return;
                default: break;
            }
        }
        genericSearchKernel<<<gridBlocks, blockSize, 0, stream()>>>(p, rb.d);
        check(cudaGetLastError(), "benchmark launch");
        check(cudaStreamSynchronize(stream()), "benchmark sync");
    };
    // Warm-up.
    launchOnce();
    std::vector<double> secs;
    for (int i = 0; i < repeats; ++i) {
        rb.reset();
        cudaEvent_t a, b;
        check(cudaEventCreate(&a), "event"); check(cudaEventCreate(&b), "event");
        check(cudaEventRecord(a, stream()), "event record");
        if (fast) { launchFastSwitch(p, blockSize, gridBlocks, interleave, rb); }
        else {
            genericSearchKernel<<<gridBlocks, blockSize, 0, stream()>>>(p, rb.d);
            check(cudaGetLastError(), "benchmark launch");
            check(cudaStreamSynchronize(stream()), "benchmark sync");
        }
        check(cudaEventRecord(b, stream()), "event record");
        check(cudaEventSynchronize(b), "event sync");
        float ms = 0; check(cudaEventElapsedTime(&ms, a, b), "event elapsed");
        cudaEventDestroy(a); cudaEventDestroy(b);
        secs.push_back(ms / 1000.0);
    }
    std::sort(secs.begin(), secs.end());
    return secs.empty() ? 0 : secs[secs.size() / 2];
}
} // namespace cudadb_cuda
