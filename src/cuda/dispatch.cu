#include "cudadb/dispatch.hpp"
#include "cudadb/enumeration.hpp"
#include "kernel_launch.hpp"
#include <algorithm>
#include <cmath>

namespace cudadb {
namespace {
cudadb_cuda::KernelParams makeParams(Algorithm alg, const Target& t, const Digest& wanted,
                                     const Bytes& alphabet, const std::vector<std::uint8_t>& baseDigits,
                                     std::uint64_t count) {
    cudadb_cuda::KernelParams p{};
    std::size_t L = baseDigits.size();
    for (std::size_t i = 0; i < L && i < 128; ++i) p.baseDigits[i] = baseDigits[i];
    for (std::size_t i = 0; i < alphabet.size() && i < 256; ++i) p.alphabet[i] = alphabet[i];
    for (std::size_t i = 0; i < t.salt.size() && i < 10; ++i) p.salt[i] = t.salt[i];
    auto w = digestWords(wanted);
    for (int i = 0; i < 5; ++i) p.digest[i] = w[i];
    p.length = (std::uint16_t)L;
    p.alphabetSize = (std::uint16_t)alphabet.size();
    p.batchCount = count;
    p.algorithm = alg == Algorithm::Oracle11g ? cudadb_cuda::kOra : cudadb_cuda::kMs;
    return p;
}
int gridFor(std::uint64_t workItems, int blockSize, const DeviceInfo& dev, const LaunchConfig& cfg) {
    std::uint64_t need = (workItems + std::uint64_t(blockSize) - 1) / std::uint64_t(blockSize);
    std::uint64_t cap = std::uint64_t(std::max(1, dev.multiProcessorCount)) * std::uint64_t(std::max(1, cfg.gridMultiplier));
    std::uint64_t g = std::min(need, cap);
    return (int)std::max<std::uint64_t>(g, 1);
}
std::optional<std::uint64_t> wrap(std::uint64_t v) {
    if (v == UINT64_MAX) return std::nullopt;
    return v;
}
} // namespace

std::optional<std::uint64_t> gpuSearch(Algorithm alg, const Target& target, const Digest& wanted,
                                       const Bytes& alphabet, const std::vector<std::uint8_t>& baseDigits,
                                       std::uint64_t count, int device, const LaunchConfig& cfg) {
    DeviceInfo dev = requireDevice(device);
    const std::uint32_t L = (std::uint32_t)baseDigits.size();
    const std::uint64_t B = alphabet.size();
    bool fastOk = L >= 1 && L <= 16 && B > 1 && (cfg.interleave == 1 || cfg.interleave == 2 || cfg.interleave == 4);
    try {
        if (!fastOk) {
            auto p = makeParams(alg, target, wanted, alphabet, baseDigits, count);
            return wrap(cudadb_cuda::runGeneric(p, cfg.blockSize, gridFor(count, cfg.blockSize, dev, cfg)));
        }
        // Compose leading partial group + aligned fast groups + trailing
        // partial group so no candidate is duplicated or skipped (spec 9.2).
        std::uint64_t done = 0;
        std::vector<std::uint8_t> digits = baseDigits;
        if (digits[0] != 0) {
            std::uint64_t lead = std::min<std::uint64_t>(count, B - digits[0]);
            auto p = makeParams(alg, target, wanted, alphabet, digits, lead);
            auto hit = cudadb_cuda::runGeneric(p, cfg.blockSize, gridFor(lead, cfg.blockSize, dev, cfg));
            if (hit != UINT64_MAX) return hit;
            digits = addOffsetToDigits(digits, (std::uint32_t)B, lead);
            done += lead;
        }
        std::uint64_t remaining = count - done;
        std::uint64_t fastCount = remaining - (remaining % B);
        if (fastCount > 0) {
            auto p = makeParams(alg, target, wanted, alphabet, digits, fastCount);
            std::uint64_t groups = (fastCount + B - 1) / B;
            // One thread owns `interleave` groups, so size the grid from the
            // thread-work count (pairs/quads), not the raw group count.
            std::uint64_t il = (std::uint64_t)cfg.interleave;
            std::uint64_t workItems = (groups + il - 1) / il;
            auto hit = cudadb_cuda::runFast(p, cfg.blockSize, gridFor(workItems, cfg.blockSize, dev, cfg), cfg.interleave);
            if (hit != UINT64_MAX) return done + hit;
            digits = addOffsetToDigits(digits, (std::uint32_t)B, fastCount);
            done += fastCount;
            remaining -= fastCount;
        }
        if (remaining > 0) {
            auto p = makeParams(alg, target, wanted, alphabet, digits, remaining);
            auto hit = cudadb_cuda::runGeneric(p, cfg.blockSize, gridFor(remaining, cfg.blockSize, dev, cfg));
            if (hit != UINT64_MAX) return done + hit;
        }
        return std::nullopt;
    } catch (const std::runtime_error& e) {
        throw CudaError(e.what());
    }
}

Digest gpuDigest(Algorithm alg, const Bytes& password, const Bytes& salt, int device) {
    requireDevice(device);
    Target t; t.salt = salt;
    Digest wanted{};
    std::vector<std::uint8_t> digits(password.size(), 0);
    Bytes alphabet(1, 0);
    auto p = makeParams(alg, t, wanted, alphabet, digits, 1);
    p.length = (std::uint16_t)password.size();
    std::uint32_t words[5] = {};
    try { cudadb_cuda::runDigest(p, password.data(), words); }
    catch (const std::runtime_error& e) { throw CudaError(e.what()); }
    return digestFromWords({words[0], words[1], words[2], words[3], words[4]});
}

BenchmarkResult runBenchmark(Algorithm alg, std::uint32_t length, const Bytes& alphabet,
                             std::uint64_t count, int repeats, int device, const LaunchConfig& cfg) {
    DeviceInfo dev = requireDevice(device);
    Target t;
    t.salt.assign(alg == Algorithm::Oracle11g ? 10 : 4, 0);
    Digest wanted{};
    std::vector<std::uint8_t> digits(length, 0);
    auto p = makeParams(alg, t, wanted, alphabet, digits, count);
    int grid = gridFor(count, cfg.blockSize, dev, cfg);
    int regs = -1; std::size_t shared = 0; bool fast = false;
    double secs = 0;
    try { secs = cudadb_cuda::timeSearchLaunches(p, cfg.blockSize, grid, cfg.interleave, repeats, &regs, &shared, &fast); }
    catch (const std::runtime_error& e) { throw CudaError(e.what()); }
    BenchmarkResult r;
    r.deviceSeconds = secs;
    r.candidates = count;
    r.hashesPerSecond = secs > 0 ? double(count) / secs : 0;
    r.blockSize = cfg.blockSize; r.interleave = cfg.interleave;
    r.registersPerThread = regs; r.sharedMemBytes = shared;
    r.kernelFamily = fast ? "fast" : "generic";
    return r;
}

} // namespace cudadb
