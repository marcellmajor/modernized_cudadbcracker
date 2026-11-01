#include "cudadb/dispatch.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cudadb {
namespace {
std::string cachePath() {
    if (const char* e = std::getenv("CUDADB_TUNE_CACHE")) return e;
    const char* home = std::getenv("HOME");
    if (home) return std::string(home) + "/.cache/cudadbcracker/tuning.txt";
    return "tuning_cache.txt";
}
std::string cacheKey(const DeviceInfo& d, Algorithm alg, std::uint32_t L, std::uint32_t B) {
    std::ostringstream ss;
    ss << d.name << "|cc" << d.computeMajor << "." << d.computeMinor << "|SM" << d.multiProcessorCount
       << "|" << toString(alg) << "|L" << L << "|B" << B;
    return ss.str();
}
bool cacheLookup(const std::string& key, LaunchConfig& cfg) {
    std::ifstream in(cachePath());
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos || line.substr(0, tab) != key) continue;
        std::istringstream ss(line.substr(tab + 1));
        if (ss >> cfg.blockSize >> cfg.interleave >> cfg.gridMultiplier) { cfg.source = "tune-cache"; return true; }
    }
    return false;
}
void cacheStore(const std::string& key, const LaunchConfig& cfg) {
    std::string path = cachePath();
    auto slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        std::string dir = path.substr(0, slash);
        std::string cmd = "mkdir -p '" + dir + "'";
        if (std::system(cmd.c_str()) != 0) return;
    }
    std::ofstream out(path, std::ios::app);
    if (out) out << key << "\t" << cfg.blockSize << " " << cfg.interleave << " " << cfg.gridMultiplier << "\n";
}
} // namespace

LaunchConfig resolveLaunchConfig(const std::string& tune, Algorithm alg, std::uint32_t length,
                                 std::uint32_t alphabetSize, int device,
                                 std::optional<int> userBlock, std::optional<int> userInterleave,
                                 std::optional<std::uint64_t> userBatch, double* tuneSecondsOut) {
    LaunchConfig cfg;
    if (userBlock) cfg.blockSize = *userBlock;
    if (userInterleave) cfg.interleave = *userInterleave;
    if (userBatch) cfg.batchCandidates = *userBatch;
    if (tuneSecondsOut) *tuneSecondsOut = 0;
    if (tune == "off" || !cudaAvailable()) return cfg;
    if (tune != "quick" && tune != "full") throw InputError("--tune must be off, quick or full");
    DeviceInfo dev = requireDevice(device);
    std::string key = cacheKey(dev, alg, length, alphabetSize);
    LaunchConfig cached = cfg;
    if (cacheLookup(key, cached)) {
        cached.batchCandidates = cfg.batchCandidates;
        return cached;
    }
    auto t0 = std::chrono::steady_clock::now();
    Bytes alphabet(alphabetSize);
    for (std::uint32_t i = 0; i < alphabetSize; ++i) alphabet[i] = std::uint8_t(i & 0xff);
    std::vector<int> blocks = tune == "full" ? std::vector<int>{64, 128, 256, 512} : std::vector<int>{128, 256};
    std::vector<int> interleaves = tune == "full" ? std::vector<int>{1, 2, 4} : std::vector<int>{1, 2};
    std::vector<int> grids = tune == "full" ? std::vector<int>{8, 32, 64} : std::vector<int>{cfg.gridMultiplier};
    LaunchConfig best = cfg; double bestRate = -1;
    for (int b : blocks) for (int il : interleaves) for (int gm : grids) {
        LaunchConfig trial = cfg; trial.blockSize = b; trial.interleave = il; trial.gridMultiplier = gm;
        try {
            auto r = runBenchmark(alg, length, alphabet, 262144, 3, device, trial);
            if (r.hashesPerSecond > bestRate) { bestRate = r.hashesPerSecond; best = trial; }
        } catch (const CudaError&) { /* variant unavailable on this device */ }
    }
    best.source = "tuned";
    cacheStore(key, best);
    auto t1 = std::chrono::steady_clock::now();
    if (tuneSecondsOut) *tuneSecondsOut = std::chrono::duration<double>(t1 - t0).count();
    return best;
}

} // namespace cudadb
