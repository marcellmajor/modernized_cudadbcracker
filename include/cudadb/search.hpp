#pragma once
#include "cudadb/checkpoint.hpp"
#include "cudadb/dispatch.hpp"
#include <atomic>
#include <string>

namespace cudadb {
extern std::atomic<bool> g_stopRequested;
// Runs the target-major search described in DECISIONS.md. Returns a process
// exit code (0 completed, 130 interrupted). Throws InputError/CudaError/
// CorrectnessError for the caller's exit-code mapping.
int runSearch(Checkpoint& cp, const std::string& checkpointPath, const std::string& resultsPath,
              int device, const std::string& tune, const LaunchConfig& baseCfg);
} // namespace cudadb
