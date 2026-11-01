#pragma once
#include "cudadb/types.hpp"
#include <string>
#include <vector>

namespace cudadb {

inline constexpr int kCheckpointSchema = 1;
inline constexpr int kResultSchema = 1;

struct TargetState {
    std::string status = "searching"; // searching|found|not_found_in_requested_range|
                                      // uppercase_match_pending_case|case_recovery_no_match|interrupted
    std::uint32_t length = 1;         // current length cursor
    std::string rank = "0";           // decimal, next untested rank within length
    std::string passwordHex;          // verified password, when found
    std::string pendingUpperHex;      // uppercase candidate awaiting case recovery
    std::string caseNextAssignment = "0"; // decimal
};

struct Checkpoint {
    Algorithm algorithm = Algorithm::Oracle11g;
    MssqlMode mssqlMode = MssqlMode::Auto;
    std::vector<Target> targets;                 // complete records: resume needs no input file
    std::vector<Bytes> alphabets;                // effective alphabet per target
    std::vector<TargetState> states;             // per target
    std::uint32_t minLength = 1, maxLength = 8;
    bool fixedLength = false;
    std::string startRank = "0";                 // fixed-length ranges only
    std::string rangeCount;                      // empty = whole space
    std::uint64_t caseBudget = kDefaultCaseBudget;
    // Informational only (spec section 13): tuning/build metadata.
    std::string buildId = "cudadbcracker-modernized-1";
    std::string tuneNote;
};

// Atomic save: write temp file in the same directory, flush, rename.
void saveCheckpoint(const std::string& path, const Checkpoint& cp);
// Throws InputError on schema/fingerprint/range inconsistency.
Checkpoint loadCheckpoint(const std::string& path);

std::string checkpointToJson(const Checkpoint& cp);
Checkpoint checkpointFromJson(const std::string& text);

// One JSONL result record (spec section 6.3). Field names are defined here
// and documented in README.md (the spec lists required content, not names).
std::string resultRecordJson(const Target& t, Algorithm alg, const std::string& mode,
                             const TargetState& st, const std::string& verification);

} // namespace cudadb
