#include "cudadb/search.hpp"
#include "cudadb/algorithms.hpp"
#include "cudadb/enumeration.hpp"
#include "cudadb/input.hpp"
#include <fstream>
#include <iostream>

namespace cudadb {
std::atomic<bool> g_stopRequested{false};
namespace {
bool terminal(const std::string& s) {
    return s == "found" || s == "not_found_in_requested_range" || s == "case_recovery_no_match";
}
void emitResult(const std::string& resultsPath, const Checkpoint& cp, std::size_t i, const std::string& mode) {
    std::string rec = resultRecordJson(cp.targets[i], cp.algorithm, mode, cp.states[i],
                                       cp.states[i].status == "found" ? "cpu-verified" : "none");
    if (resultsPath.empty()) { std::cout << rec << "\n"; std::cout.flush(); }
    else {
        std::ofstream out(resultsPath, std::ios::app);
        if (!out) throw InputError("cannot open results file: " + resultsPath);
        out << rec << "\n";
    }
}
void save(const std::string& path, const Checkpoint& cp) { if (!path.empty()) saveCheckpoint(path, cp); }

void doCaseRecovery(Checkpoint& cp, std::size_t i, const Bytes& upperPw, const std::string& mode,
                    const std::string& ckpt, const std::string& results) {
    auto& st = cp.states[i];
    auto rec = recoverCase(upperPw, cp.targets[i], cp.caseBudget, st.caseNextAssignment);
    if (rec.outcome == CaseRecoveryResult::Outcome::Found) {
        st.status = "found";
        st.passwordHex = hexEncode(rec.password);
        st.pendingUpperHex.clear();
        if (!verifyCandidate(cp.algorithm, cp.targets[i], rec.password))
            throw CorrectnessError("case-recovery result failed CPU verification");
    } else if (rec.outcome == CaseRecoveryResult::Outcome::NoMatch) {
        st.status = "case_recovery_no_match";
        st.pendingUpperHex.clear();
    } else {
        st.status = "uppercase_match_pending_case";
        st.pendingUpperHex = hexEncode(upperPw);
        st.caseNextAssignment = rec.nextAssignment;
    }
    save(ckpt, cp);
    if (terminal(st.status)) emitResult(results, cp, i, mode);
}
} // namespace

int runSearch(Checkpoint& cp, const std::string& checkpointPath, const std::string& resultsPath,
              int device, const std::string& tune, const LaunchConfig& baseCfg) {
    for (std::size_t i = 0; i < cp.targets.size(); ++i) {
        auto& st = cp.states[i];
        const auto& tgt = cp.targets[i];
        const Bytes& alphabet = cp.alphabets[i];
        const std::uint32_t B = std::uint32_t(alphabet.size());
        const bool upperMode = cp.algorithm == Algorithm::Mssql && tgt.kind == TargetKind::DualUppercase &&
                               cp.mssqlMode != MssqlMode::Direct;
        const std::string mode = cp.algorithm == Algorithm::Oracle11g ? "direct" : (upperMode ? "uppercase" : "direct");
        if (terminal(st.status)) continue;
        if (st.status == "uppercase_match_pending_case") {
            doCaseRecovery(cp, i, hexDecode(st.pendingUpperHex, "pending candidate"), mode, checkpointPath, resultsPath);
            continue;
        }
        Digest wanted = upperMode ? *tgt.upperDigest : tgt.digest;
        std::uint32_t lastLength = cp.fixedLength ? cp.minLength : cp.maxLength;
        for (std::uint32_t L = st.length; L <= lastLength; ++L) {
            st.length = L;
            BigUint space = spaceSize(B, L);
            BigUint rank = BigUint::parseDecimal(st.rank, "rank");
            BigUint rangeEnd = space;
            if (cp.fixedLength && !cp.rangeCount.empty()) {
                BigUint cnt = BigUint::parseDecimal(cp.rangeCount, "count");
                rangeEnd = BigUint::parseDecimal(cp.startRank, "start rank");
                if (cnt.fitsU64()) rangeEnd.addU64(cnt.toU64()); else rangeEnd = space;
                if (rangeEnd.compare(space) > 0) rangeEnd = space;
            }
            // Launch config per (length, B), tuned or fallback.
            LaunchConfig cfg = resolveLaunchConfig(tune, cp.algorithm, L, B, device,
                                                   baseCfg.blockSize, baseCfg.interleave,
                                                   baseCfg.batchCandidates, nullptr);
            if (baseCfg.gridMultiplierSet) cfg.gridMultiplier = baseCfg.gridMultiplier;
            std::cerr << "launch config: target=" << tgt.label << " length=" << L << " B=" << B
                      << " block=" << cfg.blockSize << " interleave=" << cfg.interleave
                      << " grid-multiplier=" << cfg.gridMultiplier
                      << " batch=" << cfg.batchCandidates << " source=" << cfg.source << "\n";
            while (rank.compare(rangeEnd) < 0) {
                if (g_stopRequested.load()) {
                    st.rank = rank.toDecimal();
                    st.status = "searching";
                    save(checkpointPath, cp);
                    std::cerr << "interrupted: checkpoint " << (checkpointPath.empty() ? std::string("(none)") : checkpointPath) << "\n";
                    return 130;
                }
                BigUint remaining = rangeEnd.sub(rank);
                std::uint64_t batch = cfg.batchCandidates;
                if (remaining.fitsU64() && remaining.toU64() < batch) batch = remaining.toU64();
                if (batch == 0) break;
                auto baseDigits = rankToDigits(rank.toDecimal(), B, L);
                auto hit = gpuSearch(cp.algorithm, tgt, wanted, alphabet, baseDigits, batch, device, cfg);
                if (hit) {
                    auto digits = addOffsetToDigits(baseDigits, B, *hit);
                    Bytes pw = candidateFromDigits(digits, alphabet);
                    BigUint foundRank = rank; foundRank.addU64(*hit);
                    st.rank = foundRank.toDecimal();
                    if (upperMode) {
                        if (!verifyUppercaseForm(cp.algorithm, tgt, pw))
                            throw CorrectnessError("GPU hit failed CPU verification (uppercase digest)");
                        st.pendingUpperHex = hexEncode(pw);
                        st.caseNextAssignment = "0";
                        save(checkpointPath, cp);
                        doCaseRecovery(cp, i, pw, mode, checkpointPath, resultsPath);
                    } else {
                        if (!verifyCandidate(cp.algorithm, tgt, pw))
                            throw CorrectnessError("GPU hit failed CPU verification");
                        st.status = "found";
                        st.passwordHex = hexEncode(pw);
                        save(checkpointPath, cp);
                        emitResult(resultsPath, cp, i, mode);
                    }
                    break; // next target
                }
                rank.addU64(batch);
                st.rank = rank.toDecimal();
                save(checkpointPath, cp);
                std::cerr << "target " << tgt.label << " length " << L << " rank " << st.rank << "/" << rangeEnd.toDecimal() << "\n";
            }
            if (terminal(st.status) || st.status == "uppercase_match_pending_case") break;
            st.rank = "0";
        }
        if (!terminal(st.status) && st.status != "uppercase_match_pending_case") {
            st.status = "not_found_in_requested_range";
            save(checkpointPath, cp);
            emitResult(resultsPath, cp, i, mode);
        }
    }
    return 0;
}

} // namespace cudadb
