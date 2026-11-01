#include "cudadb/dispatch.hpp"
#include "cudadb/algorithms.hpp"
#include "cudadb/checkpoint.hpp"
#include "cudadb/enumeration.hpp"
#include "cudadb/input.hpp"
#include "cudadb/search.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>

namespace cudadb {
namespace {
struct Opts {
    std::map<std::string, std::string> kv;
    std::map<std::string, bool> flags;
    bool has(const std::string& k) const { return kv.count(k) || flags.count(k); }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k); return it == kv.end() ? def : it->second;
    }
    std::optional<std::string> opt(const std::string& k) const {
        auto it = kv.find(k); if (it == kv.end()) return std::nullopt; return it->second;
    }
};
Opts parseOpts(int argc, char** argv, int start) {
    Opts o;
    for (int i = start; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) throw InputError("unexpected argument '" + a + "'");
        std::string key = a.substr(2);
        auto eq = key.find('=');
        if (eq != std::string::npos) { o.kv[key.substr(0, eq)] = key.substr(eq + 1); continue; }
        if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) o.kv[key] = argv[++i];
        else o.flags[key] = true;
    }
    return o;
}
std::uint64_t parseU64(const std::string& s, const char* what, std::uint64_t maxV = ~0ULL) {
    if (s.empty()) throw InputError(std::string(what) + ": empty number");
    for (char c : s) if (c < '0' || c > '9') throw InputError(std::string(what) + ": invalid number '" + s + "'");
    std::uint64_t v;
    try { v = std::stoull(s); } catch (...) { throw InputError(std::string(what) + ": number out of range '" + s + "'"); }
    if (v > maxV) throw InputError(std::string(what) + ": value too large '" + s + "'");
    return v;
}
std::uint32_t parseLen(const std::string& s, const char* what) {
    auto v = parseU64(s, what, kMaxCandidateLen);
    return std::uint32_t(v);
}
Algorithm parseAlg(const Opts& o) {
    auto a = o.get("algorithm");
    if (a == "ora11g") return Algorithm::Oracle11g;
    if (a == "mssql") return Algorithm::Mssql;
    throw InputError("--algorithm must be ora11g or mssql");
}
MssqlMode parseMode(const Opts& o) {
    auto m = o.get("mssql-mode", "auto");
    if (m == "auto") return MssqlMode::Auto;
    if (m == "uppercase") return MssqlMode::Uppercase;
    if (m == "direct") return MssqlMode::Direct;
    throw InputError("--mssql-mode must be auto, uppercase or direct");
}
int parseDevice(const Opts& o) { return (int)parseU64(o.get("device", "0"), "--device", 1024); }

void usage() {
    std::cout <<
      "cudadbcracker devices [--json]\n"
      "cudadbcracker verify --algorithm ora11g|mssql --input FILE --password-hex HEX [--backend cpu|gpu|both] [--device N]\n"
      "cudadbcracker self-test [--device N] [--extended] [--json]\n"
      "cudadbcracker search --algorithm ora11g|mssql --input FILE [--charset-classes CLASSES | --charset-hex HEX]\n"
      "    [--min-length N] [--max-length N] [--length N --start-rank DECIMAL --count DECIMAL]\n"
      "    [--mssql-mode auto|uppercase|direct] [--device N] [--tune off|quick|full]\n"
      "    [--block-size N] [--interleave N] [--batch-candidates N] [--case-budget N]\n"
      "    [--checkpoint FILE] [--results FILE]\n"
      "cudadbcracker resume --checkpoint FILE [--device N] [--tune off|quick|full] [--case-budget N] [--results FILE]\n"
      "cudadbcracker benchmark --algorithm ora11g|mssql --length N --charset-classes CLASSES --count N\n"
      "    [--repeats N] [--device N] [--block-size N] [--interleave N] [--json FILE]\n";
}

int cmdDevices(const Opts& o) {
    auto devs = listDevices();
    if (o.has("json")) {
        std::cout << "[";
        for (std::size_t i = 0; i < devs.size(); ++i) {
            auto& d = devs[i];
            if (i) std::cout << ",";
            std::cout << "{\"index\":" << d.index << ",\"name\":\"" << displayEscape(Bytes(d.name.begin(), d.name.end()))
                      << "\",\"compute_capability\":\"" << d.computeMajor << "." << d.computeMinor
                      << "\",\"sm_count\":" << d.multiProcessorCount << ",\"total_global_mem\":" << d.totalGlobalMem
                      << ",\"max_threads_per_block\":" << d.maxThreadsPerBlock << "}";
        }
        std::cout << "]\n";
        return 0;
    }
    if (devs.empty()) { std::cout << "No CUDA devices available (cuda compiled: " << (cudaCompiled() ? "yes" : "no") << ")\n"; return 0; }
    for (auto& d : devs)
        std::cout << d.index << ": " << d.name << " cc " << d.computeMajor << "." << d.computeMinor
                  << " SMs " << d.multiProcessorCount << " mem " << d.totalGlobalMem << "\n";
    return 0;
}

int cmdVerify(const Opts& o) {
    Algorithm alg = parseAlg(o);
    if (!o.has("input") || !o.has("password-hex")) throw InputError("verify requires --input and --password-hex");
    Bytes pw = hexDecode(o.get("password-hex"), "password");
    if (pw.size() > (std::size_t)kMaxCandidateLen) throw InputError("password longer than 128 bytes");
    if (alg == Algorithm::Mssql) requireMssqlAscii(pw, "MSSQL candidate");
    auto targets = parseTargetFile(o.get("input"), alg);
    std::string backend = o.get("backend", "cpu");
    if (backend != "cpu" && backend != "gpu" && backend != "both") throw InputError("--backend must be cpu, gpu or both");
    int device = parseDevice(o);
    Digest gpuDig{};
    bool haveGpu = false;
    if (backend != "cpu") {
        // Explicit-candidate GPU evaluation for the first target's salt is not
        // target-specific; evaluate per target below instead.
        (void)gpuDig; (void)haveGpu;
    }
    for (auto& t : targets) {
        Digest cpu = digestForCandidate(alg, pw, t.salt);
        if (backend != "cpu") {
            Digest g = gpuDigest(alg, pw, t.salt, device);
            if (g != cpu) throw CorrectnessError("GPU digest disagrees with CPU digest for target '" + t.label + "'");
        }
        bool exact = cpu == t.digest;
        TargetState st; st.length = std::uint32_t(pw.size()); st.rank = "0";
        st.status = exact ? "found" : "verify_mismatch";
        if (exact) st.passwordHex = o.get("password-hex");
        std::cout << resultRecordJson(t, alg, "verify", st, backend) << "\n";
    }
    return 0;
}

bool checkVector(Algorithm alg, std::uint32_t L, const Bytes& salt, const char* expectHex) {
    Bytes pw(L, 'A');
    return hexEncode(digestForCandidate(alg, pw, salt)) == expectHex;
}
int cmdSelfTest(const Opts& o) {
    Bytes saltO = hexDecode("00010203040506070809", "salt");
    Bytes saltM = hexDecode("00010203", "salt");
    bool ok = true;
    ok &= checkVector(Algorithm::Oracle11g, 0, saltO, "494179714a6cd627239dfededf2de9ef994caf03");
    ok &= checkVector(Algorithm::Oracle11g, 45, saltO, "adc1d2c074619bd081cd777316af201715e83b81");
    ok &= checkVector(Algorithm::Oracle11g, 46, saltO, "7878bf63046c4d809b8f814f8c948ae7c2c8bae9");
    ok &= checkVector(Algorithm::Oracle11g, 128, saltO, "a15657b2075f8dc758342fe45bb3b8c795d17b84");
    ok &= checkVector(Algorithm::Mssql, 0, saltM, "a02a05b025b928c039cf1ae7e8ee04e7c190c0db");
    ok &= checkVector(Algorithm::Mssql, 25, saltM, "cf2ed8ad327f0f832ba35b3ab0b62b61ae90ec48");
    ok &= checkVector(Algorithm::Mssql, 26, saltM, "f2dba69e26ad5ff6533a64c3e063dc7f5900edfb");
    ok &= checkVector(Algorithm::Mssql, 128, saltM, "507680873c3c7f1ea65333cd7c3a2f5e76049dc4");
    if (o.has("extended")) {
        // Enumeration aa, ba, ab, bb for alphabet "ab", length 2.
        Bytes alpha{'a', 'b'};
        std::vector<std::string> seq;
        for (std::uint64_t r = 0; r < 4; ++r) {
            auto d = addOffsetToDigits({0, 0}, 2, r);
            auto pw = candidateFromDigits(d, alpha);
            seq.emplace_back(pw.begin(), pw.end());
        }
        ok &= seq == std::vector<std::string>{"aa", "ba", "ab", "bb"};
    }
    if (!ok) throw CorrectnessError("self-test CPU vectors failed");
    if (o.has("device")) {
        int dev = parseDevice(o);
        Digest g = gpuDigest(Algorithm::Oracle11g, Bytes{'A'}, saltO, dev);
        if (g != digestForCandidate(Algorithm::Oracle11g, Bytes{'A'}, saltO))
            throw CorrectnessError("self-test GPU digest disagrees with CPU");
    }
    if (o.has("json")) std::cout << "{\"self_test\":\"passed\",\"extended\":" << (o.has("extended") ? "true" : "false") << "}\n";
    else std::cout << "self-test passed\n";
    return 0;
}

LaunchConfig baseConfig(const Opts& o) {
    LaunchConfig cfg;
    if (o.has("block-size")) {
        auto b = parseU64(o.get("block-size"), "--block-size", 1024);
        if (b != 64 && b != 128 && b != 256 && b != 512) throw InputError("--block-size must be 64, 128, 256 or 512");
        cfg.blockSize = (int)b;
    }
    if (o.has("interleave")) {
        auto v = parseU64(o.get("interleave"), "--interleave", 4);
        if (v != 1 && v != 2 && v != 4) throw InputError("--interleave must be 1, 2 or 4");
        cfg.interleave = (int)v;
    }
    if (o.has("batch-candidates")) cfg.batchCandidates = parseU64(o.get("batch-candidates"), "--batch-candidates", kMaxBatchCandidates);
    if (cfg.batchCandidates == 0) throw InputError("--batch-candidates must be at least 1");
    return cfg;
}

int cmdSearch(const Opts& o) {
    Algorithm alg = parseAlg(o);
    MssqlMode mode = parseMode(o);
    if (alg == Algorithm::Oracle11g && o.has("mssql-mode")) throw InputError("--mssql-mode is only valid with --algorithm mssql");
    if (!o.has("input")) throw InputError("search requires --input");
    if (o.has("charset-classes") && o.has("charset-hex")) throw InputError("--charset-classes and --charset-hex are mutually exclusive");
    auto targets = parseTargetFile(o.get("input"), alg);
    if (targets.empty()) throw InputError("input contains no targets");
    if (alg == Algorithm::Mssql && mode == MssqlMode::Uppercase)
        for (auto& t : targets) if (t.kind == TargetKind::Single)
            throw InputError("explicit uppercase mode requires dual-digest records; target '" + t.label + "' is single-digest");
    std::optional<Bytes> userAlpha;
    if (o.has("charset-classes")) userAlpha = alphabetFromClasses(o.get("charset-classes"));
    else if (o.has("charset-hex")) userAlpha = alphabetFromHex(o.get("charset-hex"));

    Checkpoint cp;
    cp.algorithm = alg; cp.mssqlMode = mode;
    cp.caseBudget = o.has("case-budget") ? parseU64(o.get("case-budget"), "--case-budget") : kDefaultCaseBudget;
    bool explicitMinMax = o.has("min-length") || o.has("max-length");
    if (o.has("length")) {
        if (explicitMinMax) throw InputError("--length conflicts with --min-length/--max-length");
        cp.fixedLength = true;
        cp.minLength = cp.maxLength = parseLen(o.get("length"), "--length");
    } else {
        cp.minLength = o.has("min-length") ? parseLen(o.get("min-length"), "--min-length") : 1;
        cp.maxLength = o.has("max-length") ? parseLen(o.get("max-length"), "--max-length") : 8;
        if (cp.minLength > cp.maxLength) throw InputError("--min-length exceeds --max-length");
    }
    if (o.has("start-rank") || o.has("count")) {
        if (!cp.fixedLength) throw InputError("--start-rank/--count require --length");
        cp.startRank = o.get("start-rank", "0");
        cp.rangeCount = o.get("count", "");
        BigUint start = BigUint::parseDecimal(cp.startRank, "--start-rank");
        (void)start;
        if (!cp.rangeCount.empty()) { BigUint c = BigUint::parseDecimal(cp.rangeCount, "--count"); if (c.isZero()) throw InputError("--count must be at least 1"); }
    }
    cp.targets = targets;
    for (auto& t : targets) {
        cp.alphabets.push_back(effectiveAlphabet(alg, mode, t, userAlpha ? &*userAlpha : nullptr));
        TargetState st; st.length = cp.minLength; st.rank = cp.fixedLength ? cp.startRank : "0";
        cp.states.push_back(st);
    }
    // Validate fixed-length start rank against each target's space.
    if (cp.fixedLength) {
        for (std::size_t i = 0; i < targets.size(); ++i) {
            std::uint32_t B = std::uint32_t(cp.alphabets[i].size());
            auto digits = rankToDigits(cp.startRank, B, cp.minLength); (void)digits;
            if (!cp.rangeCount.empty()) {
                BigUint end = BigUint::parseDecimal(cp.startRank, "--start-rank");
                BigUint cnt = BigUint::parseDecimal(cp.rangeCount, "--count");
                if (cnt.fitsU64()) end.addU64(cnt.toU64());
                if (cnt.fitsU64() && end.compare(spaceSize(B, cp.minLength)) > 0)
                    throw InputError("--start-rank + --count exceeds the candidate space");
            }
        }
    }
    std::string tune = o.get("tune", "quick");
    if (tune != "off" && tune != "quick" && tune != "full") throw InputError("--tune must be off, quick or full");
    std::cerr << "search: algorithm=" << toString(alg) << " targets=" << targets.size()
              << " lengths=" << cp.minLength << ".." << cp.maxLength << " tune=" << tune << "\n";
    return runSearch(cp, o.get("checkpoint", ""), o.get("results", ""), parseDevice(o), tune, baseConfig(o));
}

int cmdResume(const Opts& o) {
    if (!o.has("checkpoint")) throw InputError("resume requires --checkpoint");
    Checkpoint cp = loadCheckpoint(o.get("checkpoint"));
    if (o.has("case-budget")) cp.caseBudget = parseU64(o.get("case-budget"), "--case-budget");
    std::string tune = o.get("tune", "quick");
    if (tune != "off" && tune != "quick" && tune != "full") throw InputError("--tune must be off, quick or full");
    return runSearch(cp, o.get("checkpoint"), o.get("results", ""), parseDevice(o), tune, baseConfig(o));
}

int cmdBenchmark(const Opts& o) {
    Algorithm alg = parseAlg(o);
    if (!o.has("length") || !o.has("count")) throw InputError("benchmark requires --length and --count");
    std::uint32_t L = parseLen(o.get("length"), "--length");
    std::uint64_t count = parseU64(o.get("count"), "--count");
    if (count == 0) throw InputError("--count must be at least 1");
    Bytes alpha;
    if (o.has("charset-hex")) alpha = alphabetFromHex(o.get("charset-hex"));
    else alpha = alphabetFromClasses(o.get("charset-classes", "uln"));
    int repeats = (int)parseU64(o.get("repeats", "3"), "--repeats", 1000);
    if (repeats < 1) throw InputError("--repeats must be at least 1");
    LaunchConfig cfg = baseConfig(o);
    auto r = runBenchmark(alg, L, alpha, count, repeats, parseDevice(o), cfg);
    std::string js = "{\"algorithm\":\"" + std::string(toString(alg)) + "\",\"length\":" + std::to_string(L) +
        ",\"alphabet_size\":" + std::to_string(alpha.size()) + ",\"candidates\":" + std::to_string(r.candidates) +
        ",\"hashes_per_second\":" + std::to_string(r.hashesPerSecond) + ",\"device_seconds\":" + std::to_string(r.deviceSeconds) +
        ",\"block_size\":" + std::to_string(r.blockSize) + ",\"interleave\":" + std::to_string(r.interleave) +
        ",\"registers_per_thread\":" + std::to_string(r.registersPerThread) + ",\"shared_mem_bytes\":" + std::to_string(r.sharedMemBytes) +
        ",\"kernel_family\":\"" + r.kernelFamily + "\"}";
    if (o.has("json")) {
        std::string path = o.get("json");
        std::ofstream out(path); if (!out) throw InputError("cannot write --json file: " + path);
        out << js << "\n";
    }
    std::cout << js << "\n";
    return 0;
}
} // namespace

int runCli(int argc, char** argv) {
    std::signal(SIGINT, [](int) { g_stopRequested.store(true); });
    std::signal(SIGTERM, [](int) { g_stopRequested.store(true); });
    try {
        if (argc < 2) { usage(); return 2; }
        std::string cmd = argv[1];
        if (cmd == "--help" || cmd == "-h" || cmd == "help") { usage(); return 0; }
        Opts o = parseOpts(argc, argv, 2);
        if (cmd == "devices") return cmdDevices(o);
        if (cmd == "verify") return cmdVerify(o);
        if (cmd == "self-test") return cmdSelfTest(o);
        if (cmd == "search") return cmdSearch(o);
        if (cmd == "resume") return cmdResume(o);
        if (cmd == "benchmark") return cmdBenchmark(o);
        std::cerr << "unknown subcommand '" << cmd << "'\n";
        usage();
        return 2;
    } catch (const CorrectnessError& e) { std::cerr << "correctness error: " << e.what() << "\n"; return 4; }
      catch (const CudaError& e) { std::cerr << "cuda error: " << e.what() << "\n"; return 3; }
      catch (const InputError& e) { std::cerr << "error: " << e.what() << "\n"; return 2; }
      catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 2; }
}
} // namespace cudadb
