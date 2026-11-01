#include "cudadb/algorithms.hpp"
#include "cudadb/dispatch.hpp"
#include "cudadb/enumeration.hpp"
#include <iostream>
using namespace cudadb;
// Exit 77 = CTest skip (SKIP_RETURN_CODE): no device is reported, not passed.
int main() {
    if (!cudaAvailable()) {
        std::cout << "SKIPPED: no CUDA device available (cuda compiled: "
                  << (cudaCompiled() ? "yes" : "no") << ")\n";
        return 77;
    }
    Bytes saltO = hexDecode("0102030405060708090a", "s");
    Bytes saltM = hexDecode("0a0b0c0d", "s");
    // Explicit-candidate GPU digest agrees with CPU for both algorithms,
    // including MSSQL lengths 10-12 and multi-block lengths.
    for (int L : {0, 1, 10, 12, 26, 45, 128}) {
        Bytes pw(std::size_t(L), 'A');
        if (gpuDigest(Algorithm::Oracle11g, pw, saltO, 0) != digestForCandidate(Algorithm::Oracle11g, pw, saltO)) {
            std::cerr << "FAIL gpu oracle digest L=" << L << "\n"; return 1;
        }
    }
    for (int L : {0, 1, 10, 12, 25, 26, 128}) {
        Bytes pw(std::size_t(L), 'A');
        if (gpuDigest(Algorithm::Mssql, pw, saltM, 0) != digestForCandidate(Algorithm::Mssql, pw, saltM)) {
            std::cerr << "FAIL gpu mssql digest L=" << L << "\n"; return 1;
        }
    }
    // GPU search: grouped fast path and generic path find the same offsets
    // as the CPU control.
    LaunchConfig cfg; // fallback 128/1
    Bytes alpha{'a','b','c'};
    // length L, all 'a' except first byte = `first` (so rank = index(first)).
    auto med = [](int L, std::uint8_t first) { Bytes b(std::size_t(L), 'a'); b[0] = first; return b; };
    struct Case { Algorithm a; Bytes pw; Bytes salt; };
    std::vector<Case> cases = {
        {Algorithm::Oracle11g, {'c','a','b'}, saltO},
        {Algorithm::Oracle11g, {'a','a','a'}, saltO},
        {Algorithm::Oracle11g, {'c','c','c'}, saltO},
        {Algorithm::Mssql, {'B','A'}, saltM},
        // Medium lengths beyond the fast kernel (L>16): exercise the generic
        // single-block path (Oracle <=45, MSSQL <=25) against the CPU control.
        {Algorithm::Oracle11g, med(17, 'c'), saltO},
        {Algorithm::Oracle11g, med(45, 'b'), saltO}, // single-block boundary
        {Algorithm::Mssql, med(17, 'c'), saltM},
        {Algorithm::Mssql, med(25, 'b'), saltM},     // single-block boundary
    };
    for (auto& c : cases) {
        Target t; t.label = "t"; t.salt = c.salt; t.digest = digestForCandidate(c.a, c.pw, c.salt);
        std::vector<std::uint8_t> base(c.pw.size(), 0);
        auto cpu = cpuSearch(c.a, t, t.digest, alpha, base, 27);
        auto gpu = gpuSearch(c.a, t, t.digest, alpha, base, 27, 0, cfg);
        if (cpu != gpu) { std::cerr << "FAIL gpu search offset mismatch\n"; return 1; }
        LaunchConfig cfg2 = cfg; cfg2.interleave = 2;
        auto gpu2 = gpuSearch(c.a, t, t.digest, alpha, base, 27, 0, cfg2);
        if (cpu != gpu2) { std::cerr << "FAIL gpu search (interleave 2) offset mismatch\n"; return 1; }
        LaunchConfig cfg4 = cfg; cfg4.interleave = 4;
        auto gpu4 = gpuSearch(c.a, t, t.digest, alpha, base, 27, 0, cfg4);
        if (cpu != gpu4) { std::cerr << "FAIL gpu search (interleave 4) offset mismatch\n"; return 1; }
    }
    std::cout << "test_gpu passed\n";
    return 0;
}
