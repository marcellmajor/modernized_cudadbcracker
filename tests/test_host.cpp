#include "cudadb/algorithms.hpp"
#include "cudadb/enumeration.hpp"
#include "cudadb/input.hpp"
#include <cstdio>
#include <iostream>
#include <map>
using namespace cudadb;
static int failures = 0;
#define CHECK(cond) do { if(!(cond)) { std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; ++failures; } } while(0)
#define CHECK_THROWS(expr) do { bool t=false; try { expr; } catch(const InputError&) { t=true; } if(!t) { std::cerr << "FAIL (no throw) line " << __LINE__ << ": " #expr "\n"; ++failures; } } while(0)

int main() {
    // Spec section 14.1 known answers (Python hashlib).
    Bytes sO = hexDecode("00010203040506070809", "s"), sM = hexDecode("00010203", "s");
    struct V { Algorithm a; int L; const Bytes* s; const char* hex; };
    std::vector<V> vs = {
        {Algorithm::Oracle11g, 0, &sO, "494179714a6cd627239dfededf2de9ef994caf03"},
        {Algorithm::Oracle11g, 12, &sO, "9dc1e333a761dd1ddecb6a49dcd3958d2e3f7728"},
        {Algorithm::Oracle11g, 45, &sO, "adc1d2c074619bd081cd777316af201715e83b81"},
        {Algorithm::Oracle11g, 46, &sO, "7878bf63046c4d809b8f814f8c948ae7c2c8bae9"},
        {Algorithm::Oracle11g, 128, &sO, "a15657b2075f8dc758342fe45bb3b8c795d17b84"},
        {Algorithm::Mssql, 0, &sM, "a02a05b025b928c039cf1ae7e8ee04e7c190c0db"},
        {Algorithm::Mssql, 10, &sM, "d0961996ec86ebc8fd3d06c55ae345484b2e954d"},
        {Algorithm::Mssql, 12, &sM, "1ccba80b1c4e3c50c94b4d409e031ea7bdd90c87"},
        {Algorithm::Mssql, 25, &sM, "cf2ed8ad327f0f832ba35b3ab0b62b61ae90ec48"},
        {Algorithm::Mssql, 26, &sM, "f2dba69e26ad5ff6533a64c3e063dc7f5900edfb"},
        {Algorithm::Mssql, 128, &sM, "507680873c3c7f1ea65333cd7c3a2f5e76049dc4"},
    };
    for (auto& v : vs) {
        Bytes pw(std::size_t(v.L), 'A');
        CHECK(hexEncode(digestForCandidate(v.a, pw, *v.s)) == v.hex);
    }
    // Standard SHA-1 vectors.
    CHECK(hexEncode(sha1(Bytes{})) == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    Bytes abc{'a','b','c'};
    CHECK(hexEncode(sha1(abc)) == "a9993e364706816aba3e25717850c26c9cd0d89d");

    // Parsing fixtures (CRLF files).
    auto ora = parseTargetFile(std::string(FIXTURES_DIR) + "/orasha_testcase.txt", Algorithm::Oracle11g);
    auto ms = parseTargetFile(std::string(FIXTURES_DIR) + "/mssql_testcase.txt", Algorithm::Mssql);
    CHECK(ora.size() == 47);
    CHECK(ms.size() == 47);
    CHECK(ms[0].kind == TargetKind::DualUppercase);
    CHECK(ms[0].salt.size() == 4 && ora[0].salt.size() == 10);
    auto opw = parsePasswordFile(std::string(FIXTURES_DIR) + "/orasha_testcase_pass.txt");
    auto mpw = parsePasswordFile(std::string(FIXTURES_DIR) + "/mssql_testcase_pass.txt");
    CHECK(opw.size() == 47 && mpw.size() == 47);
    std::map<std::string,std::string> om(opw.begin(), opw.end()), mm(mpw.begin(), mpw.end());
    for (auto& t : ora) {
        Bytes pw(om[t.label].begin(), om[t.label].end());
        CHECK(verifyCandidate(Algorithm::Oracle11g, t, pw));
    }
    for (auto& t : ms) {
        Bytes pw(mm[t.label].begin(), mm[t.label].end());
        CHECK(verifyCandidate(Algorithm::Mssql, t, pw));
        CHECK(verifyUppercaseForm(Algorithm::Mssql, t, pw));
    }
    // Strict parser rejection (via temp paths exercised in test_resume for IO;
    // here: hex and alphabet rules).
    CHECK_THROWS(hexDecode("abc", "x"));
    CHECK_THROWS(hexDecode("zz", "x"));
    CHECK_THROWS(alphabetFromClasses("ux"));
    CHECK(alphabetFromClasses("s").size() == 32);
    CHECK(alphabetFromClasses("s")[0] == 0x21 && alphabetFromClasses("s")[19] == 0x7d);
    CHECK_THROWS(requireMssqlAscii(Bytes{0x80}, "x"));
    CHECK(effectiveAlphabet(Algorithm::Mssql, MssqlMode::Auto, ms[0], nullptr).size() == 36); // un
    CHECK(effectiveAlphabet(Algorithm::Oracle11g, MssqlMode::Auto, ora[0], nullptr).size() == 62); // uln

    // Enumeration order: digit 0 fastest (spec section 7).
    Bytes ab{'a','b'};
    std::vector<std::string> seq;
    for (std::uint64_t r = 0; r < 4; ++r) {
        auto pw = candidateFromDigits(addOffsetToDigits({0,0}, 2, r), ab);
        seq.emplace_back(pw.begin(), pw.end());
    }
    CHECK((seq == std::vector<std::string>{"aa","ba","ab","bb"}));
    CHECK(digitsToDecimal({1,0}, 2) == "1");
    CHECK(digitsToDecimal({0,1}, 2) == "2");
    CHECK((rankToDigits("3", 2, 2) == std::vector<std::uint8_t>{1,1}));
    CHECK_THROWS(rankToDigits("4", 2, 2));
    // B=1 and L=0.
    CHECK((rankToDigits("0", 1, 3) == std::vector<std::uint8_t>{0,0,0}));
    CHECK_THROWS(rankToDigits("1", 1, 3));
    CHECK(rankToDigits("0", 62, 0).empty());
    // Beyond 2^64: 62^12 space, rank near the top round-trips.
    auto top = spaceSize(62, 12).sub(BigUint::fromU64(1));
    auto dg = rankToDigits(top.toDecimal(), 62, 12);
    CHECK(digitsToDecimal(dg, 62) == top.toDecimal());
    CHECK(dg[11] == 61);

    // Strict parser rejection on malformed files.
    auto writeTmp = [](const char* name, const std::string& content) {
        std::string p = std::string("/tmp/") + name;
        FILE* f = std::fopen(p.c_str(), "wb"); std::fputs(content.c_str(), f); std::fclose(f);
        return p;
    };
    CHECK_THROWS(parseTargetFile(writeTmp("cudadb_bad1.txt", "u1:ABCD\n"), Algorithm::Oracle11g)); // short verifier
    CHECK_THROWS(parseTargetFile(writeTmp("cudadb_bad2.txt", "u1:" + std::string(59, 'A') + "Z\n"), Algorithm::Oracle11g)); // bad hex
    CHECK_THROWS(parseTargetFile(writeTmp("cudadb_bad3.txt", "u1:0x0200" + std::string(88, 'A') + "\n"), Algorithm::Mssql)); // rejected version
    CHECK_THROWS(parseTargetFile(writeTmp("cudadb_bad4.txt", "u1:0x0100" + std::string(48, 'A') + "FF\n"), Algorithm::Mssql)); // trailing data
    std::string rec = "u1:" + std::string(60, 'A') + "\r\n";
    CHECK_THROWS(parseTargetFile(writeTmp("cudadb_bad5.txt", rec + rec), Algorithm::Oracle11g)); // duplicate label
    // Accepted Oracle S: prefix and CRLF line ending.
    auto sp = parseTargetFile(writeTmp("cudadb_ok1.txt", "u1:S:" + std::string(60, 'B') + "\r\n"), Algorithm::Oracle11g);
    CHECK(sp.size() == 1 && sp[0].salt.size() == 10);

    if (failures) { std::cerr << failures << " failures\n"; return 1; }
    std::cout << "test_host passed\n";
    return 0;
}
