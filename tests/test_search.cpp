#include "cudadb/algorithms.hpp"
#include "cudadb/enumeration.hpp"
#include "cudadb/input.hpp"
#include <iostream>
using namespace cudadb;
static int failures = 0;
#define CHECK(cond) do { if(!(cond)) { std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; ++failures; } } while(0)

static Target makeTarget(Algorithm a, const Bytes& salt, const Bytes& pw) {
    Target t; t.label = "t"; t.salt = salt; t.digest = digestForCandidate(a, pw, salt);
    return t;
}
int main() {
    Bytes saltO = hexDecode("0102030405060708090a", "s");
    Bytes alpha{'a','b','c'};
    // Exhaustive tiny search: password "cab" - rank: d=[1? ] compute via digits.
    Bytes pw{'c','a','b'}; // d0=2(c), d1=0(a), d2=1(b): rank=2+0*3+1*9=11
    Target t = makeTarget(Algorithm::Oracle11g, saltO, pw);
    auto hit = cpuSearch(Algorithm::Oracle11g, t, t.digest, alpha, {0,0,0}, 27);
    CHECK(hit && *hit == 11);
    // First / last candidate.
    Target t0 = makeTarget(Algorithm::Oracle11g, saltO, Bytes{'a','a','a'});
    CHECK(cpuSearch(Algorithm::Oracle11g, t0, t0.digest, alpha, {0,0,0}, 27).value() == 0);
    Target tL = makeTarget(Algorithm::Oracle11g, saltO, Bytes{'c','c','c'});
    CHECK(cpuSearch(Algorithm::Oracle11g, tL, tL.digest, alpha, {0,0,0}, 27).value() == 26);
    // No match.
    CHECK(!cpuSearch(Algorithm::Oracle11g, t, t.digest, alpha, {0,0,0}, 5));
    // Unaligned start: rank 10.. covers hit at local offset 1.
    auto base = rankToDigits("10", 3, 3);
    auto hit2 = cpuSearch(Algorithm::Oracle11g, t, t.digest, alpha, base, 10);
    CHECK(hit2 && *hit2 == 1);
    // B=1.
    Bytes one{'z'};
    Target t1 = makeTarget(Algorithm::Oracle11g, saltO, Bytes{'z','z'});
    CHECK(cpuSearch(Algorithm::Oracle11g, t1, t1.digest, one, {0,0}, 1).value() == 0);
    // MSSQL lengths incl. former bug range 10-12 and multi-block 26.
    Bytes saltM = hexDecode("0a0b0c0d", "s");
    for (int L : {1, 9, 10, 11, 12, 25, 26}) {
        Bytes p(std::size_t(L), 'B'); p[0] = 'A';
        Target tm = makeTarget(Algorithm::Mssql, saltM, p);
        // Base digits are exactly p over alphabet {'A','B'}: A=0, B=1.
        auto digits = std::vector<std::uint8_t>(std::size_t(L), 1);
        digits[0] = 0;
        auto h = cpuSearch(Algorithm::Mssql, tm, tm.digest, Bytes{'A','B'}, digits, 1);
        CHECK(h && *h == 0);
    }
    // Case recovery: full, budgeted+resume, and no-match.
    Target dual; dual.label = "d"; dual.salt = saltM;
    Bytes mixed{'a','B','3','c'};
    dual.digest = digestForCandidate(Algorithm::Mssql, mixed, saltM);
    dual.upperDigest = digestForCandidate(Algorithm::Mssql, asciiUpperBytes(mixed), saltM);
    dual.kind = TargetKind::DualUppercase;
    Bytes upper = asciiUpperBytes(mixed);
    auto rec = recoverCase(upper, dual, 1000000, "0");
    CHECK(rec.outcome == CaseRecoveryResult::Outcome::Found);
    CHECK(rec.password == mixed);
    auto part = recoverCase(upper, dual, 1, "0");
    CHECK(part.outcome == CaseRecoveryResult::Outcome::BudgetExhausted);
    auto rec2 = recoverCase(upper, dual, 1000000, part.nextAssignment);
    CHECK(rec2.outcome == CaseRecoveryResult::Outcome::Found && rec2.password == mixed);
    Target wrong = dual; wrong.digest = digestForCandidate(Algorithm::Mssql, Bytes{'x','Y','3','z'}, saltM);
    auto none = recoverCase(upper, wrong, 1000000, "0");
    CHECK(none.outcome == CaseRecoveryResult::Outcome::NoMatch);
    // Digits do not double trials: '3' is not alphabetic.
    CHECK(countAlphabetic(upper) == 3);

    // Known hit in a length-128 tiny bounded range (Oracle).
    Bytes big(128, 'a'); big[0] = 'b';
    Target tb = makeTarget(Algorithm::Oracle11g, saltO, big);
    auto bd = std::vector<std::uint8_t>(128, 0);
    auto hb = cpuSearch(Algorithm::Oracle11g, tb, tb.digest, Bytes{'a','b'}, bd, 2);
    CHECK(hb && *hb == 1);

    if (failures) { std::cerr << failures << " failures\n"; return 1; }
    std::cout << "test_search passed\n";
    return 0;
}
