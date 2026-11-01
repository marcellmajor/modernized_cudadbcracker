#include "cudadb/enumeration.hpp"
#include "cudadb/algorithms.hpp"
#include <algorithm>

namespace cudadb {

// ---- BigUint ----
void BigUint::trim() { while (!limbs_.empty() && limbs_.back() == 0) limbs_.pop_back(); }
BigUint BigUint::fromU64(std::uint64_t v) {
    BigUint b;
    while (v) { b.limbs_.push_back(std::uint32_t(v & 0xffffffffULL)); v >>= 32; }
    return b;
}
BigUint BigUint::parseDecimal(const std::string& s, const char* what) {
    if (s.empty()) throw InputError(std::string(what) + ": empty decimal integer");
    BigUint b;
    for (char c : s) {
        if (c < '0' || c > '9') throw InputError(std::string(what) + ": invalid decimal integer '" + s + "'");
        b.mulSmall(10); b.addSmall(std::uint32_t(c - '0'));
    }
    return b;
}
std::string BigUint::toDecimal() const {
    if (isZero()) return "0";
    BigUint tmp = *this;
    std::string out;
    while (!tmp.isZero()) { std::uint32_t r = tmp.divSmall(10); out.push_back(char('0' + r)); }
    std::reverse(out.begin(), out.end());
    return out;
}
int BigUint::compare(const BigUint& o) const {
    if (limbs_.size() != o.limbs_.size()) return limbs_.size() < o.limbs_.size() ? -1 : 1;
    for (std::size_t i = limbs_.size(); i-- > 0;)
        if (limbs_[i] != o.limbs_[i]) return limbs_[i] < o.limbs_[i] ? -1 : 1;
    return 0;
}
void BigUint::addU64(std::uint64_t v) {
    std::uint64_t carry = v;
    std::size_t i = 0;
    while (carry) {
        if (i == limbs_.size()) limbs_.push_back(0);
        std::uint64_t cur = std::uint64_t(limbs_[i]) + (carry & 0xffffffffULL);
        limbs_[i] = std::uint32_t(cur & 0xffffffffULL);
        carry = (carry >> 32) + (cur >> 32);
        ++i;
    }
}
void BigUint::mulSmall(std::uint32_t m) {
    if (m == 0) { limbs_.clear(); return; }
    std::uint64_t carry = 0;
    for (auto& l : limbs_) { std::uint64_t cur = std::uint64_t(l) * m + carry; l = std::uint32_t(cur); carry = cur >> 32; }
    while (carry) { limbs_.push_back(std::uint32_t(carry)); carry >>= 32; }
}
void BigUint::addSmall(std::uint32_t v) { addU64(v); }
bool BigUint::bit(std::size_t i) const {
    std::size_t limb = i / 32;
    if (limb >= limbs_.size()) return false;
    return (limbs_[limb] >> (i % 32)) & 1u;
}
void BigUint::increment() { addU64(1); }
BigUint BigUint::sub(const BigUint& o) const {
    BigUint r = *this;
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < r.limbs_.size(); ++i) {
        std::uint64_t sub = (i < o.limbs_.size() ? o.limbs_[i] : 0) + borrow;
        std::uint64_t cur = r.limbs_[i];
        if (cur < sub) { r.limbs_[i] = std::uint32_t((cur + (1ULL << 32)) - sub); borrow = 1; }
        else { r.limbs_[i] = std::uint32_t(cur - sub); borrow = 0; }
    }
    r.trim();
    return r;
}
bool BigUint::fitsU64() const { return limbs_.size() <= 2; }
std::uint64_t BigUint::toU64() const {
    std::uint64_t v = 0;
    for (std::size_t i = limbs_.size(); i-- > 0;) v = (v << 32) | limbs_[i];
    return v;
}
std::uint32_t BigUint::divSmall(std::uint32_t d) {
    std::uint64_t rem = 0;
    for (std::size_t i = limbs_.size(); i-- > 0;) {
        std::uint64_t cur = (rem << 32) | limbs_[i];
        limbs_[i] = std::uint32_t(cur / d);
        rem = cur % d;
    }
    trim();
    return std::uint32_t(rem);
}

BigUint spaceSize(std::uint32_t B, std::uint32_t L) {
    BigUint r = BigUint::fromU64(1);
    for (std::uint32_t i = 0; i < L; ++i) r.mulSmall(B);
    return r;
}

std::vector<std::uint8_t> rankToDigits(const std::string& rankDecimal, std::uint32_t B, std::uint32_t L) {
    if (B == 0) throw InputError("alphabet size is zero");
    BigUint rank = BigUint::parseDecimal(rankDecimal, "rank");
    std::vector<std::uint8_t> digits;
    digits.reserve(L);
    BigUint tmp = rank;
    for (std::uint32_t i = 0; i < L; ++i) {
        std::uint32_t r = (B == 1) ? 0 : tmp.divSmall(B);
        if (B == 1 && !tmp.isZero() && i == 0 && L == 0) { /* handled below */ }
        digits.push_back(std::uint8_t(r));
        if (B == 1) { /* only rank 0 exists; tmp must stay zero */ }
    }
    if (!tmp.isZero()) throw InputError("rank is outside the requested length space");
    if (B == 1 && !(rank.isZero())) throw InputError("rank is outside the requested length space");
    return digits;
}
std::string digitsToDecimal(const std::vector<std::uint8_t>& digits, std::uint32_t B) {
    BigUint r;
    for (std::size_t i = digits.size(); i-- > 0;) { r.mulSmall(B); r.addSmall(digits[i]); }
    return r.toDecimal();
}
std::vector<std::uint8_t> addOffsetToDigits(std::vector<std::uint8_t> digits, std::uint32_t B, std::uint64_t offset) {
    if (B == 0) throw InputError("alphabet size is zero");
    if (B == 1) {
        if (offset != 0) throw InputError("candidate offset overflows the length space");
        return digits;
    }
    std::uint64_t carry = offset;
    std::size_t i = 0;
    while (carry) {
        if (i >= digits.size()) throw InputError("candidate offset overflows the length space");
        std::uint64_t add = carry % B;
        carry /= B;
        std::uint64_t sum = std::uint64_t(digits[i]) + add;
        if (sum >= B) { sum -= B; carry += 1; }
        digits[i] = std::uint8_t(sum);
        ++i;
    }
    return digits;
}
Bytes candidateFromDigits(const std::vector<std::uint8_t>& digits, const Bytes& alphabet) {
    Bytes out; out.reserve(digits.size());
    for (auto d : digits) {
        if (d >= alphabet.size()) throw InputError("candidate digit outside alphabet");
        out.push_back(alphabet[d]);
    }
    return out;
}

std::optional<std::uint64_t> cpuSearch(Algorithm alg, const Target& target, const Digest& wanted,
                                       const Bytes& alphabet, const std::vector<std::uint8_t>& baseDigits,
                                       std::uint64_t count) {
    const std::uint32_t B = std::uint32_t(alphabet.size());
    for (std::uint64_t off = 0; off < count; ++off) {
        auto digits = addOffsetToDigits(baseDigits, B, off);
        Bytes pw = candidateFromDigits(digits, alphabet);
        if (digestForCandidate(alg, pw, target.salt) == wanted) return off;
    }
    return std::nullopt;
}

// ---- Case recovery (spec section 4.3) ----
CaseRecoveryResult recoverCase(const Bytes& upperCandidate, const Target& t,
                               std::uint64_t budget, const std::string& startAssignment) {
    CaseRecoveryResult res;
    // Alphabetic positions, left to right.
    std::vector<std::size_t> pos;
    for (std::size_t i = 0; i < upperCandidate.size(); ++i) {
        auto c = upperCandidate[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) pos.push_back(i);
    }
    BigUint idx = BigUint::parseDecimal(startAssignment.empty() ? "0" : startAssignment, "case assignment");
    BigUint total = spaceSize(2, std::uint32_t(pos.size()));
    std::uint64_t trials = 0;
    while (idx.compare(total) < 0) {
        if (trials >= budget) {
            res.outcome = CaseRecoveryResult::Outcome::BudgetExhausted;
            res.nextAssignment = idx.toDecimal();
            res.trialsUsed = trials;
            return res;
        }
        Bytes cand = upperCandidate;
        for (std::size_t k = 0; k < pos.size(); ++k) {
            std::uint8_t c = cand[pos[k]];
            bool upper = idx.bit(k); // set = uppercase, clear = lowercase
            if (upper) cand[pos[k]] = asciiUpperByte(c);
            else cand[pos[k]] = (c >= 'A' && c <= 'Z') ? std::uint8_t(c - 'A' + 'a') : c;
        }
        ++trials;
        if (verifyCandidate(Algorithm::Mssql, t, cand)) {
            res.outcome = CaseRecoveryResult::Outcome::Found;
            res.password = cand;
            res.trialsUsed = trials;
            return res;
        }
        idx.increment();
    }
    res.outcome = CaseRecoveryResult::Outcome::NoMatch;
    res.trialsUsed = trials;
    return res;
}

} // namespace cudadb
