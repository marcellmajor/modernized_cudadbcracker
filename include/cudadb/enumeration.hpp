#pragma once
#include "cudadb/types.hpp"
#include <optional>
#include <string>
#include <vector>

namespace cudadb {

// Minimal arbitrary-precision unsigned integer (spec section 7 allows
// cpp_int "or equivalent digit arrays"; this avoids a Boost dependency).
class BigUint {
public:
    BigUint() = default;
    static BigUint fromU64(std::uint64_t v);
    static BigUint parseDecimal(const std::string& s, const char* what);
    std::string toDecimal() const;
    bool isZero() const { return limbs_.empty(); }
    int compare(const BigUint& o) const;
    bool operator==(const BigUint& o) const { return compare(o) == 0; }
    bool operator<(const BigUint& o) const { return compare(o) < 0; }
    bool operator>=(const BigUint& o) const { return compare(o) >= 0; }
    void addU64(std::uint64_t v);
    void mulSmall(std::uint32_t m);
    void addSmall(std::uint32_t v);
    bool bit(std::size_t i) const;
    void increment();
    BigUint sub(const BigUint& o) const; // requires *this >= o
    bool fitsU64() const;
    std::uint64_t toU64() const; // requires fitsU64()
    // Divide by small, returns remainder; used for base-B conversion.
    std::uint32_t divSmall(std::uint32_t d);
private:
    std::vector<std::uint32_t> limbs_; // little-endian, no trailing zeros
    void trim();
};

BigUint spaceSize(std::uint32_t B, std::uint32_t L); // B^L

// Convert a decimal rank to base-B digits, digit 0 = first password byte.
// Throws InputError if rank >= B^L. Result has exactly L digits.
std::vector<std::uint8_t> rankToDigits(const std::string& rankDecimal, std::uint32_t B, std::uint32_t L);
// Convert base-B digits back to a decimal rank string.
std::string digitsToDecimal(const std::vector<std::uint8_t>& digits, std::uint32_t B);
// digits += offset in base B. Caller guarantees the sum stays within L digits
// (range-checked batches); overflow beyond L digits throws InputError.
std::vector<std::uint8_t> addOffsetToDigits(std::vector<std::uint8_t> digits, std::uint32_t B, std::uint64_t offset);

Bytes candidateFromDigits(const std::vector<std::uint8_t>& digits, const Bytes& alphabet);

// CPU search over a bounded batch: offsets [0,count) from baseDigits.
// Returns the smallest matching offset, or nullopt. Used by tests and as the
// correctness control for the GPU path (spec sections 9.1, 14).
std::optional<std::uint64_t> cpuSearch(Algorithm alg, const Target& target, const Digest& wanted,
                                       const Bytes& alphabet, const std::vector<std::uint8_t>& baseDigits,
                                       std::uint64_t count);

} // namespace cudadb
