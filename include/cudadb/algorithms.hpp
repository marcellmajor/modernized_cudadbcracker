#pragma once
#include "cudadb/types.hpp"

namespace cudadb {

// Straightforward reference SHA-1 (FIPS 180-1). Independent of the CUDA core;
// test expectations come from Python hashlib fixtures (spec sections 8, 14).
Digest sha1(const std::uint8_t* data, std::size_t n);
inline Digest sha1(const Bytes& b) { return sha1(b.data(), b.size()); }

// Exact database constructions (spec section 4).
Bytes buildMessage(Algorithm alg, const Bytes& password, const Bytes& salt);
Digest digestForCandidate(Algorithm alg, const Bytes& password, const Bytes& salt);

// MSSQL search domain is ASCII only (spec section 4.2). Throws InputError otherwise.
void requireMssqlAscii(const Bytes& password, const char* what);

bool verifyCandidate(Algorithm alg, const Target& t, const Bytes& password);       // case-sensitive digest
bool verifyUppercaseForm(Algorithm alg, const Target& t, const Bytes& password);   // dual form: digest of ASCII-uppercase(password)

struct CaseRecoveryResult {
    enum class Outcome { Found, BudgetExhausted, NoMatch } outcome;
    Bytes password;              // valid when Found
    std::string nextAssignment;  // decimal, valid when BudgetExhausted
    std::uint64_t trialsUsed = 0;
};

// Enumerate case only at ASCII alphabetic positions (spec section 4.3).
// Assignment index: bit i (LSB first) controls the i-th alphabetic position
// left-to-right; set = uppercase, clear = lowercase. Index 0 = all lowercase.
// startAssignment is a decimal string; budget bounds trials in this call.
CaseRecoveryResult recoverCase(const Bytes& upperCandidate, const Target& t,
                               std::uint64_t budget, const std::string& startAssignment);

std::size_t countAlphabetic(const Bytes& b);

} // namespace cudadb
