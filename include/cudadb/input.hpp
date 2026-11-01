#pragma once
#include "cudadb/types.hpp"
#include <string>
#include <vector>

namespace cudadb {

// Target file parsing (spec section 6.1). One algorithm per file.
// - label is the text before the first ':'; labels do not enter the hash.
// - Blank lines and lines whose first non-space character is '#' are ignored.
// - A trailing CR (CRLF files) is part of the line ending, not the verifier.
// - Errors carry the 1-based line number.
std::vector<Target> parseTargetFile(const std::string& path, Algorithm alg);

// Password files of the legacy fixtures: "label:password" per line.
std::vector<std::pair<std::string, std::string>> parsePasswordFile(const std::string& path);

// Alphabet construction (spec section 6.2).
Bytes alphabetFromClasses(const std::string& classes); // u,l,n,s in given order; unknown letter = InputError
Bytes alphabetFromHex(const std::string& hex);         // continuous even-length hex
Bytes dedupPreserveFirst(Bytes in);
Bytes effectiveAlphabet(Algorithm alg, MssqlMode mode, const Target& t,
                        const Bytes* userAlphabet); // nullptr = spec defaults
Bytes uppercaseAlphabet(Bytes a);

// Target fingerprint for checkpoints: SHA-1 hex over a canonical string of
// algorithm, salt and digest fields. The account label is excluded
// (decision recorded in DECISIONS.md; spec leaves the algorithm open).
std::string targetFingerprint(Algorithm alg, const Target& t);

} // namespace cudadb
