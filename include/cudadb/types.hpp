#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudadb {

constexpr int kMaxCandidateLen = 128;
constexpr std::uint64_t kMaxBatchCandidates = 4294967295ULL; // 2^32-1, spec section 7
constexpr std::uint64_t kDefaultCaseBudget = 1048576ULL;     // spec section 4.3

enum class Algorithm { Oracle11g, Mssql };
enum class MssqlMode { Auto, Uppercase, Direct };
enum class TargetKind { Single, DualUppercase };

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };
struct InputError : Error { using Error::Error; };       // exit 2
struct CudaError : Error { using Error::Error; };        // exit 3
struct CorrectnessError : Error { using Error::Error; }; // exit 4

using Bytes = std::vector<std::uint8_t>;
using Digest = std::array<std::uint8_t, 20>;

struct Target {
    std::string label;
    Bytes salt;                       // 10 bytes Oracle, 4 bytes MSSQL
    Digest digest{};                  // case-sensitive digest (Oracle: the digest)
    std::optional<Digest> upperDigest; // MSSQL dual form only
    TargetKind kind = TargetKind::Single;
};

inline const char* toString(Algorithm a) { return a == Algorithm::Oracle11g ? "ora11g" : "mssql"; }
inline const char* toString(MssqlMode m) {
    switch (m) { case MssqlMode::Auto: return "auto"; case MssqlMode::Uppercase: return "uppercase"; default: return "direct"; }
}

std::string hexEncode(const std::uint8_t* data, std::size_t n, bool upper = false);
inline std::string hexEncode(const Bytes& b, bool upper = false) { return hexEncode(b.data(), b.size(), upper); }
inline std::string hexEncode(const Digest& d, bool upper = false) { return hexEncode(d.data(), d.size(), upper); }
Bytes hexDecode(const std::string& hex, const char* what); // strict, even length

std::array<std::uint32_t, 5> digestWords(const Digest& d); // big-endian, spec section 7
Digest digestFromWords(const std::array<std::uint32_t, 5>& w);

char asciiUpperChar(char c);
std::uint8_t asciiUpperByte(std::uint8_t c);
Bytes asciiUpperBytes(const Bytes& in);
bool isAscii(const Bytes& b);

// Display form for JSON: printable ASCII as-is, others as \xNN (documented in README).
std::string displayEscape(const std::uint8_t* data, std::size_t n);
inline std::string displayEscape(const Bytes& b) { return displayEscape(b.data(), b.size()); }

} // namespace cudadb
