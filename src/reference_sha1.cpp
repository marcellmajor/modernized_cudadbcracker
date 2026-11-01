#include "cudadb/algorithms.hpp"
#include <cstring>

namespace cudadb {
namespace {
inline std::uint32_t rol(std::uint32_t x, int y) { return (x << y) | (x >> (32 - y)); }

void compress(std::uint32_t h[5], const std::uint8_t* block) {
    std::uint32_t w[80];
    for (int i = 0; i < 16; ++i)
        w[i] = (std::uint32_t(block[4*i]) << 24) | (std::uint32_t(block[4*i+1]) << 16) |
               (std::uint32_t(block[4*i+2]) << 8) | std::uint32_t(block[4*i+3]);
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
        std::uint32_t f, k;
        if (i < 20)      { f = (b & c) | ((~b) & d);          k = 0x5A827999u; }
        else if (i < 40) { f = b ^ c ^ d;                     k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d);   k = 0x8F1BBCDCu; }
        else             { f = b ^ c ^ d;                     k = 0xCA62C1D6u; }
        std::uint32_t tmp = rol(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rol(b, 30); b = a; a = tmp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}
} // namespace

Digest sha1(const std::uint8_t* data, std::size_t n) {
    std::uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::uint64_t bitLen = std::uint64_t(n) * 8;
    std::vector<std::uint8_t> msg;
    if (n) msg.assign(data, data + n);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(std::uint8_t(bitLen >> (8 * i)));
    for (std::size_t off = 0; off < msg.size(); off += 64) compress(h, msg.data() + off);
    Digest out{};
    for (int i = 0; i < 5; ++i) {
        out[4*i] = std::uint8_t(h[i] >> 24); out[4*i+1] = std::uint8_t(h[i] >> 16);
        out[4*i+2] = std::uint8_t(h[i] >> 8); out[4*i+3] = std::uint8_t(h[i]);
    }
    return out;
}

Bytes buildMessage(Algorithm alg, const Bytes& password, const Bytes& salt) {
    Bytes msg;
    if (alg == Algorithm::Oracle11g) {
        msg = password;
    } else {
        requireMssqlAscii(password, "MSSQL candidate");
        msg.reserve(password.size() * 2);
        for (auto c : password) { msg.push_back(c); msg.push_back(0); }
    }
    msg.insert(msg.end(), salt.begin(), salt.end());
    return msg;
}
Digest digestForCandidate(Algorithm alg, const Bytes& password, const Bytes& salt) {
    return sha1(buildMessage(alg, password, salt));
}
void requireMssqlAscii(const Bytes& password, const char* what) {
    if (!isAscii(password)) throw InputError(std::string(what) + ": non-ASCII byte in MSSQL ASCII domain");
}
bool verifyCandidate(Algorithm alg, const Target& t, const Bytes& password) {
    return digestForCandidate(alg, password, t.salt) == t.digest;
}
bool verifyUppercaseForm(Algorithm alg, const Target& t, const Bytes& password) {
    if (!t.upperDigest) return false;
    return digestForCandidate(alg, asciiUpperBytes(password), t.salt) == *t.upperDigest;
}
std::size_t countAlphabetic(const Bytes& b) {
    std::size_t n = 0;
    for (auto c : b) if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) ++n;
    return n;
}

} // namespace cudadb
