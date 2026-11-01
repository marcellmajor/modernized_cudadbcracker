#include "cudadb/types.hpp"
#include <cctype>

namespace cudadb {

std::string hexEncode(const std::uint8_t* data, std::size_t n, bool upper) {
    const char* d = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    std::string s; s.resize(n * 2);
    for (std::size_t i = 0; i < n; ++i) { s[2*i] = d[data[i] >> 4]; s[2*i+1] = d[data[i] & 15]; }
    return s;
}
static int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
Bytes hexDecode(const std::string& hex, const char* what) {
    if (hex.size() % 2 != 0) throw InputError(std::string(what) + ": odd-length hex string");
    Bytes out; out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        int hi = hexVal(hex[i]), lo = hexVal(hex[i+1]);
        if (hi < 0 || lo < 0) throw InputError(std::string(what) + ": invalid hex character");
        out.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
    }
    return out;
}
std::array<std::uint32_t, 5> digestWords(const Digest& d) {
    std::array<std::uint32_t, 5> w{};
    for (int i = 0; i < 5; ++i)
        w[i] = (std::uint32_t(d[4*i]) << 24) | (std::uint32_t(d[4*i+1]) << 16) |
               (std::uint32_t(d[4*i+2]) << 8) | std::uint32_t(d[4*i+3]);
    return w;
}
Digest digestFromWords(const std::array<std::uint32_t, 5>& w) {
    Digest d{};
    for (int i = 0; i < 5; ++i) {
        d[4*i] = std::uint8_t(w[i] >> 24); d[4*i+1] = std::uint8_t(w[i] >> 16);
        d[4*i+2] = std::uint8_t(w[i] >> 8); d[4*i+3] = std::uint8_t(w[i]);
    }
    return d;
}
char asciiUpperChar(char c) { return (c >= 'a' && c <= 'z') ? char(c - 'a' + 'A') : c; }
std::uint8_t asciiUpperByte(std::uint8_t c) { return std::uint8_t(asciiUpperChar(char(c))); }
Bytes asciiUpperBytes(const Bytes& in) { Bytes o = in; for (auto& c : o) c = asciiUpperByte(c); return o; }
bool isAscii(const Bytes& b) { for (auto c : b) if (c > 0x7f) return false; return true; }
std::string displayEscape(const std::uint8_t* data, std::size_t n) {
    std::string s;
    const char* d = "0123456789abcdef";
    for (std::size_t i = 0; i < n; ++i) {
        std::uint8_t c = data[i];
        if (c >= 0x20 && c <= 0x7e && c != '"' && c != '\\') s.push_back(char(c));
        else if (c == '"') s += "\\\"";
        else if (c == '\\') s += "\\\\";
        else { s += "\\x"; s.push_back(d[c >> 4]); s.push_back(d[c & 15]); }
    }
    return s;
}

} // namespace cudadb
