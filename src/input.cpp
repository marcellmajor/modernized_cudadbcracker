#include "cudadb/input.hpp"
#include "cudadb/algorithms.hpp"
#include <fstream>
#include <set>
#include <sstream>

namespace cudadb {
namespace {
std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t')) --b;
    return s.substr(a, b - a);
}
std::vector<std::string> readLines(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw InputError("cannot open file: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // CRLF is a line ending
        lines.push_back(line);
    }
    return lines;
}
Digest digestFromHex(const std::string& hex, const char* what) {
    Bytes b = hexDecode(hex, what);
    if (b.size() != 20) throw InputError(std::string(what) + ": digest must be 20 bytes");
    Digest d{}; std::copy(b.begin(), b.end(), d.begin());
    return d;
}
} // namespace

std::vector<Target> parseTargetFile(const std::string& path, Algorithm alg) {
    auto lines = readLines(path);
    std::vector<Target> out;
    std::set<std::string> labels;
    for (std::size_t li = 0; li < lines.size(); ++li) {
        std::string where = path + ":" + std::to_string(li + 1) + ": ";
        std::string t = trim(lines[li]);
        if (t.empty() || t[0] == '#') continue;
        std::size_t colon = t.find(':');
        if (colon == std::string::npos) throw InputError(where + "missing ':' separator");
        std::string label = trim(t.substr(0, colon));
        std::string ver = trim(t.substr(colon + 1));
        if (label.empty()) throw InputError(where + "empty account label");
        if (!labels.insert(label).second) throw InputError(where + "duplicate account label '" + label + "'");
        Target tgt; tgt.label = label;
        if (alg == Algorithm::Oracle11g) {
            if (ver.rfind("S:", 0) == 0 || ver.rfind("s:", 0) == 0) ver = ver.substr(2);
            if (ver.size() != 60) throw InputError(where + "Oracle verifier must be 60 hex characters");
            tgt.digest = digestFromHex(ver.substr(0, 40), "Oracle digest");
            tgt.salt = hexDecode(ver.substr(40, 20), "Oracle salt");
            tgt.kind = TargetKind::Single;
        } else {
            if (!(ver.rfind("0x0100", 0) == 0 || ver.rfind("0X0100", 0) == 0))
                throw InputError(where + "MSSQL verifier must start with 0x0100 (other versions rejected)");
            std::string rest = ver.substr(6);
            if (rest.size() != 48 && rest.size() != 88)
                throw InputError(where + "MSSQL verifier must be 54 or 94 characters in total");
            tgt.salt = hexDecode(rest.substr(0, 8), "MSSQL salt");
            // Layout: salt || case-sensitive digest || uppercase digest (dual form).
            Digest first = digestFromHex(rest.substr(8, 40), "MSSQL case-sensitive digest");
            if (rest.size() == 48) {
                tgt.digest = first;
                tgt.kind = TargetKind::Single;
            } else {
                tgt.digest = first;
                tgt.upperDigest = digestFromHex(rest.substr(48, 40), "MSSQL uppercase digest");
                tgt.kind = TargetKind::DualUppercase;
            }
        }
        out.push_back(tgt);
    }
    return out;
}

std::vector<std::pair<std::string, std::string>> parsePasswordFile(const std::string& path) {
    auto lines = readLines(path);
    std::vector<std::pair<std::string, std::string>> out;
    for (auto& raw : lines) {
        std::string t = trim(raw);
        if (t.empty()) continue;
        auto c = t.find(':');
        if (c == std::string::npos) continue;
        out.emplace_back(trim(t.substr(0, c)), t.substr(c + 1));
    }
    return out;
}

Bytes dedupPreserveFirst(Bytes in) {
    Bytes out; bool seen[256] = {};
    for (auto c : in) if (!seen[c]) { seen[c] = true; out.push_back(c); }
    return out;
}
Bytes uppercaseAlphabet(Bytes a) {
    for (auto& c : a) c = asciiUpperByte(c);
    return dedupPreserveFirst(std::move(a));
}
Bytes alphabetFromClasses(const std::string& classes) {
    if (classes.empty()) throw InputError("empty charset classes");
    static const char* kLower = "abcdefghijklmnopqrstuvwxyz";
    static const char* kUpper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static const char* kNum = "0123456789";
    // Original charset_spec, original order (spec section 6.2).
    static const std::uint8_t kSpec[] = {0x21,0x40,0x23,0x24,0x25,0x5e,0x26,0x2a,0x28,0x29,0x2d,0x5f,
                                         0x2b,0x3d,0x7e,0x60,0x5b,0x5d,0x7b,0x7d,0x7c,0x5c,0x3a,0x3b,
                                         0x22,0x27,0x3c,0x3e,0x2c,0x2e,0x3f,0x2f};
    Bytes out;
    auto appendStr = [&](const char* s) { while (*s) out.push_back(std::uint8_t(*s++)); };
    for (char c : classes) {
        switch (c) {
            case 'u': appendStr(kUpper); break;
            case 'l': appendStr(kLower); break;
            case 'n': appendStr(kNum); break;
            case 's': out.insert(out.end(), std::begin(kSpec), std::end(kSpec)); break;
            default: throw InputError(std::string("unknown charset class '") + c + "' (expected u, l, n or s)");
        }
    }
    out = dedupPreserveFirst(std::move(out));
    if (out.empty()) throw InputError("empty alphabet");
    return out;
}
Bytes alphabetFromHex(const std::string& hex) {
    Bytes out = dedupPreserveFirst(hexDecode(hex, "charset"));
    if (out.empty()) throw InputError("empty alphabet");
    return out;
}
Bytes effectiveAlphabet(Algorithm alg, MssqlMode mode, const Target& t, const Bytes* userAlphabet) {
    Bytes a;
    if (userAlphabet) a = *userAlphabet;
    else if (alg == Algorithm::Oracle11g) a = alphabetFromClasses("uln");
    else if (t.kind == TargetKind::DualUppercase && mode != MssqlMode::Direct) a = alphabetFromClasses("un");
    else a = alphabetFromClasses("uln");
    bool upperMode = (alg == Algorithm::Mssql && t.kind == TargetKind::DualUppercase && mode != MssqlMode::Direct);
    if (upperMode) a = uppercaseAlphabet(a);
    if (alg == Algorithm::Mssql) {
        for (auto c : a) if (c > 0x7f) throw InputError("MSSQL alphabet contains a non-ASCII byte");
    }
    if (a.empty()) throw InputError("empty alphabet");
    if (a.size() > 256) throw InputError("alphabet larger than 256 bytes");
    return a;
}
std::string targetFingerprint(Algorithm alg, const Target& t) {
    std::string canon = std::string(toString(alg)) + "|" + hexEncode(t.salt) + "|" + hexEncode(t.digest);
    if (t.upperDigest) canon += "|" + hexEncode(*t.upperDigest);
    Digest d = sha1(Bytes(canon.begin(), canon.end()));
    return hexEncode(d);
}

} // namespace cudadb
