#include "cudadb/checkpoint.hpp"
#include "cudadb/input.hpp"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

namespace cudadb {
namespace {
// ---- minimal JSON (objects, arrays, strings, numbers, bool, null) ----
struct Json {
    enum class T { Null, Bool, Num, Str, Arr, Obj } t = T::Null;
    bool b = false; double num = 0; std::string str;
    std::vector<Json> arr; std::map<std::string, Json> obj;
    static Json s(std::string v) { Json j; j.t = T::Str; j.str = std::move(v); return j; }
    static Json n(double v) { Json j; j.t = T::Num; j.num = v; return j; }
    static Json a(std::vector<Json> v) { Json j; j.t = T::Arr; j.arr = std::move(v); return j; }
    static Json o(std::map<std::string, Json> v) { Json j; j.t = T::Obj; j.obj = std::move(v); return j; }
    const Json& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw InputError("checkpoint JSON: missing key '" + k + "'");
        return it->second;
    }
    const Json* find(const std::string& k) const { auto it = obj.find(k); return it == obj.end() ? nullptr : &it->second; }
    std::string asStr(const std::string& def = "") const { return t == T::Str ? str : def; }
    long long asInt(long long def = 0) const { return t == T::Num ? (long long)num : def; }
};
std::string jsonEscape(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); o += buf; }
                else o += c;
        }
    }
    return o + "\"";
}
void writeJson(const Json& j, std::string& out) {
    switch (j.t) {
        case Json::T::Null: out += "null"; break;
        case Json::T::Bool: out += j.b ? "true" : "false"; break;
        case Json::T::Num: { char buf[32]; std::snprintf(buf, sizeof buf, "%lld", (long long)j.num); out += buf; break; }
        case Json::T::Str: out += jsonEscape(j.str); break;
        case Json::T::Arr: out += "["; for (std::size_t i = 0; i < j.arr.size(); ++i) { if (i) out += ","; writeJson(j.arr[i], out); } out += "]"; break;
        case Json::T::Obj: out += "{"; bool first = true; for (auto& kv : j.obj) { if (!first) out += ","; first = false; out += jsonEscape(kv.first); out += ":"; writeJson(kv.second, out); } out += "}"; break;
    }
}
struct Parser {
    const char* p; const char* end;
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    char peek() { ws(); if (p >= end) throw InputError("checkpoint JSON: unexpected end"); return *p; }
    void expect(char c) { if (peek() != c) throw InputError(std::string("checkpoint JSON: expected '") + c + "'"); ++p; }
    Json parse() { Json j = value(); ws(); if (p != end) throw InputError("checkpoint JSON: trailing data"); return j; }
    Json value() {
        char c = peek();
        if (c == '{') { ++p; Json j = Json::o({}); if (peek() == '}') { ++p; return j; }
            while (true) { std::string k = string(); expect(':'); j.obj[k] = value(); if (peek() == ',') { ++p; continue; } expect('}'); return j; } }
        if (c == '[') { ++p; Json j = Json::a({}); if (peek() == ']') { ++p; return j; }
            while (true) { j.arr.push_back(value()); if (peek() == ',') { ++p; continue; } expect(']'); return j; } }
        if (c == '"') { Json j; j.t = Json::T::Str; j.str = string(); return j; }
        if (c == 't') { literal("true"); Json j; j.t = Json::T::Bool; j.b = true; return j; }
        if (c == 'f') { literal("false"); Json j; j.t = Json::T::Bool; return j; }
        if (c == 'n') { literal("null"); return Json{}; }
        return number();
    }
    void literal(const char* s) { while (*s) { if (p >= end || *p != *s) throw InputError("checkpoint JSON: bad literal"); ++p; ++s; } }
    Json number() {
        const char* start = p;
        if (p < end && (*p == '-')) ++p;
        while (p < end && (std::isdigit((unsigned char)*p) || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
        if (start == p) throw InputError("checkpoint JSON: bad value");
        Json j; j.t = Json::T::Num; j.num = std::stod(std::string(start, p)); return j;
    }
    std::string string() {
        expect('"'); std::string out;
        while (true) {
            if (p >= end) throw InputError("checkpoint JSON: unterminated string");
            char c = *p++;
            if (c == '"') return out;
            if (c == '\\') {
                if (p >= end) throw InputError("checkpoint JSON: bad escape");
                char e = *p++;
                switch (e) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
                    case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                    case 'u': {
                        if (end - p < 4) throw InputError("checkpoint JSON: bad \\u escape");
                        unsigned v = 0; for (int i = 0; i < 4; ++i) { char h = *p++; v <<= 4; v |= (h >= '0' && h <= '9') ? unsigned(h - '0') : (h >= 'a' && h <= 'f') ? unsigned(h - 'a' + 10) : (h >= 'A' && h <= 'F') ? unsigned(h - 'A' + 10) : 0; }
                        if (v < 0x80) out += char(v);
                        else if (v < 0x800) { out += char(0xC0 | (v >> 6)); out += char(0x80 | (v & 63)); }
                        else { out += char(0xE0 | (v >> 12)); out += char(0x80 | ((v >> 6) & 63)); out += char(0x80 | (v & 63)); }
                        break; }
                    default: throw InputError("checkpoint JSON: unknown escape");
                }
            } else out += c;
        }
    }
};
Json targetJson(const Target& t, Algorithm alg, const Bytes& alpha) {
    std::map<std::string, Json> o;
    o["label"] = Json::s(t.label);
    o["fingerprint"] = Json::s(targetFingerprint(alg, t));
    o["salt_hex"] = Json::s(hexEncode(t.salt));
    o["digest_hex"] = Json::s(hexEncode(t.digest));
    if (t.upperDigest) o["upper_digest_hex"] = Json::s(hexEncode(*t.upperDigest));
    o["kind"] = Json::s(t.kind == TargetKind::DualUppercase ? "dual" : "single");
    o["alphabet_hex"] = Json::s(hexEncode(alpha));
    return Json::o(o);
}
Target targetFromJson(const Json& j, Algorithm alg, Bytes& alphaOut) {
    Target t; t.label = j.at("label").asStr();
    t.salt = hexDecode(j.at("salt_hex").asStr(), "checkpoint salt");
    auto dig = hexDecode(j.at("digest_hex").asStr(), "checkpoint digest");
    if (dig.size() != 20) throw InputError("checkpoint digest must be 20 bytes");
    std::copy(dig.begin(), dig.end(), t.digest.begin());
    if (auto* u = j.find("upper_digest_hex")) {
        auto ub = hexDecode(u->asStr(), "checkpoint upper digest");
        if (ub.size() != 20) throw InputError("checkpoint upper digest must be 20 bytes");
        Digest d{}; std::copy(ub.begin(), ub.end(), d.begin()); t.upperDigest = d; t.kind = TargetKind::DualUppercase;
    }
    alphaOut = hexDecode(j.at("alphabet_hex").asStr(), "checkpoint alphabet");
    std::string expect = targetFingerprint(alg, t);
    if (j.at("fingerprint").asStr() != expect) throw InputError("checkpoint target fingerprint mismatch for '" + t.label + "'");
    return t;
}
} // namespace

std::string checkpointToJson(const Checkpoint& cp) {
    std::map<std::string, Json> root;
    root["schema"] = Json::n(kCheckpointSchema);
    root["algorithm"] = Json::s(toString(cp.algorithm));
    root["mssql_mode"] = Json::s(toString(cp.mssqlMode));
    root["min_length"] = Json::n(cp.minLength);
    root["max_length"] = Json::n(cp.maxLength);
    root["fixed_length"] = Json::n(cp.fixedLength ? 1 : 0);
    root["start_rank"] = Json::s(cp.startRank);
    root["range_count"] = Json::s(cp.rangeCount);
    root["case_budget"] = Json::s(std::to_string(cp.caseBudget));
    root["build_id"] = Json::s(cp.buildId);
    root["tune_note"] = Json::s(cp.tuneNote);
    std::vector<Json> targets, states;
    for (std::size_t i = 0; i < cp.targets.size(); ++i) {
        targets.push_back(targetJson(cp.targets[i], cp.algorithm, cp.alphabets[i]));
        const auto& st = cp.states[i];
        std::map<std::string, Json> so;
        so["status"] = Json::s(st.status); so["length"] = Json::n(st.length); so["rank"] = Json::s(st.rank);
        so["password_hex"] = Json::s(st.passwordHex); so["pending_upper_hex"] = Json::s(st.pendingUpperHex);
        so["case_next_assignment"] = Json::s(st.caseNextAssignment);
        states.push_back(Json::o(so));
    }
    root["targets"] = Json::a(targets);
    root["states"] = Json::a(states);
    std::string out; writeJson(Json::o(root), out); out += "\n";
    return out;
}
Checkpoint checkpointFromJson(const std::string& text) {
    Parser p{text.data(), text.data() + text.size()};
    Json root = p.parse();
    if (root.at("schema").asInt(-1) != kCheckpointSchema) throw InputError("unsupported checkpoint schema");
    Checkpoint cp;
    std::string alg = root.at("algorithm").asStr();
    if (alg == "ora11g") cp.algorithm = Algorithm::Oracle11g;
    else if (alg == "mssql") cp.algorithm = Algorithm::Mssql;
    else throw InputError("checkpoint: unknown algorithm '" + alg + "'");
    std::string mode = root.at("mssql_mode").asStr("auto");
    cp.mssqlMode = mode == "uppercase" ? MssqlMode::Uppercase : mode == "direct" ? MssqlMode::Direct : MssqlMode::Auto;
    cp.minLength = std::uint32_t(root.at("min_length").asInt(1));
    cp.maxLength = std::uint32_t(root.at("max_length").asInt(8));
    cp.fixedLength = root.at("fixed_length").asInt(0) != 0;
    cp.startRank = root.at("start_rank").asStr("0");
    cp.rangeCount = root.at("range_count").asStr("");
    cp.caseBudget = std::stoull(root.at("case_budget").asStr("1048576"));
    cp.buildId = root.at("build_id").asStr("");
    cp.tuneNote = root.at("tune_note").asStr("");
    const auto& tj = root.at("targets").arr;
    const auto& sj = root.at("states").arr;
    if (tj.size() != sj.size()) throw InputError("checkpoint: targets/states size mismatch");
    for (std::size_t i = 0; i < tj.size(); ++i) {
        Bytes alpha; Target t = targetFromJson(tj[i], cp.algorithm, alpha);
        std::size_t wantSalt = cp.algorithm == Algorithm::Oracle11g ? 10 : 4;
        if (t.salt.size() != wantSalt) throw InputError("checkpoint: wrong salt size for '" + t.label + "'");
        cp.targets.push_back(t); cp.alphabets.push_back(alpha);
        TargetState st;
        st.status = sj[i].at("status").asStr("searching");
        st.length = std::uint32_t(sj[i].at("length").asInt(1));
        st.rank = sj[i].at("rank").asStr("0");
        st.passwordHex = sj[i].at("password_hex").asStr("");
        st.pendingUpperHex = sj[i].at("pending_upper_hex").asStr("");
        st.caseNextAssignment = sj[i].at("case_next_assignment").asStr("0");
        cp.states.push_back(st);
    }
    return cp;
}
void saveCheckpoint(const std::string& path, const Checkpoint& cp) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw InputError("cannot write checkpoint: " + tmp);
        out << checkpointToJson(cp);
        out.flush();
        if (!out) throw InputError("cannot write checkpoint: " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw InputError("cannot replace checkpoint: " + path);
}
Checkpoint loadCheckpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw InputError("cannot open checkpoint: " + path);
    std::ostringstream ss; ss << in.rdbuf();
    return checkpointFromJson(ss.str());
}
std::string resultRecordJson(const Target& t, Algorithm alg, const std::string& mode,
                             const TargetState& st, const std::string& verification) {
    std::map<std::string, Json> o;
    o["schema"] = Json::n(kResultSchema);
    o["target_id"] = Json::s(t.label);
    o["algorithm"] = Json::s(toString(alg));
    o["mode"] = Json::s(mode);
    o["status"] = Json::s(st.status);
    o["verification"] = Json::s(verification);
    if (st.status == "found") {
        Bytes pw = hexDecode(st.passwordHex, "password");
        o["password_hex"] = Json::s(st.passwordHex);
        o["password_display"] = Json::s(displayEscape(pw));
        o["candidate_length"] = Json::n((long long)pw.size());
    }
    if (!st.pendingUpperHex.empty()) o["pending_upper_hex"] = Json::s(st.pendingUpperHex);
    o["length"] = Json::n(st.length);
    o["rank"] = Json::s(st.rank);
    std::string out; writeJson(Json::o(o), out);
    return out;
}

} // namespace cudadb
