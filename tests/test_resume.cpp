#include "cudadb/checkpoint.hpp"
#include "cudadb/input.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace cudadb;
static int failures = 0;
#define CHECK(cond) do { if(!(cond)) { std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; ++failures; } } while(0)
#define CHECK_THROWS(expr) do { bool t=false; try { expr; } catch(const InputError&) { t=true; } if(!t) { std::cerr << "FAIL (no throw) line " << __LINE__ << ": " #expr "\n"; ++failures; } } while(0)

int main() {
    const std::string path = "/tmp/cudadb_test_checkpoint.json";
    std::remove(path.c_str());
    Checkpoint cp;
    cp.algorithm = Algorithm::Mssql; cp.mssqlMode = MssqlMode::Auto;
    cp.minLength = 1; cp.maxLength = 4;
    Target t; t.label = "user \"quoted\"";
    t.salt = hexDecode("01020304", "s");
    t.digest.fill(7);
    cp.targets.push_back(t);
    cp.alphabets.push_back(alphabetFromClasses("un"));
    TargetState st; st.status = "searching"; st.length = 3; st.rank = "12345";
    cp.states.push_back(st);
    saveCheckpoint(path, cp);
    Checkpoint back = loadCheckpoint(path);
    CHECK(back.targets.size() == 1);
    CHECK(back.targets[0].label == t.label);
    CHECK(back.states[0].rank == "12345");
    CHECK(back.states[0].length == 3);
    CHECK(back.alphabets[0] == cp.alphabets[0]);
    // JSON round trip through string form.
    Checkpoint back2 = checkpointFromJson(checkpointToJson(cp));
    CHECK(back2.targets[0].salt == t.salt);
    // Corrupted checkpoint rejected.
    { std::ofstream o("/tmp/cudadb_bad.json"); o << "{ not json"; }
    CHECK_THROWS(loadCheckpoint("/tmp/cudadb_bad.json"));
    // Fingerprint tamper rejected.
    std::string js = checkpointToJson(cp);
    auto pos = js.find("digest_hex");
    js[pos + 14] = (js[pos + 14] == '0' ? '1' : '0');
    CHECK_THROWS(checkpointFromJson(js));
    // Schema reject.
    CHECK_THROWS(checkpointFromJson("{\"schema\":99}"));
    std::remove(path.c_str());
    if (failures) { std::cerr << failures << " failures\n"; return 1; }
    std::cout << "test_resume passed\n";
    return 0;
}
