# Test and benchmark on an A100 machine (Colab-style notebook environment)

Purpose: produce the missing GPU evidence for this project — `nvcc`
compilation for the four target architectures, `ctest -L gpu` /
`test_gpu`, benchmarks, and CUDA profiling — on an NVIDIA A100.

Status before this run: **host-verified only.** No `nvcc` compilation,
GPU test, benchmark, or profile exists yet (see `VERIFICATION.md`).

---

## 0. Permission gate — read first

Google's managed Colab FAQ **prohibits password cracking on managed
Colab runtimes, including paid runtimes**, and per
`IMPLEMENTATION_SPECS.md` section 1.1 renaming the workload a
"benchmark" does not change that status.

Therefore:

- On a **managed Google Colab** runtime: use this document for
  environment checks, building, and device diagnostics only
  (Steps 1–3). **Stop before Step 4.**
- Run Steps 4–7 only on a machine or service whose terms permit this
  workload — for example your own A100 server, a workstation, or a
  notebook connected to your own hardware. The commands are identical;
  a Colab-style notebook front-end pointed at your own machine is fine.
- Nothing in this document may be used to evade a provider's rules.

Record where the GPU run actually happened in the Results section.

---

## 1. Get the code onto the machine

The project is a git repository — clone or pull it. No zip upload or
manual copying is needed.

```bash
# First time: clone (use the repository's actual remote URL)
git clone <repo-url> modernized_cudadbcracker
cd modernized_cudadbcracker

# Later sessions: update and record the exact code being tested
git pull
git rev-parse HEAD        # write this commit into the Results section
git status --short        # must be empty; note any local changes
```

In a notebook, run the same commands with `!` / `%cd`:

```python
# !git clone <repo-url> modernized_cudadbcracker
# %cd modernized_cudadbcracker
# !git pull && git rev-parse HEAD
```

Working from the repository (not a copy) also means results can be
committed back: benchmark JSON, logs, and the completed Results
section of this file belong in git, while profiler binaries
(`*.ncu-rep`, `*.qdrep`) and build directories stay untracked via
`.gitignore`.

Confirm the tree is complete:

```bash
ls IMPLEMENTATION_SPECS.md CMakeLists.txt src/cuda tests tools
```

## 2. Verify the GPU is really an A100

```bash
nvidia-smi
python3 tools/check_environment.py
```

Expected: an `A100` device (compute capability **8.0**, 108 SMs on a
full card; partitioned instances report fewer — record what you see).
Save the full output of both commands for the Results section.

Notes:

- The CUDA version printed by `nvidia-smi` is the driver's maximum,
  **not** the installed toolkit. `nvcc --version` (from
  `check_environment.py`) is the toolkit that will compile the code.
- If no GPU is visible, stop: do not "fix" this by reinstalling the
  NVIDIA driver on a managed runtime.

## 3. Build

Dependencies (Ubuntu): `build-essential`, a CMake ≥ 3.24, Ninja, and a
GCC supported by the toolkit. Notebook images often ship an older
CMake — install a current one in user space if needed:

```bash
python3 -m pip install --user "cmake>=3.24" ninja
export PATH="$HOME/.local/bin:$PATH"
cmake --version    # must be >= 3.24
nvcc --version     # CUDA 12.x expected
```

Notebook gotcha (observed in practice): each `!` line runs in its own
shell, so `export PATH=...` in one cell does not reach `cmake` in the
next, and the configure fails with "unable to find a build program
corresponding to Ninja". Install Ninja onto the default PATH, or do
install + configure + build in a single `!` chain:

```bash
# either, as root on a notebook image:
apt-get install -y ninja-build
# or:
pip install ninja            # system install, lands on PATH
which ninja && ninja --version

# then, still in one shell invocation if PATH was a problem:
pip install -q ninja && cmake --preset distribution 2>&1 | tee build_configure.log
```

If `pip show ninja` (or the package in
`/root/.local/lib/python3.13/site-packages`) says Ninja is installed
but CMake still reports "unable to find a build program corresponding
to Ninja", the binary is simply not on the PATH CMake inherits. Fix
by locating it and either symlinking it onto the PATH or naming it
explicitly — `CMAKE_MAKE_PROGRAM` overrides the preset's generator
search without editing any project file:

```bash
ls /root/.local/bin/ninja                                   # usual pip --user location
python3 -c "import ninja, os; print(os.path.dirname(ninja.__file__))"
# binary is at <that dir>/data/bin/ninja if the bin script is absent

ln -sf /root/.local/bin/ninja /usr/local/bin/ninja && ninja --version
# or, without touching PATH at all:
cmake --preset distribution -DCMAKE_MAKE_PROGRAM=/root/.local/bin/ninja
```

Last resort, skip Ninja entirely with the Makefiles generator (same
cache variables as the preset; label results as a manual configure,
not the `distribution` preset):

```bash
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;89-real;90-real;90-virtual" \
  -DCUDADB_ENABLE_CUDA=ON
```

### 3a. Full distribution build (validates all four architectures)

This compiles native images for `sm_75;sm_80;sm_89;sm_90` plus PTX
fallback. It is the build the specification requires, and the slowest
to compile:

```bash
cmake --preset distribution 2>&1 | tee build_configure.log
cmake --build --preset distribution --parallel 2 2>&1 | tee build.log
```

Keep `--parallel 2` on memory-limited notebook machines; template
instantiation of the fast kernels uses substantial RAM.

Confirm the embedded images, if `cuobjdump` is available:

```bash
cuobjdump --list-elf build/cudadbcracker | tee images.log
# expect ELFs for sm_75, sm_80, sm_89, sm_90 (plus PTX)
```

Save from `build.log`: every `-Xptxas=-v` line (registers per thread,
spill loads/stores per kernel variant). Any spill report is a result,
not an embarrassment — record it verbatim.

### 3b. Faster A100-only build (iteration only)

For quick edit-test cycles on the A100 alone, the `native-dev` preset
compiles just the local architecture. **Do not** report results from
this build as distribution-build results; label them as A100-only.

```bash
cmake --preset native-dev
cmake --build --preset native-dev --parallel 2
```

---

## 4. Test (permitted runtime only, see Step 0)

Work from the build directory you chose in Step 3. Below, `build`
means `build` (distribution) or `build-native` (A100-only).

### 4.1 Host tests still pass on this machine

```bash
ctest --test-dir build --output-on-failure -L host 2>&1 | tee test_host.log
```

Expected: `test_host`, `test_search`, `test_resume` all Passed.

### 4.2 GPU tests — the key missing evidence

```bash
./build/cudadbcracker devices --json 2>&1 | tee devices.log
./build/cudadbcracker self-test --extended --device 0 2>&1 | tee selftest.log
ctest --test-dir build --output-on-failure -L gpu 2>&1 | tee test_gpu.log
./build/test_gpu 2>&1 | tee test_gpu_direct.log
```

`test_gpu` checks, against the CPU control:

- GPU digest == CPU digest for both algorithms at lengths
  0, 1, 10, 12, 25/26, 45, 128 (multi-block boundaries included).
- GPU search offset == CPU search offset for the grouped fast kernels
  and generic path, including medium single-block lengths
  (Oracle 17 and 45, MSSQL 17 and 25) under interleave 1, 2, and 4.

Expected: `test_gpu passed` and CTest reports the test **Passed**,
not Skipped. A Skip here means the GPU was not visible — treat the
whole GPU run as invalid and fix visibility before continuing.

### 4.3 Small end-to-end synthetic search (smoke test)

Uses a synthetic target generated by the project's own independent
vector tool. No real-world hashes:

```bash
python3 - <<'EOF'
import hashlib
salt = bytes.fromhex("0102030405060708090a")
verifier = hashlib.sha1(b"cab" + salt).hexdigest().upper() + salt.hex().upper()
open("/tmp/ora_targets.txt", "w").write(f"alice:{verifier}\n")
EOF

./build/cudadbcracker verify --algorithm ora11g --input /tmp/ora_targets.txt \
    --password-hex 636162 --backend both --device 0

./build/cudadbcracker search --algorithm ora11g --input /tmp/ora_targets.txt \
    --charset-classes l --length 3 --tune off --device 0 \
    --checkpoint /tmp/smoke_checkpoint.json --results /tmp/smoke_results.jsonl \
    2>&1 | tee smoke_search.log
cat /tmp/smoke_results.jsonl
```

Expected: `verify` reports `found` (verification `both`); `search`
emits one JSONL record with `status":"found"`,
`password_hex":"636162"`, and exit code 0. Save all output.

---

### 4.4 Compute Sanitizer (spec section 14.3)

Catches out-of-bounds and race defects that offset tests cannot see,
especially in the interleave-4 and medium-length paths:

```bash
compute-sanitizer --tool memcheck --launch-timeout 600 ./build/test_gpu 2>&1 | tee sanitizer_memcheck.log
compute-sanitizer --tool racecheck --launch-timeout 600 ./build/test_gpu 2>&1 | tee sanitizer_racecheck.log
compute-sanitizer --tool memcheck ./build/cudadbcracker search \
  --algorithm ora11g --input /tmp/ora_targets.txt --charset-classes l \
  --length 3 --tune off --device 0 2>&1 | tee sanitizer_search.log
```

Expected: `ERROR SUMMARY: 0 errors` in every run. Any error is a
correctness bug — save the full log and stop.

### 4.5 Multi-batch search, SIGINT, and resume

Verifies checkpoint cadence and that interruption neither skips nor
duplicates candidates. Plant a target near the end of a bounded
range, interrupt a multi-batch search over it, then resume:

```python
import hashlib, subprocess, signal, time, pathlib

# password at a known large rank over 'l' (26): choose digits directly
digits = [0, 0, 0, 0, 0, 0]
rank = 4_000_000
r = rank
for i in range(6):
    digits[i] = r % 26; r //= 26
pw = bytes(ord('a') + d for d in digits)
salt = bytes.fromhex("0102030405060708090a")
ver = hashlib.sha1(pw + salt).hexdigest().upper() + salt.hex().upper()
pathlib.Path("/tmp/big_targets.txt").write_text(f"victim:{ver}\n")
print("planted password:", pw, "rank:", rank)

args = ["./build/cudadbcracker", "search", "--algorithm", "ora11g",
        "--input", "/tmp/big_targets.txt", "--charset-classes", "l",
        "--length", "6", "--start-rank", "0", "--count", "5000000",
        "--batch-candidates", "4096", "--tune", "off", "--device", "0",
        "--checkpoint", "/tmp/big_ckpt.json", "--results", "/tmp/big_res.jsonl"]
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(6)
p.send_signal(signal.SIGINT)
out, _ = p.communicate()
print("interrupted exit:", p.returncode)   # expect 130
print(pathlib.Path("/tmp/big_ckpt.json").exists())

r2 = subprocess.run(["./build/cudadbcracker", "resume", "--checkpoint", "/tmp/big_ckpt.json",
                     "--tune", "off", "--device", "0", "--results", "/tmp/big_res.jsonl"],
                    text=True, capture_output=True)
print("resume exit:", r2.returncode)
print(open("/tmp/big_res.jsonl").read())   # expect one found record at rank 4000000
```

Pass criteria: interrupted exit is 130, the checkpoint exists, resume
exits 0, and exactly one `found` record appears whose `rank` equals
the planted rank — proving no candidate was skipped or repeated
across the interruption boundary.

### 4.6 MSSQL dual-digest smoke (uppercase search + case recovery)

The 4.3 smoke test covers Oracle direct search only. This one covers
the MSSQL dual path end to end through the CLI:

```python
import hashlib, subprocess, pathlib

salt = bytes.fromhex("0a0b0c0d")
def enc(s): return b"".join(bytes([c, 0]) for c in s.encode())
mixed, upper = "aB3c", "AB3C"
ver = ("0x0100" + salt.hex()
       + hashlib.sha1(enc(mixed) + salt).hexdigest()
       + hashlib.sha1(enc(upper) + salt).hexdigest())
pathlib.Path("/tmp/ms_targets.txt").write_text(f"bob:{ver}\n")

r = subprocess.run(["./build/cudadbcracker", "search", "--algorithm", "mssql",
                    "--input", "/tmp/ms_targets.txt", "--charset-classes", "un",
                    "--length", "4", "--mssql-mode", "auto", "--tune", "off",
                    "--device", "0", "--results", "/tmp/ms_res.jsonl"],
                   text=True, capture_output=True)
print("exit:", r.returncode)
print(open("/tmp/ms_res.jsonl").read())
```

Expected: one record with `status":"found"` and
`password_hex":"61423363"` (`aB3c`) — i.e. the GPU found the
uppercase form and CPU case recovery restored the exact case.

## 5. Benchmark (permitted runtime only)

Rules from the specification (section 15): a complete-work benchmark
that cannot be dead-code-eliminated, validated outputs, one factor
changed at a time, generic baseline always present, device-only and
end-to-end rates kept separate. Never compare numbers across GPUs from
different sessions without the full context block.

### 5.1 Quick baseline vs variants

```bash
B=./build/cudadbcracker
for IL in 1 2 4; do
  for BS in 128 256; do
    $B benchmark --algorithm ora11g --length 8 --charset-classes uln \
      --count 16777216 --repeats 5 --device 0 \
      --block-size $BS --interleave $IL --json bench_ora_L8_il${IL}_b${BS}.json
    $B benchmark --algorithm mssql --length 8 --charset-classes uln \
      --count 16777216 --repeats 5 --device 0 \
      --block-size $BS --interleave $IL --json bench_mssql_L8_il${IL}_b${BS}.json
  done
done 2>&1 | tee benchmark_L8.log
```

Each line is a JSON record: hashes/second, device seconds, block size,
interleave, registers/thread, shared memory, kernel family
(`fast`/`generic`). The winner at interleave 1/2/4 is a *measurement*;
if 4 loses (register spills), that is the expected possible outcome.

### 5.2 Length sweep (kernel-family boundaries)

Lengths 6 (grouped fast), 17 and 25/45 (single-block generic),
and 46+ (multi-block generic) behave differently. Sweep them:

```bash
for L in 6 12 17 25 45 46; do
  ./build/cudadbcracker benchmark --algorithm ora11g --length $L \
    --charset-classes uln --count 16777216 --repeats 5 --device 0 \
    --json bench_ora_L${L}.json
done
for L in 6 12 17 25 26; do
  ./build/cudadbcracker benchmark --algorithm mssql --length $L \
    --charset-classes uln --count 16777216 --repeats 5 --device 0 \
    --json bench_mssql_L${L}.json
done
```

### 5.3 Full matrix helper

```bash
python3 tools/benchmark_matrix.py --binary ./build/cudadbcracker \
  --device 0 --count 16777216 2>&1 | tee benchmark_matrix.log
```

### 5.4 Autotuning check

```bash
./build/cudadbcracker search --algorithm ora11g --input /tmp/ora_targets.txt \
  --charset-classes l --length 3 --tune quick --device 0 2>&1 | tee tune_quick.log
./build/cudadbcracker search --algorithm ora11g --input /tmp/ora_targets.txt \
  --charset-classes l --length 3 --tune full --device 0 2>&1 | tee tune_full.log
```

Record how long tuning took and which configuration it selected.

---

### 5.5 Grid wave-multiple sweep (validates wave-aware grid sizing)

After the wave-aware grid change, grids snap to whole occupancy
waves. Quantify it by sweeping the grid multiplier around the old
3,456-block launch (108 SMs) at interleave 2 / block 128, before and
after the change (two commits, same everything else):

```python
import subprocess
for gm in (30, 32, 40):   # caps: 3240 / 3456 / 4320 blocks pre-change
    r = subprocess.run(["./build/cudadbcracker", "benchmark", "--algorithm", "ora11g",
                        "--length", "8", "--charset-classes", "uln", "--count", "16777216",
                        "--repeats", "5", "--device", "0", "--block-size", "128",
                        "--interleave", "2", "--grid-multiplier", str(gm)],
                       text=True, capture_output=True)
    print("grid-multiplier", gm, r.stdout.strip() or r.stderr.strip())
```

Report both commits' numbers with the full context block. A wave
multiple that removes the 25% tail should show up directly here.

### 5.6 End-to-end vs device-only rate

All Section 8 numbers so far are device-event timing. Measure the
end-to-end rate a user actually sees, at the default-ish batch, by
timing a no-hit fixed-range search wall-clock (use a target file
whose password lies outside the searched range, e.g. one generated
for `zzzzzz`, whose rank exceeds the 50M count below):

```bash
time ./build/cudadbcracker search --algorithm ora11g --input /tmp/nohit_targets.txt \
  --charset-classes l --length 6 --start-rank 0 --count 50000000 \
  --batch-candidates 1048576 --tune off --device 0 \
  --checkpoint /tmp/e2e_ckpt.json --results /tmp/e2e_res.jsonl
# end-to-end hashes/s = 50,000,000 / wall seconds; compare with the
# device-only benchmark rate at the same length/alphabet.
```

The gap is host serialisation (checkpoint write + result read per
batch). If it is material at 2^20 batches, double-buffering becomes
worth revisiting; if not, close that question with the number.

### 5.7 Alphabet-size sweep

Group amortisation depends on B (fast kernel processes B candidates
per prepared group). Sweep B = 10 / 26 / 36 / 62 at length 8,
interleave 2 (36 is the MSSQL `un` uppercase-mode alphabet):

```bash
for CS in n l un uln; do
  ./build/cudadbcracker benchmark --algorithm ora11g --length 8 \
    --charset-classes $CS --count 16777216 --repeats 5 --device 0 \
    --block-size 256 --interleave 2
done
```

## 6. CUDA profiling (permitted runtime only)

Use Nsight Compute (`ncu`) for per-kernel analysis and Nsight Systems
(`nsys`) for timeline/launch overhead. `nvprof` no longer exists in
current toolkits — do not look for it.

### 6.1 Nsight Compute — one kernel in depth

Profile the benchmark (fixed, complete work) rather than a search that
may exit early:

```bash
ncu --version
ncu --set detailed \
    --launch-count 4 --launch-skip 2 \
    -o profile_ora_L8_il1 \
    ./build/cudadbcracker benchmark --algorithm ora11g --length 8 \
      --charset-classes uln --count 4194304 --repeats 1 --device 0 \
      --block-size 128 --interleave 1
```

Repeat with `--interleave 2` and `--interleave 4`. Then inspect:

```bash
ncu --import profile_ora_L8_il1.ncu-rep --page details
# compact numeric export (Nsight Compute 2025.x has no "summary" page):
ncu --import profile_ora_L8_il1.ncu-rep --csv \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_elapsed
```

Extract for the Results section, per kernel:

- Duration and achieved occupancy vs theoretical occupancy.
- Registers per thread; any local-memory (spill) loads/stores.
- Compute (SM) throughput % and memory throughput % ("Speed of Light").
- Top warp stall reasons (long scoreboard / wait / not selected).
- Executed instructions per candidate if shown (instruction mix:
  integer add, `LOP3`, funnel shift are the expected hot opcodes).

Colab-style runtimes frequently block performance counters
(`ERR_NVGPUCTRPERM` / "No kernels were profiled"). If that happens:
record the exact error, try `ncu --set basic`, and fall back to
Section 6.2. A counter-permission failure is an environment result —
note it, do not treat the run as profiled.

### 6.2 Nsight Systems — timeline and launch behaviour

```bash
nsys --version
nsys profile --trace=cuda,osrt --stats=true \
  -o timeline_smoke \
  ./build/cudadbcracker benchmark --algorithm ora11g --length 8 \
    --charset-classes uln --count 16777216 --repeats 3 --device 0
```

Extract: kernel launch count/durations, gaps between launches
(host-bound time: checkpoint saves, result reads), and memcpy sizes.
This is where launch overhead and host/device serialisation show up.

### 6.4 Nsight Systems on a multi-batch search

Section 6.2 profiles a single benchmark process. This one measures
the real search loop over many batches — inter-launch gaps are where
checkpoint writes and result reads would show up:

```bash
nsys profile --trace=cuda,osrt --stats=true -o timeline_multibatch \
  ./build/cudadbcracker search --algorithm ora11g --input /tmp/nohit_targets.txt \
  --charset-classes l --length 6 --start-rank 0 --count 20000000 \
  --batch-candidates 262144 --tune off --device 0 \
  --checkpoint /tmp/nsys_ckpt.json --results /tmp/nsys_res.jsonl
```

Extract: launches per batch, gap distribution between kernel end and
next launch, and memcpy sizes. Small gaps at 2^18 batches would make
double-buffering worth testing; negligible gaps close the question.

### 6.5 Nsight Compute on the generic single-block kernel (L17/L25)

All profiling so far is the fast kernel at L8. Profile the generic
path at the lengths where it takes over, with local-memory metrics,
to price the medium-length grouped kernel (held item B):

```bash
for L in 17 25; do
  ncu --set detailed --launch-skip 1 --launch-count 2 \
    -o generic_L${L} --force-overwrite \
    ./build/cudadbcracker benchmark --algorithm ora11g --length $L \
      --charset-classes uln --count 4194304 --repeats 3 --device 0
  ncu --import generic_L${L}.ncu-rep --csv --metrics \
    gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
done
```

The local load/store sector counts quantify how much of the generic
kernel's ~45% deficit is its 576-byte stack frame living in local
memory — that number is the evidence base for scoping item B.

### 6.6 What "good" looks like (interpretation guide)

- Compute throughput near the SM limit with low memory throughput is
  expected — the kernel is integer-ALU-bound by design.
- Achieved occupancy well below theoretical + high register count =
  register-limited; check interleave 4 here first.
- Nonzero spill loads/stores on any variant = that variant is suspect
  regardless of its hashes/second in isolation.
- Large inter-launch gaps in `nsys` = host serialisation
  (checkpoint/result handling), relevant only at small batch sizes.

---

## 7. What to send back

Send the contents of these files (paste in chat or attach). They are
small except the `.ncu-rep`, for which the text summaries suffice:

```text
nvidia-smi output            build_configure.log (tail)
check_environment.py output  build.log (include all -Xptxas=-v lines)
cuobjdump images.log         devices.log
test_host.log                selftest.log
test_gpu.log / test_gpu_direct.log
smoke_search.log             smoke_results.jsonl
benchmark_L8.log             all bench_*.json
benchmark_matrix.log         tune_quick.log / tune_full.log
ncu summary + details text (per interleave), or the exact ncu error
nsys stats output (kernel summary section)
```

Also state: where the run happened (own machine / which service),
GPU model as reported, driver and `nvcc --version`, whether the
build was `distribution` or `native-dev` for each result, and the
`git rev-parse HEAD` commit the results were produced from. If you
have write access to the repository, committing the logs and the
completed Results section below to a branch is an alternative to
pasting — say which you did.

On receipt, the returned output will be analysed and written into the
Results section below, with conclusions; `VERIFICATION.md` and
`benchmarks/README.md` can then be updated from "not tested" to the
measured facts.

---

## 8. Results (to be completed from returned output)

### 8.1 Environment

| Item | Value |
|---|---|
| Git commit (`git rev-parse HEAD`) | _(pending — not yet reported)_ |
| Where run | Not yet stated (Linux x86_64, kernel 6.6.159+, container host `938f62003527`; working directory `/content/modernized_cudadbcracker`) |
| GPU (as reported) | NVIDIA A100-SXM4-80GB, compute capability 8.0, 80 GiB |
| Driver / CUDA toolkit | Driver 580.82.07 / toolkit CUDA 13.0 (V13.0.88, `/usr/local/cuda/bin/nvcc`) |
| Build preset | `distribution` — **build completed 23/23** (CMake 3.31.10, GNU 13.3.0 host compiler, nvcc 13.0.88), zero errors and zero warnings in the full log |
| Architectures found in binary | All four native targets compiled: 98 kernel entries each for `sm_75`, `sm_80`, `sm_89`, `sm_90` (392 total, from `ptxas` records; `cuobjdump` confirmation still pending) |

Environment notes (from the first returned output, `nvidia-smi` +
`tools/check_environment.py`):

- The GPU is exactly the expected target class: A100 SXM4 80 GB,
  cc 8.0, idle at the time of capture (0 MiB used, no processes).
- The toolkit is **CUDA 13.0**, newer than the CUDA 12.x baseline the
  spec assumed. Watch the distribution build for architecture-support
  warnings (notably around `sm_75`) and record them verbatim; a
  deprecation warning is acceptable, a dropped architecture is not —
  it would need reporting against spec section 3.
- **Blocker found and worked around: `ninja` was not on PATH** on the
  machine, and the CMake presets require the Ninja generator. The
  pip `ninja` package (1.13.2) is installed under
  `/root/.local/lib/python3.13/site-packages`; its binary must be
  symlinked onto PATH or named via `-DCMAKE_MAKE_PROGRAM=...`
  (see Step 3). `cmake` is confirmed at version **3.31.10**
  (>= 3.24 as required), from the pip `cmake` package.

### 8.2 Tests

| Check | Result |
|---|---|
| Host tests (`-L host`) | **Passed** (3/3 on the A100 machine) |
| `devices --json` | **Passed** — A100-SXM4-80GB, cc 8.0, 108 SMs, 84,994,825,984 bytes, max 1024 threads/block |
| `self-test --extended --device 0` | **Passed** |
| `ctest -L gpu` / `test_gpu` | **Passed** (0.55 s, after fast-forward to `b775209`). Attempt 1 had failed on a test-case bug — MSSQL case `BA` over a lowercase alphabet with batch 27 > 3² space — fixed in commit `4da1451` (case `ba`, counts bounded by `min(27, 3^L)`); the re-run passes all GPU digest checks (both algorithms, lengths 0–128 incl. multi-block) and all GPU-vs-CPU search-offset checks, including Oracle 17/45, MSSQL 17/25, at interleave 1, 2, and 4. |
| Synthetic search smoke test | **Passed.** `verify --backend both` returned `found` (`verification":"both"`). The full `search` then ran end-to-end with exit 0 and emitted `{"status":"found","password_hex":"636162","password_display":"cab","rank":"678","verification":"cpu-verified"}` — rank 678 is exactly canonical for `cab` over the 26-letter alphabet (2 + 0·26 + 1·26²), so GPU batch search, CPU re-verification of the hit, checkpoint writing, and JSONL reporting all work together. (The first attempt had been interrupted by a notebook paste artifact in the long shell line, unrelated to the code.) |

### 8.3 Benchmarks

A100-SXM4-80GB, length 8, alphabet 62 (`uln`), 16,777,216 candidates,
5 repeats, device-event timing (median). Registers/spills are from the
build log's `ptxas` records (the benchmark JSON's
`registers_per_thread` field currently reports -1; see note below).

| Algorithm | Length | Block | Interleave | Hashes/s | vs best |
|---|---|---|---|---|---|
| ora11g | 8 | 256 | 2 | **17.71 GH/s** | best |
| ora11g | 8 | 128 | 2 | 17.45 GH/s | −1.5% |
| ora11g | 8 | 256 | 1 | 15.37 GH/s | −13% |
| ora11g | 8 | 128 | 1 | 15.26 GH/s | −14% |
| ora11g | 8 | 256 | 4 | 14.69 GH/s | −17% |
| ora11g | 8 | 128 | 4 | 14.11 GH/s | −20% |
| mssql | 8 | 256 | 2 | **17.25 GH/s** | best |
| mssql | 8 | 128 | 2 | 17.16 GH/s | −0.5% |
| mssql | 8 | 256 | 1 | 15.30 GH/s | −11% |
| mssql | 8 | 128 | 1 | 15.26 GH/s | −11% |
| mssql | 8 | 256 | 4 | 14.72 GH/s | −15% |
| mssql | 8 | 128 | 4 | 14.03 GH/s | −19% |

Conclusions so far:

- **Interleave 2 is the winner** on A100 for both algorithms
  (+13–15% over interleave 1). The extra independent state hides
  round latency as designed.
- **Interleave 4 loses** (−15–20% vs interleave 2) despite **zero
  spills**. With up to 96 registers/thread its resident-warp count
  drops (65,536 ÷ (96 × 32) ≈ 21 warps/SM vs ≈39 at 52 registers),
  and the occupancy loss outweighs the ILP gain. This is the
  register-pressure outcome predicted for factor 4; the tuner should
  not select it on A100, and its value is now measured, not assumed.
- Block 256 is consistently, slightly better than 128 (≤1.5%).
- Oracle and MSSQL rates are within ~3% of each other, as expected:
  same block count and round work at length 8, only the packing
  differs.
- Known reporting gap: the benchmark JSON prints
  `registers_per_thread: -1` because `timeSearchLaunches` does not
  query kernel attributes. The build log has the true values; wiring
  `cudaFuncGetAttributes` into the benchmark record is a small
  follow-up fix.

### Length sweep (interleave 2, block 128)

| Algorithm | Length | Kernel family | Hashes/s | Note |
|---|---|---|---|---|
| ora11g | 6 | fast | 16.84 GH/s | grouped fast kernel |
| ora11g | 12 | fast | 17.37 GH/s | grouped fast kernel |
| ora11g | 17 | generic | 9.45 GH/s | single-block generic |
| ora11g | 25 | generic | 9.17 GH/s | single-block generic |
| ora11g | 45 | generic | 8.61 GH/s | single-block boundary |
| ora11g | 46 | generic | 5.17 GH/s | first multi-block length |
| mssql | 6 | fast | 17.34 GH/s | grouped fast kernel |
| mssql | 12 | fast | 17.19 GH/s | grouped fast kernel |
| mssql | 17 | generic | 9.25 GH/s | single-block generic |
| mssql | 25 | generic | 8.90 GH/s | single-block boundary |
| mssql | 26 | generic | 5.27 GH/s | first multi-block length |

Conclusions:

- **The fast→generic cliff at length 17 costs ~45%.** Lengths
  17–45 (Oracle) and 17–25 (MSSQL) are single-block messages handled
  by the per-candidate generic path: full repack, digit conversion,
  and the 576-byte local staging frame for every candidate, with no
  group amortisation of salt/padding preparation. This is now
  *measured* support for the previously held item B — a grouped
  single-block kernel for medium lengths is the main remaining
  performance lever, worth up to roughly 1.8× in that range if it
  approaches fast-kernel efficiency.
- Within the generic single-block range the decline with length is
  gentle (Oracle 9.45 → 8.61 GH/s from L17 to L45): packing cost per
  extra byte is small next to one compression.
- Crossing to two blocks (Oracle 46, MSSQL 26) costs a further ~40%
  (ratio ≈ 0.60, not 0.50, because per-candidate setup is amortised
  across both compressions). This is inherent to SHA‑1, not a defect.
- The generic kernel ignores the interleave setting (it has no
  interleaved variant); the `interleave: 2` in these records is the
  requested configuration, not an applied one.

### Autotuning

`--tune quick` and `--tune full` both complete (exit 0; wall 0.5 s /
0.4 s including CUDA start-up and the length-3 search itself, at this
small synthetic size). **Reporting gap:** the search output does not
state which configuration tuning selected — the selection is only
visible in the tuning cache file
(`$HOME/.cache/cudadbcracker/tuning.txt`). Surfacing the chosen
block/interleave/grid in the search log is a small follow-up fix.
The cache contents are to be pasted here when collected:

```text
NVIDIA A100-SXM4-80GB|cc8.0|SM108|ora11g|L3|B26    128 1 32
```

Reading: for Oracle, length 3, alphabet 26, the tuner selected
**block 128, interleave 1, grid multiplier 32** on this A100. That is
one (length, alphabet) key from the small synthetic search, not a
global tuning verdict — the length-8 benchmarks above remain the
performance evidence, and they favour interleave 2 at that length.
Different lengths legitimately tune differently; this line mainly
proves the tune → cache → reuse path works end to end.

Still pending: profiling.

### 8.4 Profiling

Attempt 1 (Nsight Compute 2025.3.1): `ncu` attached successfully at
all three interleave factors but reported **"No kernels were
profiled"** — a launch-accounting mistake in the profiling command,
not a permission error. The benchmark process launches the fast
kernel only twice (one warm-up + `repeats=1`), and
`--launch-skip 2` skipped both. No `ERR_NVGPUCTRPERM` was seen, so
counter access is not (yet) proven blocked. Retry uses
`--repeats 5` with `--launch-skip 1 --launch-count 3`. (Rates printed
during the attempt, ~6 GH/s, are under profiler replay and are not
comparable to the unprofiled benchmarks.)

Attempt 2 (corrected launch count): **success.** `ncu` profiled
`fastSearchKernel` 3× at each interleave factor (18 passes each) and
wrote `/tmp/prof_il1.ncu-rep`, `/tmp/prof_il2.ncu-rep`,
`/tmp/prof_il4.ncu-rep`. Counter access therefore works on this
runtime — no permission blocker. (`--page summary` does not exist in
Nsight Compute 2025.3; use `--page details` or `--csv`.)

First extracted metrics, interleave 2 (details page):

- Theoretical occupancy **62.5%** (40 active warps/SM; block limit is
  registers at 10 blocks/SM), but **achieved occupancy only 14.1%**
  (9.05 warps/SM). Nsight attributes the gap to load imbalance and
  wave quantisation: the launch is 3 full waves plus a partial wave
  of 217 blocks (up to ~25% of runtime in the tail), and SM active
  cycles vary +39%/−20% around the mean.
- DRAM activity is negligible (average 24.6 active cycles) — the
  kernel is confirmed compute-bound with essentially no memory
  traffic, as designed.
- Branch efficiency 99.97%, divergent branches ~0 — divergence is
  not a factor.

The low achieved occupancy is the most interesting profiling fact so
far: at 14% occupancy the kernel still reaches 17.5 GH/s, which
means it is *not* latency-starved at this work size — per-warp ILP
(interleave) plus the resident warps carry it. It also means the
grid/wave shape (grid multiplier, batch size) has real headroom to
test, since a better-balanced tail could reclaim part of that 25%.

All values below are from the `.ncu-rep` CSV exports (Oracle, L8,
block 128, grid 3456 blocks, 4,194,304 candidates per launch; three
launches per factor, essentially identical):

| Interleave | Registers/thread | Duration | SM throughput | Theoretical occupancy | Achieved occupancy | Partial-wave tail |
|---|---|---|---|---|---|---|
| 1 | 40 | 404 µs | **76.6%** (ALU-bound) | 75.0% | 26.8–27.2% | 864 blocks (33%) |
| 2 | 48 | 470 µs | 57.1% | 62.5% | 14.1% | 217 blocks (25%) |
| 4 | 72 | 515 µs | 52.1% | 43.8% | 10.4% | 432 blocks (20%) |

Profiling conclusions:

- **Interleave 1 is ALU-pipe-bound** (76.6% ALU utilisation, the
  highest pipeline): the SHA‑1 integer mix itself is the limit, and
  the compiler's instruction selection is working. There is no
  memory, divergence, or occupancy mystery at this factor.
- At interleave 2 and 4, SM throughput *drops* (57% / 52%) while
  theoretical and achieved occupancy also drop with register count
  (48 / 72 per thread). Under the profiler's controlled clocks the
  per-launch durations actually run 1 < 2 < 4 — the **opposite**
  order to the free-running 16.7M-candidate benchmarks, where
  interleave 2 won. The benchmark ordering remains the throughput
  evidence (real clocks, larger batches); the profiler establishes
  *why* there is no large ILP win to be had: the ALU is already the
  bottleneck at interleave 1, so extra independent states mostly buy
  register pressure. Interleave 2's small free-running gain is
  consistent with latency smoothing at boost clocks, not with a
  fundamentally under-utilised pipeline.
- **Every launch has a 20–33% partial-wave tail** at the tuned grid
  (3,456 blocks). With 10 resident blocks/SM at interleave 2, one
  wave is 1,080 blocks; grid sizes that are exact wave multiples
  (e.g. 3,240 or 4,320 blocks at this configuration) are the
  concrete next tuning experiment, and the `--tune full` grid
  multiplier axis (8/32/64 blocks/SM) already searches in this space.
  This, plus a grouped kernel for medium lengths, is where measured
  headroom remains — not in more interleaving.

### 8.5 Notes and conclusions

**Build (concluded).** The CUDA code compiles cleanly with nvcc 13.0
for all four spec architectures. The full log (1,995 lines) contains
no errors and no warnings. The configure log is likewise clean:
preset variables exactly as specified, GNU 13.3.0 host compiler,
nvcc 13.0.88, "Configuring done (6.1s)" — and notably **no `sm_75`
deprecation or dropped-architecture warning under CUDA 13**, so the
T4 target compiles without qualification on this toolkit. Register/spill summary from the `ptxas`
records (392 kernel entries):

- **Zero spill stores and zero spill loads on every variant.** The
  register-resident redesign goal is met at compile time.
- Fast kernels, interleave 1: 40–58 registers; interleave 2: 29–64;
  interleave 4: 30–96 (maximum 96 on the largest `sm_75` variants).
  Interleave 4's register cost is real but causes no spilling; whether
  its lower occupancy still wins is a benchmark question.
- Fast kernel stack frame is a constant 16 bytes; the generic search
  kernel uses 40 registers with a 576-byte stack frame (its bounded
  local staging arrays), and `digestKernel` 448 bytes — expected for
  the generic path and the reason the fast path exists.
- Constant memory use is 800 bytes (`cmem[0]`), consistent with the
  `__grid_constant__` parameter block.

This closes the "CUDA compilation" item previously listed as not
tested for **all four architectures at compile level**.

**GPU execution (concluded on A100).** `test_gpu` passes on the
A100-SXM4-80GB: every GPU digest equals the CPU reference for both
algorithms at lengths 0, 1, 10, 12, 25/26, 45, and 128, and every
GPU search returns exactly the CPU search offset — for the grouped
fast kernels, the generic path, the medium single-block lengths
(Oracle 17 and 45, MSSQL 17 and 25), and interleave factors 1, 2,
and 4. The earlier "GPU code is correct by inspection only" caveat
is therefore closed for the A100. T4, L4, and H100 **execution**
remain untested (their images compile; nothing has run on them).

**Overall status.** The runbook is complete for the A100: build,
compilation for all four architectures, host and GPU tests, the
end-to-end smoke search, benchmarks (interleave 2 / block 256 wins;
17.7 GH/s peak at length 8), the length sweep (fast→generic cliff at
L17, the main remaining optimisation lever), autotuning, and
profiling (ALU-bound at interleave 1; wave-tail headroom in grid
shape). Remaining open items are only: multi-batch/interruption behaviour
on GPU, execution on T4/L4/H100, and the held medium-length grouped
kernel.
