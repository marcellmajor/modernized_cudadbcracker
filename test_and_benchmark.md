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
ncu --import profile_ora_L8_il1.ncu-rep --page summary
ncu --import profile_ora_L8_il1.ncu-rep --page details
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

### 6.3 What "good" looks like (interpretation guide)

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
| `ctest -L gpu` / `test_gpu` | **Attempt 1: FAILED — test bug, not a kernel failure.** Aborted with `candidate offset overflows the length space`. Root cause: the MSSQL search case used password `BA` (uppercase) over the lowercase alphabet `abc`, so no candidate could match; the scan then ran past the end of the length-2 space (batch count 27 > 3² = 9), where offset arithmetic correctly throws. Fixed in `tests/test_gpu.cpp` (case is now `ba`, and every case's batch count is bounded by `min(27, 3^L)`). The GPU digest section and the Oracle cases preceding it had already run without a mismatch report. Awaiting a rebuilt re-run. |
| Synthetic search smoke test | _(pending)_ |

### 8.3 Benchmarks

| Algorithm | Length | Block | Interleave | Hashes/s | Registers | Spills | Kernel |
|---|---|---|---|---|---|---|---|
| _(pending)_ | | | | | | | |

### 8.4 Profiling

| Kernel / interleave | Duration | Achieved occupancy | SM throughput | Top stall reason |
|---|---|---|---|---|
| _(pending)_ | | | | |

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
tested for **all four architectures at compile level**. Execution is
so far possible only on the A100 present; T4/L4/H100 remain
execution-untested.

_(Still to be written from returned output: whether GPU offsets match
the CPU control everywhere including lengths 17/45/25 and interleave
4, which interleave/block configuration actually wins on A100 and by
how much, and what profiling says the bottleneck is.)_
