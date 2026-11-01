# Verification report

Development machine: macOS (ARM64), Apple clang, CMake 4.4 (pip-provided
into a session-local `/tmp` directory for local validation; it is not a
system install and may not persist between sessions), Ninja. **No NVIDIA
GPU and no CUDA toolkit are present in this environment.** Where CMake
is unavailable, the identical host compilation and tests can be
reproduced directly with `clang++ -std=c++17 -Wall -Wextra`.

## What was run

Host-only build (`CUDADB_ENABLE_CUDA=OFF`), configured and built with
CMake + Ninja, sources also compiled directly with `-Wall -Wextra`
without warnings:

```text
ctest -L host
  test_host    Passed
  test_search  Passed
  test_resume  Passed
ctest -L gpu
  test_gpu     Skipped (SKIP_RETURN_CODE 77, prints SKIPPED: no CUDA device)
```

Coverage exercised by those tests:

- All 11 known-answer vectors from `IMPLEMENTATION_SPECS.md` §14.1
  (Oracle/MSSQL, lengths 0, 10, 12, 25, 26, 45, 46, 128), plus standard
  SHA‑1 vectors.
- All 47 shipped Oracle records and all 47 shipped MSSQL records from
  `tests/fixtures/`, including both MSSQL digest fields and exact
  password verification.
- Strict parser rejection: short verifier, bad hex, `0x0200`, trailing
  data, duplicate label; acceptance of Oracle `S:` prefix and CRLF.
- Enumeration order (`aa, ba, ab, bb`), B=1, L=0, ranks beyond 2^64
  (`62^12` space round-trip).
- CPU search hits at first/middle/last offsets, unaligned starts,
  no-match, MSSQL lengths 1–26 including 10–12, length‑128 bounded hit.
- Case recovery: full recovery, budget exhaustion + resume from the
  saved assignment, and `case_recovery_no_match`.
- Checkpoint round-trip, corrupted-checkpoint rejection, fingerprint
  tamper rejection, schema rejection.

CLI end-to-end (host-only binary):

- `verify` (CPU) finds the correct password for synthetic Oracle and
  MSSQL dual records, reports `verify_mismatch` for a wrong password,
  and exits 0 for both; `--backend gpu` in a host-only build exits 3.
- `search` in a host-only build exits 3 with an explicit CUDA error
  (no silent CPU fallback).
- `resume` on a checkpoint whose targets are terminal exits 0;
  `resume` on a searching checkpoint in a host-only build exits 3.
- `devices` prints an empty list / `[]`; `self-test --extended` passes.

Python tools compile (`py_compile`), notebook and preset JSON parse,
and `tools/generate_vectors.py --table` reproduces the §14.1 table.

## What was NOT run

- **CUDA compilation — now verified externally.** The distribution
  build was compiled on an A100 machine with nvcc 13.0.88 / GNU 13.3
  (CMake 3.31.10): 23/23 targets, zero errors/warnings, 98 kernel
  entries per architecture for `sm_75`, `sm_80`, `sm_89`, `sm_90`,
  and **zero register spills on all 392 entries** (fast kernels
  29–96 registers by interleave; see `test_and_benchmark.md` section
  8). This verifies *compilation* for all four architectures.
  *Execution* is verified only where `test_gpu` has actually run;
  T4, L4, and H100 execution remain **not tested**.
- **Reusable stream.** All kernel launches use a single process-wide
  named stream (`<<<grid, block, 0, stream()>>>` with
  `cudaStreamSynchronize`), per spec section 10. Compiled with the
  distribution build above and exercised by the passing `test_gpu`.
- **GPU tests — passed on A100.** `test_gpu` passes on an
  A100-SXM4-80GB (commit `b775209`): GPU digest agreement incl.
  MSSQL 10–12 and multi-block lengths, and GPU vs CPU search offsets
  for generic/fast kernels at interleave 1/2/4, including medium
  single-block lengths Oracle 17/45 and MSSQL 17/25. Host tests and
  `self-test --extended --device 0` also pass on that machine.
  Execution on T4, L4, and H100 remains **not tested**.
- **Benchmarks — first numbers on A100.** Length 8, alphabet 62,
  device-event timing: interleave 2 / block 256 is fastest for both
  algorithms (Oracle 17.71 GH/s, MSSQL 17.25 GH/s); interleave 1 is
  ~13% slower and interleave 4 is 15–20% slower despite zero spills
  (register-limited occupancy at up to 96 registers/thread). Full
  table and analysis in `test_and_benchmark.md` section 8.3.
  Autotuning runs, the length sweep, and profiling are still
  outstanding; no other GPU has benchmark numbers.
- **The GPU search loop** is exercised end-to-end on the A100: the
  synthetic Oracle search found `cab` at the canonical rank 678 with
  CPU re-verification, checkpoint, and JSONL output (see
  `test_and_benchmark.md` Step 4.3 results). Multi-batch checkpoint
  cadence over a long search and signal interruption on GPU have not
  been separately exercised.

## Post-review optimisations (unmeasured)

Implemented after an external optimisation review, as hypotheses to
measure against the corrected generic baseline — none is claimed faster:

- Fast-kernel group buffers sized by compile-time length `L` (was
  `[128]`), and the per-group `KernelParams` copy removed
  (`packSingleBlockEx` takes algorithm/length explicitly).
- Kernel parameters marked `__grid_constant__`.
- Interleave factor 4 (four round-interleaved states), opt-in via
  `--interleave 4` or `--tune full`; default remains 1.
- Generic path uses a single-block digest without the multi-block
  staging buffer for lengths that fit one block (Oracle ≤45,
  MSSQL ≤25), covering lengths 17–45/25 without new template
  instantiations.
- `--tune full` also sweeps grid multiplier (8/32/64) and interleave 4.

Second-round refinements (also source-review only, unmeasured):

- Fast-kernel grid sized from the interleave-adjusted thread-work count
  (`ceil(groups / interleave)`) rather than the raw group count, so
  interleave 2/4 no longer launch 2×/4× redundant threads and the grid
  multiplier means the same thing at every interleave factor.
- `#pragma unroll` added to the interleave-4 group/word loops, matching
  the interleave 1/2 paths.
- `digestKernel` uses the single-block path when eligible, so
  `verify`/`self-test` cover `digestSingleBlock` (not only search); and
  `test_gpu` gained Oracle 17/45 and MSSQL 17/25 search cases that
  differential-test the generic single-block path and its exact boundary
  against the CPU control.

Deliberately not implemented: reversed final rounds with one-word early
reject (the only real saving — skipping the five final additions —
requires surgery inside the verified SHA-1 cores and would have to be
replicated across the single/pair/quad variants, for ~1%, unvalidatable
here); group-invariant schedule precomputation (complex, correctness
risk without a GPU); a grouped single-block kernel for medium lengths
17–45/25 (real headroom, but a new search-kernel family that should be
written against nvcc and measured, not added blind); host/device
double-buffering (minor at the default batch size); changing the
canonical enumeration order (spec section 7 fixes it).

## Next verification steps on a permitted CUDA machine

1. `cmake --preset distribution`, build with `--parallel 2`, inspect
   `-Xptxas=-v` output for registers/spills per variant.
2. `ctest -L host` and `ctest -L gpu`; `self-test --extended --device 0`.
3. Confirm embedded images with `cuobjdump` for `sm_75/80/89/90`.
4. Run the benchmark matrix, fill `benchmarks/README.md` records, and
   record per-GPU status here, marking any untested GPU explicitly.
