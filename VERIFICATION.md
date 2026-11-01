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

- **CUDA compilation.** The `.cu` sources have not been compiled by
  `nvcc` in this environment. They were written against the CUDA 12
  runtime API and reviewed, but T4 (`sm_75`), A100 (`sm_80`), L4
  (`sm_89`), and H100 (`sm_90`) images are **not tested** and must not
  be represented as verified.
- **Reusable stream (post-review change).** All kernel launches use a
  single process-wide named stream (`<<<grid, block, 0, stream()>>>`
  with `cudaStreamSynchronize`), per spec section 10. This change is
  source-reviewed only; like the rest of the CUDA path it awaits nvcc
  compilation on a CUDA machine.
- **GPU tests.** `test_gpu` (GPU digest agreement incl. MSSQL 10–12 and
  multi-block lengths; GPU vs CPU search offsets for generic/fast and
  interleave 1/2/4, plus medium single-block lengths Oracle 17/45 and
  MSSQL 17/25) is implemented but skipped here for lack of a device.
- **Benchmarks and autotuning.** No hashes/second numbers exist yet.
  See `benchmarks/README.md` for the measurement protocol.
- **The full GPU search loop** (`runSearch` with GPU batches,
  checkpointing between batches, interruption) is exercised only up to
  the explicit CUDA-unavailable error on this machine.

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
