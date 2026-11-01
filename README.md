# Modernized CUDA DB Cracker

Standalone Linux C++/CUDA reimplementation of the original salted SHA‑1
cracker for Oracle 11g and legacy MSSQL verifiers. It fixes the original
correctness and portability defects, supports candidate lengths 0–128,
and provides measured tuning for NVIDIA T4, A100, L4, and H100 GPUs.

Derived from the original `cudadbcracker` by Marcell Major
(marcell.major@gmail.com), GPL‑3.0‑or‑later; see `LICENSE`. The original
`../source` and `../binary` directories are unchanged.

> **Deployment restriction.** Managed Google Colab prohibits password
> cracking, including paid runtimes. The build workflow is
> notebook-compatible, but actual searches must run only on a machine or
> service that permits this workload. The notebook never starts a search.

## Constructions

```text
Oracle 11g:  SHA1(password_bytes || salt_10_bytes)
MSSQL ASCII: SHA1(UTF16LE_ASCII(password) || salt_4_bytes)
```

MSSQL input is ASCII-only; non-ASCII candidates are rejected explicitly.
Dual-digest MSSQL records (`0x0100` + salt + case-sensitive digest +
uppercase digest) can be searched in uppercase mode, followed by CPU
case recovery against the case-sensitive digest.

## Build

Requirements: CMake ≥ 3.24, Ninja, a supported GCC, and (for GPU builds)
a CUDA 12.x toolkit. Host-only builds need no GPU or CUDA.

```bash
# Host-only (CPU reference, parsers, checkpoint, tests)
cmake --preset host-only
cmake --build --preset host-only --parallel 2
ctest --test-dir build-host --output-on-failure -L host

# Distribution build with native images for T4/A100/L4/H100
cmake --preset distribution
cmake --build --preset distribution --parallel 2
ctest --test-dir build --output-on-failure -L host
./build/cudadbcracker devices --json
```

On a permitted CUDA runtime, additionally:

```bash
./build/cudadbcracker self-test --extended --device 0
ctest --test-dir build --output-on-failure -L gpu
```

`tools/check_environment.py` prints read-only environment diagnostics.

## Usage

```text
cudadbcracker devices [--json]
cudadbcracker verify --algorithm ora11g|mssql --input FILE --password-hex HEX
    [--backend cpu|gpu|both] [--device N]
cudadbcracker self-test [--device N] [--extended] [--json]
cudadbcracker search --algorithm ora11g|mssql --input FILE
    --charset-classes CLASSES | --charset-hex HEX
    [--min-length N] [--max-length N] | [--length N --start-rank D --count D]
    [--mssql-mode auto|uppercase|direct] [--device N] [--tune off|quick|full]
    [--block-size N] [--interleave N] [--batch-candidates N]
    [--case-budget N] [--checkpoint FILE] [--results FILE]
cudadbcracker resume --checkpoint FILE [--device N] [--tune off|quick|full]
    [--case-budget N] [--results FILE]
cudadbcracker benchmark --algorithm ora11g|mssql --length N
    --charset-classes CLASSES --count N [--repeats N] [--device N]
    [--block-size N] [--interleave N] [--json FILE]
```

Exit codes: 0 completed operation (including range exhausted and verify
mismatch), 2 invalid input/configuration, 3 CUDA/runtime failure,
4 GPU/CPU correctness disagreement, 130 interrupted with checkpoint.

Results are JSON Lines on stdout (or `--results`); progress goes to
stderr. Every GPU hit is recomputed on the CPU before being reported.

## Layout

```text
include/cudadb/   public headers (types, algorithms, enumeration,
                  input, checkpoint, dispatch, search)
src/              host implementation and CLI
src/cuda/         SHA-1 core, message packing, generic and fast kernels,
                  launch, dispatch, and device queries
tests/            host, search, resume, and GPU tests plus fixtures
tools/            hashlib vector generator, environment check, benchmarks
notebooks/        build-and-validate notebook (never starts a search)
benchmarks/       benchmark reporting notes
```

The implementation contract is `IMPLEMENTATION_SPECS.md`. Choices made
where that specification is silent are recorded in `DECISIONS.md`.
Measured results and per-GPU test status belong in `VERIFICATION.md`;
un-tested GPUs are reported as not tested, never as verified.
