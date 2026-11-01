# Benchmarks

First measured results exist for one GPU (A100-SXM4-80GB, CUDA 13.0,
driver 580.82.07; full records and context in
`test_and_benchmark.md` section 8). Headlines, length 8, alphabet 62,
device-event timing: fast kernels reach **17.7 GH/s** (interleave 2,
block 256); single-block generic lengths 17–45 run at 8.6–9.5 GH/s;
the first multi-block lengths at ~5.2 GH/s. No other GPU has numbers;
unmeasured GPUs must not be represented as verified
(spec sections 11, 15).

On a permitted CUDA runtime, collect measurements with:

```bash
./build/cudadbcracker benchmark --algorithm ora11g --length 8 \
  --charset-classes uln --count 16777216 --repeats 5 --device 0
python3 tools/benchmark_matrix.py --binary ./build/cudadbcracker --device 0
```

Each record must include: GPU/model/partition, driver, toolkit, build
ID, compiler flags, architecture image, algorithm/mode, length,
alphabet size, salt/target count, block/grid, batch count, interleave,
registers per thread, shared memory, repetitions, timing distribution,
verified hashes/second (device-only and end-to-end separately), and
correctness status. A hash is one candidate/salt digest evaluation.
CPU case-recovery trials are reported separately from GPU hashes.

Compare variants against the corrected generic baseline and change one
factor at a time where practical: kernel family, interleave factor,
block size, grid multiplier, batch size, charset placement.
