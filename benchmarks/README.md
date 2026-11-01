# Benchmarks

No benchmark numbers are reported yet: this development machine has no
NVIDIA GPU, and unmeasured GPUs must not be represented as verified
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
