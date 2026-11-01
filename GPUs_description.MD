# GPUs in the specification: architecture and expected performance

Scope: the four NVIDIA targets in `IMPLEMENTATION_SPECS.md` — T4, A100,
L4, and H100 — ranked by **expected** performance for this project's
salted SHA‑1 kernels (`SHA1(password‖salt)` constructions, candidate
lengths 0–128).

> **Status: unmeasured.** No GPU in this list has been benchmarked with
> this code (no CUDA device is available in the development
> environment; see `VERIFICATION.md`). The ranking below is an
> engineering expectation derived from published hardware
> characteristics that matter to this kernel. It is not a measured
> hashes/second result, and it must be replaced by measurements from
> `benchmarks/README.md`'s protocol before being quoted as performance.

## What this kernel actually needs

The SHA‑1 search kernels have a specific profile, and most headline GPU
figures are irrelevant to them:

- **Integer arithmetic throughput.** A hash is 80 rounds of 32-bit
  additions, three-input Boolean functions (ideally one `LOP3` each),
  fixed rotations (funnel shifts), plus the message-schedule XOR/rotate
  expansion. There is no floating point and no matrix work.
- **Latency hiding.** Each hash is a long serial dependency chain
  (~80 dependent rounds). Throughput comes from running many
  independent candidates concurrently (warps per SM) and, in the fast
  kernels, from interleaving 1/2/4 independent hashes per thread.
- **Registers.** All four GPUs provide 65,536 32-bit registers per SM.
  Register pressure — especially at interleave 4 — can limit resident
  warps before raw ALU capacity does.
- **Almost no memory traffic.** Inputs are a small descriptor, a
  charset (≤256 bytes), a salt and a digest. Candidates are generated
  on-device. Memory bandwidth and capacity are therefore *not*
  differentiators for a single search, which is why HBM vs GDDR matters
  far less here than in AI/HPC workloads.
- **Not used:** Tensor Cores, FP64, ray tracing, TMA, thread-block
  clusters. Tensor TOPS and FP32/FP16 TFLOPS do not predict SHA‑1 rate.

A useful first-order model is therefore the **peak INT32 issue rate**:

```text
INT32 ops/s ≈ SMs × 64 INT32 lanes/SM/clock × sustained clock
```

All four architectures provide 64 INT32 lanes per SM per clock
(Turing: 64 dedicated INT32 cores/SM; GA100: 64 INT32 cores/SM;
Ada: 16 FP32-only + 16 FP32/INT32 per SM partition × 4 = 64 INT32/SM;
Hopper follows the same 64 INT32/SM organisation). Real hashes/second
will be a fraction of this ceiling — each round costs several integer
instructions, not one — but the *ratio* between GPUs should roughly
track this model provided the kernel is not register- or
occupancy-limited on any of them.

## Ranking (expected, fastest first)

| Rank | GPU | INT32 ceiling (approx.) | Relative to T4 |
|---:|---|---:|---:|
| 1 | H100 SXM | ~16.7 × 10¹² ops/s (132 SM × 64 × 1.98 GHz) | ~4.1× |
| 2 | H100 PCIe | ~13.0–13.4 × 10¹² ops/s (114 SM × 64 × ~1.8 GHz) | ~3.2× |
| 3 | A100 | ~9.7 × 10¹² ops/s (108 SM × 64 × 1.41 GHz) | ~2.4× |
| 4 | L4 | ~7.6 × 10¹² ops/s (58 SM × 64 × 2.04 GHz) | ~1.9× |
| 5 | T4 | ~4.1 × 10¹² ops/s (40 SM × 64 × 1.59 GHz) | 1.0× |

Clocks are boost clocks; sustained clocks under a continuous integer
load, power caps, and partitioned instances (for example MIG slices)
can lower all figures and, in principle, compress the gaps. The two
H100 variants are separated because their SM counts and power limits
differ materially.

### Performance per watt (different ranking)

Using the same ceiling model divided by TDP, the order reverses for
the low-power parts:

| GPU | TDP | INT32 ceiling / watt (approx.) |
|---|---:|---:|
| L4 | 72 W | ~105 Gops/W |
| T4 | 70 W | ~58 Gops/W |
| H100 PCIe | 350 W | ~38 Gops/W |
| A100 (PCIe 250 W / SXM 400 W) | 250–400 W | ~24–39 Gops/W |
| H100 SXM | 700 W | ~24 Gops/W |

If throughput per watt or per slot is the constraint (dense inference
servers, cloud cost), the L4 is the most attractive card in the list
despite ranking fourth in absolute speed.

## The GPUs in detail

### 1. H100 — Hopper (GH100), compute capability 9.0, target `sm_90`

- **Variants:** SXM (132 SMs, ~1.98 GHz boost, 80 GB HBM3 at
  ~3.35 TB/s, 700 W) and PCIe (114 SMs, ~1.8 GHz boost, ~2 TB/s,
  350 W). Both are built on TSMC 4N; the full GH100 die is larger than
  either product enables.
- **Per SM:** 128 FP32 lanes, 64 INT32 lanes, 65,536 registers,
  up to 228 KiB shared memory per SM (largest in this list), 64 resident
  warps maximum.
- **Why it ranks first:** the most SMs *and* a high clock give it
  roughly 1.7× the A100's integer ceiling. The large shared-memory and
  register budgets also give the register-heavy interleave variants the
  most headroom, so H100 is the least likely card to be
  occupancy-limited by this kernel.
- **Caveats:** the SXM part's 700 W envelope and platform cost are
  aimed at AI training; for this kernel its extra HBM bandwidth and
  fourth-generation Tensor Cores buy nothing. Perf/watt is the worst
  in the list. MIG partitioning can expose a fraction of the SMs, so
  the runtime must query the actual device (the implementation does).

### 2. H100 PCIe — same silicon family, reduced configuration

See above. With 114 SMs and a 350 W cap it sits between the SXM part
and the A100 in the model (~1.35× the A100 ceiling). It is listed
separately in the ranking because a "H100" label alone does not say
which configuration is present; `cudadbcracker devices` reports the
actual SM count.

### 3. A100 — Ampere (GA100), compute capability 8.0, target `sm_80`

- **Configuration:** 108 SMs, 6,912 FP32 + 6,912 INT32 cores
  (64 + 64 per SM), ~1.41 GHz boost, TSMC 7 nm. Memory is HBM2e:
  40 GB at ~1.56 TB/s or 80 GB at ~1.94 TB/s; TDP 250 W (PCIe) to
  400 W (SXM). Up to 164 KiB shared memory/SM, 64 resident warps.
- **Why it ranks third:** it has 86% more SMs than the L4. The L4's
  ~45% clock advantage and newer process do not close that gap in the
  INT32 model (~9.7 vs ~7.6 Tops/s). Its separate FP32 and INT32
  datapaths also mean integer work never competes with floating point
  for issue slots.
- **Caveats:** an older, power-hungry HPC part; perf/watt is poor next
  to the L4. As with H100, its headline AI figures (Tensor Cores,
  HBM bandwidth) are irrelevant to SHA‑1.

### 4. L4 — Ada Lovelace (AD104), compute capability 8.9, target `sm_89`

- **Configuration:** 58 SMs (of 60 on the full AD104 die), 7,424 FP32
  lanes with 64 INT32 lanes/SM available through the FP32/INT32
  datapaths, ~2.04 GHz boost, TSMC 4N. 24 GB GDDR6 at ~300 GB/s,
  single-slot low-profile card at 72 W. Up to 100 KiB shared
  memory/SM, 48 resident warps.
- **Why it ranks fourth in speed but first in efficiency:** the high
  clock partly offsets having the second-fewest SMs. It delivers
  roughly 1.9× the T4's integer ceiling at essentially the same power,
  making it the natural upgrade in T4-class slots and the best
  perf/watt and (usually) perf/cost choice for this workload.
- **Caveats:** 48 rather than 64 maximum resident warps per SM and a
  smaller shared-memory budget give slightly less latency-hiding
  headroom per SM; at interleave 4 its register file per SM is the
  same 65,536, so per-thread register limits — not SM resources — are
  what the tuner must watch.

### 5. T4 — Turing (TU104), compute capability 7.5, target `sm_75`

- **Configuration:** 40 SMs, 2,560 FP32 + 2,560 INT32 cores
  (64 + 64 per SM), ~1.59 GHz boost, TSMC 12 nm. 16 GB GDDR6 at
  ~320 GB/s, single-slot at 70 W. Up to 64 KiB shared memory/SM,
  32 resident warps.
- **Why it ranks last:** fewest SMs and the joint-lowest clock product
  in the INT32 model. It is roughly a quarter of an H100 SXM's ceiling.
- **Role in the list:** it is the baseline target — widely available,
  cheap, and low-power — and the oldest architecture the modern
  toolchain in `IMPLEMENTATION_SPECS.md` supports. It remains
  perfectly serviceable for small searches; it is simply not the card
  to choose when wall-clock time matters.

## What could change the ranking in practice

1. **Register limits at high interleave.** Interleave 4 keeps four
   SHA‑1 states and schedules live. If it spills on one architecture
   but not another, measured order between neighbouring ranks
   (especially L4 vs A100) could narrow. This is why factor 4 is
   opt-in and tuner-selected.
2. **Sustained clocks.** Boost figures are not guaranteed under an
   all-integer load. Power-capped or partitioned instances can reorder
   close pairs.
3. **Kernel family mix.** Very long candidates (generic multi-block
   path) add per-candidate setup that is proportionally larger on
   faster cards; short lengths use the grouped fast kernels and track
   the INT32 model most closely.
4. **Nothing memory-related.** Unless a future variant becomes
   host-transfer-bound (tiny batches, frequent checkpoints), the
   5–10× memory-bandwidth differences in this list should not appear
   in hashes/second at all.

## How to replace this expectation with measurements

On a permitted CUDA machine, per GPU: build the distribution preset,
run `ctest -L gpu`, then `tools/benchmark_matrix.py` over lengths,
block sizes, and interleave factors (see `benchmarks/README.md`).
Record hashes/second with GPU model/partition, driver, registers,
spills, and correctness status, and update the ranking table from
those rows. Until then, every figure above is a model, not a result.

## Sources

- `IMPLEMENTATION_SPECS.md` section 3 (targets, SM counts, register
  and shared-memory limits) and `optimization_adaptation.txt`
  sections 6–7 in the parent directory.
- [NVIDIA CUDA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA Ada GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) (AD104 SM organisation, L4)
- [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html),
  [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
  [Ada](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), and
  [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
  tuning guides (per-SM limits used by the occupancy discussion).
