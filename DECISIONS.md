# Decisions where IMPLEMENTATION_SPECS.md is silent

`IMPLEMENTATION_SPECS.md` is the definitive guide. The choices below only
fill gaps identified in `IMPLEMENTATION_SPECS_muse_amendment.md` and
`IMPLEMENTATION_SPECS_grok_amendment.md`; none of them overrides the spec.
They are recorded here so they can be changed in one place.

1. **MSSQL record layout.** `0x0100 || salt_4 || case_sensitive_digest ||
   uppercase_digest` for the dual form; the single form omits the last
   field. Verified against the shipped fixtures and the original parser
   offsets in `../source/main.cpp`.
2. **Search scheduling.** Target-major: each target is searched over
   lengths in increasing order before the next target. A target with
   pending case recovery is suspended and the next target continues.
   The checkpoint cursor per target is `{length, rank}` with `rank` a
   decimal string of the next untested length-local rank.
3. **Checkpoint contents.** The checkpoint embeds complete canonical
   target records (label, salt, digests, per-target effective alphabet),
   so `resume` needs no `--input`. The fingerprint is SHA‑1 hex over
   `algorithm|salt_hex|digest_hex[|upper_digest_hex]`; the label is not
   part of the fingerprint.
4. **Checkpoint frequency.** After every completed batch, after a hit,
   after case-recovery state changes, and on interruption.
5. **Case assignment order.** Bit `i` (LSB first) controls the `i`-th
   ASCII alphabetic position left-to-right; set = uppercase, clear =
   lowercase. Index 0 is all-lowercase. Only `a`–`z`/`A`–`Z` positions
   consume bits.
6. **Ranges.** `--start-rank` is inclusive; `--count` is a candidate
   count; the range is `[start, start+count)`. Overshooting the space is
   an input error. `--count` must be at least 1.
7. **Length zero.** Valid only when explicitly requested (`--length 0`
   or `--min-length 0`); the defaults remain lengths 1–8.
8. **Parser lexical rules.** CRLF is a line ending. Lines are trimmed of
   surrounding spaces/tabs. A line whose first non-space character is
   `#` is a comment. Empty labels are rejected. Oracle `S:`/`s:` and
   MSSQL `0x0100`/`0X0100` prefixes are accepted; hex is case-insensitive.
   Duplicate labels are always rejected (no disambiguation switch).
9. **Additional statuses.** `case_recovery_no_match` (required by spec
   §4.3) and `verify_mismatch` are used. A verify mismatch is a completed
   operation (exit 0); exit 4 is reserved for GPU/CPU disagreement.
10. **JSON field names.** Results use: `schema`, `target_id` (the account
    label), `algorithm`, `mode`, `status`, `verification`,
    `password_hex`, `password_display`, `candidate_length`, `length`,
    `rank`, and `pending_upper_hex` when applicable. `password_display`
    keeps printable ASCII and escapes other bytes as `\xNN`.
11. **Defaults.** Search lengths 1–8, device 0, tune `quick`,
    case budget 1,048,576, batch 1,048,576 candidates. Tune-off fallback:
    block 128, interleave 1, grid multiplier 32 blocks/SM. Results go to
    stdout unless `--results` is given. No checkpoint is written unless
    `--checkpoint` is given (interruption then reports `(none)`).
12. **Interleave 4.** Implemented as an opt-in experiment per spec §9.3:
    `--interleave` accepts 1, 2, or 4; the default stays 1 and factor 4
    is only selected explicitly or by `--tune full`. It is register-heavy
    and unmeasured; the tuner must reject it where it loses.
13. **Dependencies.** No Boost, nlohmann/json, or OpenSSL link dependency.
    Arbitrary precision uses an internal `BigUint` (spec §7 permits
    "equivalent digit arrays"); JSON uses an internal minimal
    parser/writer; independent test expectations come from Python
    `hashlib` fixtures. This keeps host-only builds reproducible offline.
14. **Tuning cache.** A plain-text cache at `$CUDADB_TUNE_CACHE` or
    `$HOME/.cache/cudadbcracker/tuning.txt`, keyed by device name,
    compute capability, SM count, algorithm, length, and alphabet size.
15. **Optimisation scope (held items).** Low-risk review refinements were
    applied source-only (interleave-adjusted grid sizing, interleave-4
    `#pragma unroll`, `digestKernel` single-block coverage); see
    `VERIFICATION.md`. Two items were deliberately deferred:
    (a) *reversed final rounds / one-word early reject* — the only real
    saving (skipping the five final additions) needs changes inside the
    verified SHA-1 cores, replicated across the single/pair/quad
    variants, for roughly 1%, and cannot be validated here without a GPU;
    (b) *a grouped single-block kernel for medium lengths* (Oracle 17–45,
    MSSQL 17–25) — genuine throughput headroom, but a new search-kernel
    family that should be written against `nvcc` and measured, not added
    blind. Both remain open work for a permitted CUDA machine.
