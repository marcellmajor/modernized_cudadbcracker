#!/usr/bin/env python3
"""Run the cudadbcracker benchmark subcommand over a small matrix (spec 15).

Usage:
  benchmark_matrix.py --binary PATH [--device N] [--count 1048576]
                      [--lengths 6,12] [--blocks 128,256] [--interleaves 1,2]
Writes one JSON object per line to stdout. Refuses to invent numbers: if the
binary or a GPU is unavailable it reports the error and exits non-zero.
Only run this on a machine/service that permits this workload.
"""
import argparse
import json
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--count", default="1048576")
    ap.add_argument("--lengths", default="6,12")
    ap.add_argument("--blocks", default="128,256")
    ap.add_argument("--interleaves", default="1,2")
    args = ap.parse_args()
    rc = 0
    for alg in ("ora11g", "mssql"):
        for length in args.lengths.split(","):
            for block in args.blocks.split(","):
                for inter in args.interleaves.split(","):
                    cmd = [args.binary, "benchmark", "--algorithm", alg, "--length", length,
                           "--charset-classes", "uln", "--count", args.count,
                           "--device", args.device, "--block-size", block, "--interleave", inter]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        print(json.dumps({"algorithm": alg, "length": int(length),
                                          "error": (proc.stderr or proc.stdout).strip(),
                                          "exit_code": proc.returncode}))
                        rc = proc.returncode
                        continue
                    print(proc.stdout.strip())
    return rc


if __name__ == "__main__":
    sys.exit(main())
