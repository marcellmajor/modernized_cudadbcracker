#!/usr/bin/env python3
"""Print build/run environment diagnostics (spec section 16).

Read-only: never installs drivers, changes CUDA symlinks, or starts a search.
Usage: check_environment.py
Exit code: 0 if basic build tools are present, 1 otherwise.
"""
import shutil
import subprocess
import sys


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"(unavailable: {exc})"


def main() -> int:
    print("OS:", run(["uname", "-a"]))
    for tool in ("cmake", "ninja", "g++", "nvcc", "nvidia-smi", "python3"):
        print(f"{tool}:", shutil.which(tool) or "(not found)")
    if shutil.which("nvcc"):
        print("nvcc --version:\n" + run(["nvcc", "--version"]))
    if shutil.which("nvidia-smi"):
        print("nvidia-smi (driver/toolkit banner is NOT the installed toolkit):")
        print(run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv"]))
    else:
        print("nvidia-smi: no NVIDIA driver/GPU visible in this environment")
    missing = [t for t in ("cmake", "ninja", "g++") if not shutil.which(t)]
    if missing:
        print("Missing build tools:", ", ".join(missing))
        return 1
    print("Basic build tools present. CUDA toolkit:", "present" if shutil.which("nvcc") else "absent (host-only build possible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
