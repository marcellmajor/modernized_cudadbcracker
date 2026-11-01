#!/usr/bin/env python3
"""Generate independent SHA-1 test vectors with hashlib (spec sections 2, 14).

Usage:
  generate_vectors.py --algorithm ora11g --password-hex HEX --salt-hex HEX
  generate_vectors.py --table          # the section 14.1 'A'*L table
Prints the expected digest hex. This is the independent oracle; the C++
implementation must never be used to generate its own expectations.
"""
import argparse
import hashlib
import sys


def digest(algorithm: str, password: bytes, salt: bytes) -> str:
    if algorithm == "ora11g":
        if len(salt) != 10:
            raise SystemExit("oracle salt must be 10 bytes")
        msg = password + salt
    else:
        if len(salt) != 4:
            raise SystemExit("mssql salt must be 4 bytes")
        if any(b > 0x7F for b in password):
            raise SystemExit("mssql password must be ASCII")
        msg = b"".join(bytes([b, 0]) for b in password) + salt
    return hashlib.sha1(msg).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["ora11g", "mssql"])
    ap.add_argument("--password-hex", default="")
    ap.add_argument("--salt-hex")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        for alg, salt, lengths in (
            ("ora11g", bytes(range(10)), [0, 12, 45, 46, 128]),
            ("mssql", bytes(range(4)), [0, 10, 12, 25, 26, 128]),
        ):
            for length in lengths:
                print(alg, length, digest(alg, b"A" * length, salt))
        return 0
    if not args.algorithm or args.salt_hex is None:
        ap.error("--algorithm and --salt-hex are required unless --table is given")
    password = bytes.fromhex(args.password_hex)
    salt = bytes.fromhex(args.salt_hex)
    print(digest(args.algorithm, password, salt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
