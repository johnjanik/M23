#!/usr/bin/env python3
"""Recount selected rows of data/rustmath_wpf_tables.json with the Python reference
implementation (a different code base, a different plane model for X) and record the result.

Usage: crosscheck_rustmath.py [processes] [p ...]      default primes: 83 97 107
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
primes = [int(a) for a in sys.argv[2:]] or [83, 97, 107]
ref = {r["p"]: r for r in json.loads((ROOT / "data" / "rustmath_wpf_tables.json").read_text())["X"]}
out_path = ROOT / "data" / "rustmath_crosscheck.json"
out = json.loads(out_path.read_text()) if out_path.exists() else {}
for p in primes:
    counts = [curve.count_points(p, k, fibre="V", processes=procs) for k in (1, 2, 3, 4)]
    P = curve.frobenius_polynomial(p, counts)
    ok = counts == ref[p]["counts"] and P[1:5] == ref[p]["a"]
    out[str(p)] = {"counts": counts, "a": P[1:5], "equal_to_rustmath": ok}
    print(p, counts, P[1:5], "EQUAL" if ok else f"DIFFERENT from {ref[p]['counts']}", flush=True)
    out_path.write_text(json.dumps(out, indent=1) + "\n")
