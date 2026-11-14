#!/usr/bin/env python3
"""The six passports (C, 23A, 23B), C a rational class, with branch locus {0, +-sqrt(-23)}.

For C in {2A, 3A, 4A, 5A, 6A, 8A}: size of the inner Nielsen class (= degree of the
Hurwitz scheme over Q), genus of the degree 23 cover, number of real points of the
Hurwitz scheme, and for each real cover the number of ovals.  Checks along the way:
dim W = g (Golay Lagrangian), dim Jac(R)[2] = g + r - 1, and for C = 2A agreement
with the published seven-element class.  Writes data/passports_C_23A_23B.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import nielsen  # noqa: E402

out = {}
for cls in ("2A", "3A", "4A", "5A", "6A", "8A"):
    rep = nielsen.passport_report(cls)
    out[cls] = rep
    ovals = sorted(r["components"] for r in rep["real"])
    print(f"({cls},23A,23B): genus {rep['genus']}, |Ni| = {rep['size']}, real points {rep['real_classes']}, "
          f"ovals of the real covers {ovals}", flush=True)
two = nielsen.nielsen_class_of("2A")
assert two == nielsen.nielsen_class(), "2A enumeration disagrees with nielsen_class()"
old = sorted(nielsen.real_components(t, nielsen.real_structure(t)) for t in two if nielsen.real_structure(t))
assert old == sorted(r["components"] for r in out["2A"]["real"])
(Path(__file__).resolve().parent.parent / "data" / "passports_C_23A_23B.json").write_text(json.dumps(out, indent=1) + "\n")
print("all checks passed")
