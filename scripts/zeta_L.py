#!/usr/bin/env python3
"""L-polynomials of the covers defined over the sextic field L at primes of degree one.

For a prime p and a root s0 of the defining polynomial of L modulo p, the model
calF of [HJLPPZ26, Section 3.7] reduces to a curve over F_p.  We count points,
form the Frobenius polynomial, and test the Golay factorisation P = Q Q^rev (mod 2)
predicted by the Lagrangian theorem for all seven covers.

As a check of the point counter against the literature: at p = 53 the three
degree one primes give #X(F_53) = 45, 57, 52; the values 45 and 52 are those of
[HJLPPZ26, Proposition 3.14].

Usage: zeta_L.py [max_prime] [processes]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flint import nmod_poly  # noqa: E402
from m23golay import curve, data  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "zeta_L.json"
bound = int(sys.argv[1]) if len(sys.argv) > 1 else 40
procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

results = []
n53 = sorted(curve.count_points(53, 1, coeffs=data.calF_mod(53, int(r)), fibre="V", strict=False)
             for r, _ in nmod_poly([c % 53 for c in data.L_POLY], 53).roots())
print("check at 53:", n53, flush=True)
assert n53 == [45, 52, 57]
for p in range(5, bound + 1):
    if any(p % d == 0 for d in range(2, int(p**0.5) + 1)) or p == 23:
        continue
    for r, _m in nmod_poly([c % p for c in data.L_POLY], p).roots():
        s0 = int(r)
        A = data.calF_mod(p, s0)
        entry = {"p": p, "s0": s0}
        try:
            counts = [curve.count_points(p, k, coeffs=A, fibre="V", strict=False, processes=procs) for k in (1, 2, 3, 4)]
            P = curve.frobenius_polynomial(p, counts)
            if not curve.is_weil_polynomial(p, P):
                raise ValueError("not a Weil polynomial")
            split = curve.golay_splitting(P)
            entry.update({"status": "ok", "counts": counts, "P": P,
                          "P mod 2": [[curve.bits_to_str(f), m] for f, m in split["P_mod_2_factors"]],
                          "Q candidates": [curve.bits_to_str(q) for q in split["Q_candidates"]],
                          "golay_split": split["splits"]})
        except Exception as exc:
            entry.update({"status": "skipped", "reason": repr(exc)})
        results.append(entry)
        print(json.dumps(entry), flush=True)
        OUT.write_text(json.dumps({"check_53": n53, "results": results}, indent=1) + "\n")
