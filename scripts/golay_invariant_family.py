#!/usr/bin/env python3
"""The Golay invariant J_5 of the regular family F(t, V) as a rational function of t.

For an integer t let a = c_23(t) be the leading coefficient of F(t, V) and
G_t(W) = a^22 F(t, W/a), a monic integral polynomial with roots a v_i.  Then
N(t) := J_5(a v_1, ..., a v_23) = a^5 J_5(v_1, ..., v_23) is an integer, and
N is a polynomial in t of degree at most 20 (the v_i are finite at t = infinity).
We compute N(t) for 26 integers t by certified Steiner labelling, interpolate
on 21 of them and check the other 5.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import arith, data  # noqa: E402

F = data.F()


def monic_specialisation(t: int) -> list[int]:
    c = data.specialise(F, t)
    a = c[23]
    return [c[j] * a ** (22 - j) for j in range(23)] + [1], a


values = {}
ts = [t for t in range(-13, 14) if t != 0]
for t in ts:
    G, a = monic_specialisation(t)
    lab = arith.steiner_labelling(G)
    j5 = arith.golay_invariants(G, lab)["J5"]
    values[t] = j5
    print(f"t = {t:3d}: labelling prime {lab.p}, sign {lab.sign:+d}, N(t) has {len(str(abs(j5)))} digits", flush=True)


def interpolate(points):
    """Lagrange interpolation; returns ascending rational coefficients."""
    xs = [x for x, _ in points]
    coeffs = [Fraction(0)] * len(points)
    for i, (xi, yi) in enumerate(points):
        basis = [Fraction(1)]
        denom = Fraction(1)
        for j, xj in enumerate(xs):
            if j == i:
                continue
            basis = [Fraction(0)] + basis
            for k in range(len(basis) - 1):
                basis[k] -= xj * basis[k + 1]
            denom *= (xi - xj)
        for k, b in enumerate(basis):
            coeffs[k] += yi * b / denom
    return coeffs


fit = [(t, values[t]) for t in ts[:21]]
N = interpolate(fit)
ok = all(sum(c * t**k for k, c in enumerate(N)) == values[t] for t in ts[21:])
deg = max(k for k, c in enumerate(N) if c != 0)
print("degree of N:", deg, "; check on", len(ts) - 21, "further points:", ok)
den = 1
for c in N:
    from math import lcm
    den = lcm(den, c.denominator)
print("common denominator of the coefficients of N:", den)
out = {"description": "J_5 of the roots of F(t, V) equals N(t) / c23(t)^5, c23 = leading coefficient in V",
       "c23": [str(F[i][23]) for i in range(5)],
       "N_numerators": [str(int(c * den)) for c in N], "N_denominator": str(den),
       "values": {str(t): str(v) for t, v in values.items()}, "verified_on_extra_points": ok}
(Path(__file__).resolve().parent.parent / "data" / "J5_family.json").write_text(json.dumps(out, indent=0) + "\n")
sys.exit(0 if ok else 1)
