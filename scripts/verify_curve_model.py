#!/usr/bin/env python3
"""Certify the reduction of the plane model F(T,V) = 0 at given primes (default: all
primes of data/zeta_X.json with p <= 31).

For each prime p:
  1. the identity disc_V(F) = 2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h62(T)^2 is
     verified over Z once (h62 from data/h62.json);
  2. curve.reduction_certificate: the model mod p is absolutely irreducible and the delta
     invariants of its singular points, found by resolving each of them, add up to 62.
     By the reduction lemma of the paper, X then has good reduction at p and its
     reduction is the normalization of the model mod p;
  3. the singular points are consistent with h62: 62 nodes when h62 mod p is squarefree
     of degree 62 and prime to T(T^2+23);
  4. #X(F_{p^k}) by both fibrations (degree 23 and degree 4) agree for small k, and the
     Frobenius polynomial is a Weil polynomial satisfying the congruences mod 23.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flint import fmpz_poly, nmod_poly  # noqa: E402
from m23golay import curve, data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
F = data.F()
h62 = [int(c) for c in json.loads((ROOT / "data" / "h62.json").read_text())["h62"]]
stored = {r["p"]: r for r in json.loads((ROOT / "data" / "zeta_X.json").read_text()) if r["status"] == "ok"}
primes = [int(a) for a in sys.argv[1:]] or [p for p in stored if p <= 31]
ok_all = True

E = 2**1652 * 3**626 * 23**228 * 131**506
hz = fmpz_poly(h62)
identity, signs = True, set()
for t in range(1, 191):
    dv = int(fmpz_poly(data.specialise(F, t)).discriminant())
    identity &= abs(dv) == E * t**8 * (t * t + 23) ** 22 * int(hz(t)) ** 2
    signs.add(dv > 0)
identity &= len(signs) == 1
print(f"disc_V F = {'+' if signs == {True} else '-'} 2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h62^2 "
      f"(checked at 190 integers; degree 176): {identity}")
ok_all &= identity

for p in primes:
    cert = curve.reduction_certificate(p)
    hp = nmod_poly([c % p for c in h62], p)
    generic = (hp.degree() == 62 and all(m == 1 for _, m in hp.factor()[1])
               and hp.gcd(nmod_poly([0, 23 % p, 0, 1], p)).degree() == 0)
    consistent = (not generic) or cert["types"] == {"A_1": 62}
    kmax_t = 3 if p <= 13 else 2
    ct = [curve.count_points(p, k, fibre="T") for k in range(1, kmax_t + 1)]
    kmax_v = 4 if p <= 31 else 2
    cv = [curve.count_points(p, k, fibre="V") for k in range(1, kmax_v + 1)]
    agree = ct == cv[:kmax_t] and cv == stored[p]["counts"][:kmax_v]
    P = stored[p]["P"]
    weil = curve.is_weil_polynomial(p, P) and stored[p]["P(chi) mod 23"] == 0 and stored[p]["P(chi p) mod 23"] == 0
    print(f"p = {p}: singular points {cert['types']}, delta {cert['delta']}, absolutely irreducible {cert['irreducible']}, "
          f"certified {cert['certified']}; consistent with h62: {consistent}; fibrations and stored counts agree: {agree}; "
          f"Weil and mod 23: {weil}")
    ok_all &= cert["certified"] and consistent and agree and weil
print("ALL CHECKS PASSED" if ok_all else "SOME CHECK FAILED")
sys.exit(0 if ok_all else 1)
