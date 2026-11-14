#!/usr/bin/env python3
"""Certify that the plane model F(T,V)=0 reduces well at given primes (default 7, 11).

For each prime p:
  1. the identity disc_V(F) = +-2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h62(T)^2
     is verified over Z (h62 from data/h62.json); p does not divide the constant,
     deg h62 mod p = 62, and h62 mod p is squarefree and prime to T(T^2+23).
     Hence the only singular points lie above the roots of h62, and the fibres
     above 0, the roots of T^2+23 and infinity are as in characteristic 0.
  2. above each root of h62 (in its residue field) the fibre has exactly one
     repeated root, of multiplicity 2, at which the model has a node.
So the reduction has exactly 62 nodes, geometric genus 66 - 62 = 4, and X has
good reduction at p.
  3. #X(F_{p^k}), k = 1..4, by both fibrations (degree 23 and degree 4) agree.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flint import fmpz_poly, fq_default_ctx, fq_default_poly_ctx, nmod_poly  # noqa: E402
from m23golay import curve, data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
F = data.F()
h62 = [int(c) for c in json.loads((ROOT / "data" / "h62.json").read_text())["h62"]]
primes = [int(a) for a in sys.argv[1:]] or [7, 11]
ok_all = True

# 0. the discriminant identity over Z, at 190 integers (degree 176 in T):
#    disc_V F = sign * 2^1652 3^626 23^228 131^506 * T^8 (T^2+23)^22 h62(T)^2
E = 2**1652 * 3**626 * 23**228 * 131**506
hz = fmpz_poly(h62)
signs = set()
identity = True
for t in range(1, 191):
    dv = int(fmpz_poly(data.specialise(F, t)).discriminant())
    rhs = E * t**8 * (t * t + 23) ** 22 * int(hz(t)) ** 2
    identity &= abs(dv) == rhs
    signs.add(dv > 0)
identity &= len(signs) == 1
print(f"disc_V F = {'+' if signs == {True} else '-'} 2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h62^2 "
      f"(checked at 190 integers; degree 176): {identity}")
ok_all &= identity

for p in primes:
    good = p not in (2, 3, 23, 131)
    hp = nmod_poly([c % p for c in h62], p)
    fac = hp.factor()[1]
    squarefree = all(m == 1 for _, m in fac) and hp.degree() == 62
    coprime = hp.gcd(nmod_poly([0, 23 % p, 0, 1], p)).degree() == 0
    print(f"p = {p}: p does not divide the constant: {good}; "
          f"h62 mod p squarefree of degree 62: {squarefree}; prime to T(T^2+23): {coprime}")
    # 2. nodes above the roots of h62
    nodes = 0
    for d in sorted({f.degree() for f, _ in fac}):
        cd = fq_default_ctx(p, d, var="a")
        Rd = fq_default_poly_ctx(cd)
        colsd = [Rd([F[i][j] % p for i in range(5)]) for j in range(24)]
        A = [list(row) for row in F]
        for t, _m in Rd([c % p for c in h62]).roots():
            # exact degree d?
            x, deg = t, 1
            while True:
                x = x.frobenius()
                if x == t:
                    break
                deg += 1
            if deg != d:
                continue
            cnt, multiple = curve._projective_root_count(Rd, [c.evaluate(t) for c in colsd])
            assert len(multiple) == 1 and multiple[0][1] == 2 and multiple[0][0] is not None, multiple
            br = curve._singular_point_branches(cd, A, p, t, multiple[0][0])
            assert br in (0, 2), "not a node"
            nodes += 1
    print(f"       nodes above the roots of h62: {nodes}")
    # 3. two fibrations
    ct = [curve.count_points(p, kk, fibre="T") for kk in (1, 2, 3, 4)]
    cv = [curve.count_points(p, kk, fibre="V") for kk in (1, 2, 3, 4)]
    P = curve.frobenius_polynomial(p, cv)
    print(f"       point counts {cv}; both fibrations agree: {ct == cv}; Weil polynomial: {curve.is_weil_polynomial(p, P)}")
    ok = good and squarefree and coprime and nodes == 62 and ct == cv and curve.is_weil_polynomial(p, P)
    ok_all &= ok
print("ALL CHECKS PASSED" if ok_all else "SOME CHECK FAILED")
sys.exit(0 if ok_all else 1)
