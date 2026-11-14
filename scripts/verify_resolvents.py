#!/usr/bin/env python3
"""Independent checks of data/golay_resolvents.json (no p-adic arithmetic involved).

For each of f1, f2:
  1. the duad resolvent D satisfies Res_x(f(x), f(z - x)) = 2^23 f(z/2) D(z)^2
     at several integers z (D is a symmetric resolvent, so this certifies it);
  2. for all primes p < BOUND not dividing disc(f) at which both polynomials are
     squarefree, the heptad resolvent H and D have the same factorisation
     pattern mod p (Gassmann equivalence), and the pattern is the one predicted
     from the factorisation pattern of f;
  3. H is irreducible over Q: there are primes where all its factors mod p have
     degree 23, and primes where all have degree 11;
  4. the sum of the seven real roots of f is a root of H (the real roots form a
     heptad) and is not a root of D.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flint import fmpq, fmpq_poly, fmpz_poly, nmod_poly  # noqa: E402
import mpmath  # noqa: E402

BOUND = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "data" / "golay_resolvents.json").read_text())

# predicted cycle types on the 253 heptads, computed from class representatives
from m23golay import invariants, m23  # noqa: E402
from m23golay.perm import cycle_type  # noqa: E402
_system = list(m23.steiner())
PREDICTED = {cycle_type(g): dict(Counter(invariants.cycle_type_on(_system, g)))
             for g in m23.class_representatives().values()}


def pattern(coeffs, p):
    fac = nmod_poly([c % p for c in coeffs], p).factor()[1]
    if any(m > 1 for _, m in fac):
        return None
    return tuple(sorted(f.degree() for f, _ in fac))


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


ok_all = True
for name in ("f1", "f2"):
    f = data[name]["f"]
    H = [int(c) for c in data[name]["R253_heptad"]]
    D = [int(c) for c in data[name]["R253_duad"]]
    fz, Dz = fmpz_poly(f), fmpz_poly(D)
    # 1. symmetric formula
    sym = True
    for z0 in range(-3, 4):
        shifted = fmpz_poly([sum(f[k] * (-1) ** j * __import__("math").comb(k, j) * z0 ** (k - j)
                                 for k in range(j, 24)) for j in range(24)])   # f(z0 - x)
        lhs = fz.resultant(shifted)
        rhs = fmpq(2) ** 23 * fmpq_poly(f)(fmpq(z0, 2)) * fmpq(int(Dz(z0))) ** 2
        sym &= fmpq(int(lhs)) == rhs
    # 2, 3. factorisation patterns
    disc = int(fz.discriminant())
    tested = agree = predicted_ok = 0
    seen = {}
    for p in range(5, BOUND):
        if not is_prime(p) or disc % p == 0:
            continue
        ph, pd, pf = pattern(H, p), pattern(D, p), pattern(f, p)
        if ph is None or pd is None or pf is None:
            continue
        tested += 1
        agree += ph == pd
        predicted_ok += dict(Counter(ph)) == PREDICTED[pf]
        seen.setdefault(pf, p)
    irreducible = (23,) in seen and (1, 11, 11) in seen
    # 4. real heptad
    mpmath.mp.dps = 400
    roots = mpmath.polyroots([mpmath.mpf(c) for c in f[::-1]], maxsteps=2000, extraprec=2000)
    real = [r.real for r in roots if abs(r.imag) < mpmath.mpf(10) ** -100]
    s = mpmath.fsum(real)
    hv = mpmath.polyval([mpmath.mpf(c) for c in H[::-1]], s)
    hd = mpmath.polyval([mpmath.mpf(c * k) for k, c in enumerate(H)][:0:-1], s)
    dv = mpmath.polyval([mpmath.mpf(c) for c in D[::-1]], s)
    dd = mpmath.polyval([mpmath.mpf(c * k) for k, c in enumerate(D)][:0:-1], s)
    newton_h, newton_d = abs(hv / hd), abs(dv / dd)
    print(f"{name}: J5 = {data[name]['J5']}, J7 = {data[name]['J7']}")
    print(f"  duad resolvent = symmetric resolvent at 7 points: {sym}")
    print(f"  primes < {BOUND} tested: {tested}; heptad/duad patterns agree: {agree}; match prediction: {predicted_ok}")
    print(f"  cycle types of f seen: {len(seen)} of 12; H irreducible over Q: {irreducible}")
    print(f"  real roots of f: {len(real)}; Newton step of H at their sum: {mpmath.nstr(newton_h, 5)}"
          f" (of D: {mpmath.nstr(newton_d, 5)})")
    ok = sym and tested == agree == predicted_ok and irreducible and len(real) == 7 \
        and newton_h < mpmath.mpf(10) ** -200 and newton_d > mpmath.mpf(10) ** -20
    ok_all &= ok
print("ALL CHECKS PASSED" if ok_all else "SOME CHECK FAILED")
sys.exit(0 if ok_all else 1)
