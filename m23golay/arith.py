"""Certified Steiner labelling of the roots of an M23-polynomial, and Golay invariants.

Let f in Z[x] be monic of degree 23 with Galois group M23 (natural action), and
let p be a prime with f irreducible mod p.  Then R = Z_p[x]/(f) is the ring of
integers of the unramified extension of degree 23, f splits in R, and its
roots are alpha_i = phi^i(x), i in Z/23, with phi the Frobenius automorphism.
An embedding of the splitting field into Frac(R) identifies Gal(f) with a
conjugate of M23 in Sym(Z/23) containing the shift i -> i+1.  There are exactly
two such conjugates: the automorphism groups of the cyclic Golay codes G^+ and
G^- (golay.py).  Hence the Steiner system carried by the roots is one of two
explicit candidates, and a Galois-invariant of the correct one is a rational
integer of bounded size, which decides between them rigorously.

All computations are exact in (Z/p^N)[x]/(f).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

from flint import fmpz_mod_ctx, fmpz_mod_poly_ctx, nmod_poly

from . import golay

N23 = 23


def _iroot_ceil(a: int, k: int) -> int:
    """Smallest integer r >= 0 with r^k >= a (a >= 0)."""
    if a <= 1:
        return a
    hi = 1 << ((a.bit_length() + k - 1) // k)
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid**k >= a:
            hi = mid
        else:
            lo = mid
    return hi


def root_bound(f) -> Fraction:
    """An upper bound for the absolute values of the roots of monic f (ascending
    integer coefficients): Fujiwara's bound 2 max_k |a_{n-k}|^{1/k}, computed with
    exact integer roots at 8 extra binary digits, so the result is rigorous."""
    n = len(f) - 1
    assert f[-1] == 1
    scale = 256
    best = 0
    for k in range(1, n + 1):
        a = abs(f[n - k])
        if a:
            best = max(best, _iroot_ceil(a * scale**k, k))     # ceil(scale * a^(1/k))
    return Fraction(2 * best, scale)


def inert_primes(f, start: int = 3, count: int = 1, skip=()) -> list[int]:
    """Primes p for which f is irreducible mod p (Frobenius is a 23-cycle)."""
    out, p = [], start
    while len(out) < count:
        if all(p % d for d in range(2, int(p**0.5) + 1)) and p not in skip:
            fp = nmod_poly([c % p for c in f], p)
            if fp.degree() == len(f) - 1:
                fac = fp.factor()[1]
                if len(fac) == 1 and fac[0][1] == 1:
                    out.append(p)
        p += 1
    return out


@dataclass
class Lift:
    """The roots alpha_i = phi^i(x) of f in (Z/p^N)[x]/(f)."""
    f: list
    p: int
    N: int
    ctx: object
    fbar: object
    roots: list

    @property
    def modulus(self) -> int:
        return self.p**self.N

    def const(self, a) -> int:
        """The integer in (-p^N/2, p^N/2] represented by a constant ring element."""
        if a.degree() > 0:
            raise ValueError("element is not a constant: not Frobenius-invariant")
        c = int(a.constant_coefficient()) if a.degree() == 0 else 0
        m = self.modulus
        return c - m if c > m // 2 else c

    def mul(self, a, b):
        return a.mul_mod(b, self.fbar)


def frobenius_lift(f, p: int, N: int) -> Lift:
    """Hensel-lift the Frobenius image of x to precision p^N and return all 23 roots."""
    n = len(f) - 1
    prec, y_int, inv_int = 1, None, None
    # start mod p
    ctx = fmpz_mod_poly_ctx(fmpz_mod_ctx(p))
    fb = ctx(f)
    x = ctx([0, 1])
    y = x.pow_mod(p, fb)
    dfy = fb.derivative().compose_mod(y, fb)
    inv = dfy.inverse_mod(fb)                 # F_p[x]/(f) is a field
    y_int = [int(c) for c in y.coeffs()]
    inv_int = [int(c) for c in inv.coeffs()]
    while prec < N:
        prec = min(2 * prec, N)
        ctx = fmpz_mod_poly_ctx(fmpz_mod_ctx(p**prec))
        fb = ctx(f)
        y, inv = ctx(y_int), ctx(inv_int)
        dfy = fb.derivative().compose_mod(y, fb)
        inv = inv.mul_mod(ctx([2]) - dfy.mul_mod(inv, fb), fb)      # Newton for 1/f'(y)
        y = y - fb.compose_mod(y, fb).mul_mod(inv, fb)                # Newton for f(y) = 0
        dfy = fb.derivative().compose_mod(y, fb)
        inv = inv.mul_mod(ctx([2]) - dfy.mul_mod(inv, fb), fb)
        y_int = [int(c) for c in y.coeffs()]
        inv_int = [int(c) for c in inv.coeffs()]
    if prec == 1 and N == 1:
        pass
    ctx = fmpz_mod_poly_ctx(fmpz_mod_ctx(p**N))
    fb = ctx(f)
    y = ctx(y_int)
    assert fb.compose_mod(y, fb).is_zero(), "Hensel lifting failed"
    roots = [ctx([0, 1])]
    for _ in range(n - 1):
        roots.append(roots[-1].compose_mod(y, fb))               # phi(g(x)) = g(phi(x))
    assert roots[-1].compose_mod(y, fb) == roots[0], "Frobenius does not have order 23"
    return Lift(list(f), p, N, ctx, fb, roots)


def block_sums(lift: Lift, system) -> dict:
    out = {}
    for block in system:
        s = lift.ctx([0])
        for i in block:
            s = s + lift.roots[i]
        out[block] = s
    return out


def power_sum(lift: Lift, sums, k: int):
    """sum over blocks of (block sum)^k, as a ring element."""
    total = lift.ctx([0])
    for s in sums.values():
        total = total + s.pow_mod(k, lift.fbar)
    return total


def elementary_of_block(lift: Lift, block, k: int):
    """e_k of the roots in a block, as a ring element."""
    e = [lift.ctx([1])] + [lift.ctx([0])] * k
    for i in block:
        a = lift.roots[i]
        for j in range(k, 0, -1):
            e[j] = e[j] + lift.mul(e[j - 1], a)
    return e[k]


def invariant_sum(lift: Lift, system, k: int) -> int:
    """sum over blocks of e_k(block): J_7 for k = 7, and J_5 for k = 5 (each
    five-subset of a heptad lies in exactly one heptad)."""
    total = lift.ctx([0])
    for block in system:
        total = total + elementary_of_block(lift, block, k)
    return lift.const(total)


def orbit_polynomial(lift: Lift, sums) -> list[int]:
    """Ascending integer coefficients of prod (Z - s) over the given block sums."""
    coeffs = [lift.ctx([1])]                       # ascending in Z
    for s in sums:
        new = [lift.ctx([0])] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i + 1] = new[i + 1] + c
            new[i] = new[i] - lift.mul(c, s)
        coeffs = new
    return [lift.const(c) for c in coeffs]


@dataclass
class Labelling:
    p: int
    N: int
    sign: int            # +1: the roots carry the G^+ system; -1: the G^- system
    t5: dict             # sign -> p-adic value of sum_H s_H^5 (symmetric residue)
    bound_t5: int
    lift: Lift

    @property
    def system(self):
        return golay.heptads(self.sign)


def steiner_labelling(f, p: int | None = None, extra_digits: int = 30) -> Labelling:
    """Decide which cyclic Golay system the Frobenius-ordered roots of f carry.

    Rigorous given Gal(f) = M23: the invariant T5 = sum_H (sum_{i in H} alpha_i)^5
    of the true system is an integer with |T5| <= 253 (7 M)^5, M a root bound.
    We compute both candidates modulo p^N > 2 * 253 (7M)^5 * 10^extra_digits and
    require that exactly one of them is an integer within the bound.
    """
    if p is None:
        p = inert_primes(f)[0]
    M = root_bound(f)
    bound = int(253 * (7 * M) ** 5) + 1
    N = 1
    while p**N <= 2 * bound * 10**extra_digits:
        N += 1
    lift = frobenius_lift(f, p, N)
    t5, ok = {}, {}
    for sign in (+1, -1):
        sums = block_sums(lift, golay.heptads(sign))
        t5[sign] = lift.const(power_sum(lift, sums, 5))
        ok[sign] = abs(t5[sign]) <= bound
    if ok[+1] == ok[-1]:
        raise RuntimeError(f"undecided at p = {p}: {ok}")
    return Labelling(p, N, +1 if ok[+1] else -1, t5, bound, lift)


def relift(f, lab: Labelling, coeff_bound: int, extra_digits: int = 30) -> Lift:
    """A lift at the same prime with p^N > 2 * coeff_bound * 10^extra_digits."""
    N = 1
    while lab.p**N <= 2 * coeff_bound * 10**extra_digits:
        N += 1
    return frobenius_lift(f, lab.p, N)


def heptad_resolvent(f, lab: Labelling) -> list[int]:
    """R_253(Z) = prod over heptads (Z - sum of the roots in the heptad), ascending coeffs."""
    M = root_bound(f)
    bound = int((1 + 7 * M) ** 253) + 1
    lift = relift(f, lab, bound)
    sums = block_sums(lift, lab.system)
    coeffs = orbit_polynomial(lift, [sums[b] for b in sorted(sums, key=sorted)])
    assert all(abs(c) <= bound for c in coeffs)
    return coeffs


def duad_resolvent(f, lab: Labelling) -> list[int]:
    """prod over i < j of (Z - alpha_i - alpha_j); a symmetric resolvent."""
    M = root_bound(f)
    bound = int((1 + 2 * M) ** 253) + 1
    lift = relift(f, lab, bound)
    sums = [lift.roots[i] + lift.roots[j] for i, j in itertools.combinations(range(N23), 2)]
    coeffs = orbit_polynomial(lift, sums)
    assert all(abs(c) <= bound for c in coeffs)
    return coeffs


def golay_invariants(f, lab: Labelling) -> dict:
    """The integers J_5(f) and J_7(f)."""
    M = root_bound(f)
    out = {}
    for k, nterms in ((5, 5313), (7, 253)):
        lift = relift(f, lab, int(nterms * M**k) + 1)
        out[f"J{k}"] = invariant_sum(lift, lab.system, k)
    return out


def tschirnhaus(f, lab: Labelling):
    """P in Q[X], deg <= 22, with P(alpha_i) = D_i(alpha) = dJ7/dx_i (alpha).

    D_0 = sum over heptads H containing 0 of prod_{j in H, j != 0} alpha_j lies in
    Z[alpha_0]^dual, so disc(f) * P has integer coefficients.  Returns
    (numerators, denominator) with P = sum numerators[k] X^k / denominator.
    """
    from flint import fmpz_poly
    M = root_bound(f)
    n = N23
    disc = abs(int(fmpz_poly(f).discriminant()))
    # coefficients of P w.r.t. the power basis are bounded via Lagrange interpolation;
    # use a generous bound and verify the result exactly afterwards.
    bound = disc * int(77 * M**6 + 1) * int((1 + M) ** 22 + 1) * 2**23
    lift = relift(f, lab, bound)
    d0 = lift.ctx([0])
    for h in lab.system:
        if 0 in h:
            prod = lift.ctx([1])
            for j in h:
                if j != 0:
                    prod = lift.mul(prod, lift.roots[j])
            d0 = d0 + prod
    # d0 is an element of (Z/p^N)[x]/(f) with x = alpha_0: its coefficients are those of P.
    m = lift.modulus
    cs = [int(c) for c in d0.coeffs()] + [0] * n
    num = []
    for c in cs[:n]:
        v = (c * disc) % m
        num.append(v - m if v > m // 2 else v)
    return num, disc


def frobenius_sign(f, p: int, t5: int) -> int:
    """Sign (+1/-1) of the 23-cycle Frob_p from the known integer T5 (mod p only).

    Returns 0 if the two candidates cannot be separated modulo p.
    """
    lift = frobenius_lift(f, p, 1)
    vals = {}
    for sign in (+1, -1):
        sums = block_sums(lift, golay.heptads(sign))
        vals[sign] = lift.const(power_sum(lift, sums, 5)) % p
    target = t5 % p
    hits = [s for s in (+1, -1) if vals[s] == target]
    return hits[0] if len(hits) == 1 else 0
