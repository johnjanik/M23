"""Point counts and L-polynomials of the genus 4 curve X of [HJLPPZ26].

X is the normalisation of the closure X0 of F(T,V) = 0 in P^1 x P^1, where F
has bidegree (4,23) and X0 has exactly 62 nodes [HJLPPZ26, Prop. 3.10].  For a
prime p of good reduction,

    #X(F_q) = sum over t in P^1(F_q) of #{distinct roots of F(t, .) in P^1(F_q)}
              + sum over F_q-rational nodes of (b - 1),

where b = 2 for a node with F_q-rational branches and b = 0 otherwise.  A
fibre away from T(T^2+23) = 0 with a repeated root is a fibre through a node.

The output is validated a posteriori: the degree 8 polynomial reconstructed
from N_1..N_4 must be a Weil polynomial.
"""
from __future__ import annotations

from fractions import Fraction

from flint import fmpz_poly, fq_default_ctx, fq_default_poly_ctx, nmod_poly

from . import data


class BadFibre(Exception):
    pass


def _element(ctx, p, k, n):
    digits = []
    for _ in range(k):
        digits.append(n % p)
        n //= p
    return ctx(digits)


def _code(x, p, k):
    out = 0
    for d in reversed(x.to_list()):
        out = out * p + int(d)
    return out


def _projective_root_count(R, cs):
    """(#distinct roots in P^1(F_q), repeated roots as (root, mult); root None = infinity)."""
    full = len(cs) - 1
    deg = full
    while deg >= 0 and cs[deg].is_zero():
        deg -= 1
    if deg < 0:
        raise BadFibre("fibre polynomial vanishes identically")
    roots = R(cs[: deg + 1]).roots() if deg > 0 else []
    multiple = [(r, m) for r, m in roots if m > 1]
    n = len(roots)
    if deg < full:
        n += 1
        if full - deg > 1:
            multiple.append((None, full - deg))
    return n, multiple


def _singular_point_branches(ctx, A, p, b, w):
    """A[i][j] = coefficient of B^i W^j.  None if (b, w) is a smooth point of the
    plane model; otherwise the number (2 or 0) of F_q-rational branches of the node."""
    nb, nw = len(A), len(A[0])
    bp = [ctx(1)]
    for _ in range(nb - 1):
        bp.append(bp[-1] * b)
    wp = [ctx(1)]
    for _ in range(nw - 1):
        wp.append(wp[-1] * w)
    g_b = g_w = h_bb = h_bw = h_ww = ctx(0)
    for i in range(nb):
        for j in range(nw):
            c = A[i][j] % p
            if not c:
                continue
            if i >= 1:
                g_b += c * i * bp[i - 1] * wp[j]
            if j >= 1:
                g_w += c * j * bp[i] * wp[j - 1]
            if i >= 2:
                h_bb += c * i * (i - 1) * bp[i - 2] * wp[j]
            if i >= 1 and j >= 1:
                h_bw += c * i * j * bp[i - 1] * wp[j - 1]
            if j >= 2:
                h_ww += c * j * (j - 1) * bp[i] * wp[j - 2]
    if not (g_b.is_zero() and g_w.is_zero()):
        return None
    disc = h_bw * h_bw - h_bb * h_ww
    if disc.is_zero():
        return -1                      # a singular point that is not a node
    return 2 if disc.is_square() else 0


def count_points(p: int, k: int, coeffs=None, fibre: str = "T", return_nodes: bool = False,
                 strict: bool = True):
    """#X(F_{p^k}) for an odd prime p at which the plane model stays nodal.

    fibre = "T": sum over t of the points in the fibre of the degree 23 map t.
    fibre = "V": sum over v of the points in the fibre of the degree 4 map v
                 (quartic fibre polynomials; much faster, and an independent check).

    Every repeated root of a fibre polynomial is examined: a smooth point of
    the plane model (a ramification point of the projection) counts once, a
    node counts 2 or 0 according to the rationality of its branches.

    If the reduction has a singular point that is not a node (two nodes, or a
    node and a ramification point, coalescing mod p) the count is refused when
    `strict`; otherwise such a point is counted once (as if unibranch), and
    the result must be validated by the Weil-polynomial test.
    """
    coeffs = data.F() if coeffs is None else coeffs
    if fibre == "T":
        A = [list(row) for row in coeffs]                                   # A[i][j]: T^i V^j
    else:
        A = [[coeffs[i][j] for i in range(5)] for j in range(24)]          # A[j][i]: V^j T^i
    nb, nw = len(A), len(A[0])
    ctx = fq_default_ctx(p, k, var="a")
    R = fq_default_poly_ctx(ctx)
    cols = [R([A[i][j] % p for i in range(nb)]) for j in range(nw)]        # in the base variable
    A_rev = [row[::-1] for row in A]                                        # W -> 1/W
    q = p**k
    total = node_count = 0
    seen = bytearray(q)

    def fibre_count(cs, b):
        cnt, multiple = _projective_root_count(R, cs)
        nodes = 0
        for w, _mult in multiple:
            if w is None:
                br = _singular_point_branches(ctx, A_rev, p, b, ctx(0))
            else:
                br = _singular_point_branches(ctx, A, p, b, w)
            if br == -1:
                if strict:
                    raise BadFibre("singular point that is not a node")
            elif br is not None:
                cnt += br - 1
                nodes += 1
        return cnt, nodes

    for n in range(q):
        if seen[n]:
            continue
        b = _element(ctx, p, k, n)
        orbit, x = {n}, b
        for _ in range(k - 1):
            x = x.frobenius()
            orbit.add(_code(x, p, k))
        for c in orbit:
            seen[c] = 1
        cnt, nodes = fibre_count([col.evaluate(b) for col in cols], b)
        total += len(orbit) * cnt
        node_count += len(orbit) * nodes
    # fibre over base = infinity: leading coefficients in the base variable
    lead = [ctx(A[nb - 1][j] % p) for j in range(nw)]
    cnt, multiple = _projective_root_count(R, lead)
    if multiple:
        # a repeated root over infinity; examine it in the chart 1/base
        A_inf = A[::-1]
        A_inf_rev = [row[::-1] for row in A_inf]
        for w, _mult in multiple:
            br = (_singular_point_branches(ctx, A_inf_rev, p, ctx(0), ctx(0)) if w is None
                  else _singular_point_branches(ctx, A_inf, p, ctx(0), w))
            if br == -1:
                if strict:
                    raise BadFibre("singular point that is not a node")
            elif br is not None:
                cnt += br - 1
                node_count += 1
    total += cnt
    return (total, node_count) if return_nodes else total


def frobenius_polynomial(p: int, counts) -> list[int]:
    """Coefficients (descending) of P(T) = det(T - Frob | H^1) from N_1..N_4 (genus 4)."""
    g = 4
    s = [Fraction(p**k + 1 - n) for k, n in enumerate(counts, start=1)]
    e = [Fraction(1)]
    for m in range(1, g + 1):
        acc = Fraction(0)
        for i in range(1, m + 1):
            acc += (-1) ** (i - 1) * e[m - i] * s[i - 1]
        e.append(acc / m)
    if any(x.denominator != 1 for x in e):
        raise ValueError("non-integral coefficients: wrong point counts or bad reduction")
    e = [int(x) for x in e]
    for i in range(g + 1, 2 * g + 1):
        e.append(p ** (i - g) * e[2 * g - i])
    return [(-1) ** i * e[i] for i in range(2 * g + 1)]


def is_weil_polynomial(p: int, poly) -> bool:
    import numpy as np
    roots = np.roots(poly)
    return bool(np.allclose(np.abs(roots), p**0.5, rtol=1e-6))


def reverse_poly(poly: nmod_poly) -> nmod_poly:
    cs = [int(c) for c in poly.coeffs()]
    return nmod_poly(cs[::-1], poly.modulus())


def golay_splitting(poly) -> dict:
    """Factor P mod 2 and search for Q with P = Q * Q^rev (Q of degree 4).

    A rational Lagrangian W in Jac(X)[2] forces such a factorisation, with
    Q(T) = det(T - Frob | W).
    """
    P2 = nmod_poly([c % 2 for c in poly[::-1]], 2)
    _, fac = P2.factor()
    factors = [([int(c) for c in f.coeffs()], m) for f, m in fac]
    solutions = []

    def rec(idx, current):
        if idx == len(fac):
            if current.degree() == 4 and current * reverse_poly(current) == P2:
                solutions.append(current)
            return
        f, m = fac[idx]
        power = nmod_poly([1], 2)
        for _ in range(m + 1):
            if (current * power).degree() <= 4:
                rec(idx + 1, current * power)
            power = power * f
    rec(0, nmod_poly([1], 2))
    pretty = sorted({tuple(int(c) for c in s.coeffs()) for s in solutions})
    return {"P_mod_2_factors": factors, "Q_candidates": pretty, "splits": bool(pretty)}


def bits_to_str(cs) -> str:
    terms = []
    for i in reversed(range(len(cs))):
        if cs[i]:
            terms.append("1" if i == 0 else ("x" if i == 1 else f"x^{i}"))
    return "+".join(terms) or "0"


_ = fmpz_poly
