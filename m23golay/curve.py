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


def _taylor_shift(cs, a):
    """Coefficients of c(x + a), in place, for a list of field elements."""
    n = len(cs)
    for i in range(n - 1):
        for j in range(n - 2, i - 1, -1):
            cs[j] = cs[j] + a * cs[j + 1]
    return cs


def _local_equation(ctx, A, p, b, w):
    """The polynomial F(b + x, w + y) as a dict {(i, j): coefficient of x^i y^j}.
    A[i][j] is the coefficient of B^i W^j."""
    nb, nw = len(A), len(A[0])
    rows = [_taylor_shift([ctx(A[i][j] % p) for j in range(nw)], w) for i in range(nb)]
    for j in range(nw):
        col = _taylor_shift([rows[i][j] for i in range(nb)], b)
        for i in range(nb):
            rows[i][j] = col[i]
    return {(i, j): rows[i][j] for i in range(nb) for j in range(nw) if not rows[i][j].is_zero()}


def _shift_y(f, r):
    """f(x, y + r) for a dict polynomial."""
    by_x = {}
    for (i, j), c in f.items():
        by_x.setdefault(i, {})[j] = c
    out = {}
    for i, col in by_x.items():
        n = max(col) + 1
        zero = r - r
        cs = _taylor_shift([col.get(j, zero) for j in range(n)], r)
        for j, c in enumerate(cs):
            if not c.is_zero():
                out[(i, j)] = c
    return out


def resolve_point(f, R, depth: int = 0):
    """Resolve the plane curve germ f = 0 at the origin by blowing up points.

    f: dict {(i, j): c} over F_q with f(0, 0) = 0.  Returns (places, delta, label):
      places  number of F_q-rational points of the normalization above the origin
              (always exact);
      delta   the delta invariant, or None when a repeated tangent direction is not
              F_q-rational (the count of places is still exact in that case);
      label   "smooth", "A_k", "ordinary m-fold", or "other".
    A point of multiplicity m contributes m(m-1)/2 to delta, plus the contributions of
    the points of the strict transform on the exceptional line, which are the roots of
    the tangent cone.  A simple root of the tangent cone is a smooth point of the strict
    transform.  Characteristic different from 2 is assumed only for the labels.
    """
    if depth > 80:
        raise BadFibre("resolution does not terminate (non-reduced germ?)")
    m = min(i + j for (i, j) in f)
    if m == 0:
        raise BadFibre("the point is not on the curve")
    if m == 1:
        return 1, 0, "smooth"
    zero = next(iter(f.values())) * 0
    phi = [f.get((m - j, j), zero) for j in range(m + 1)]            # tangent cone in y1 = y/x
    deg = max(j for j in range(m + 1) if not phi[j].is_zero())
    places, delta, exact = 0, m * (m - 1) // 2, True
    sub_labels, directions, repeated = [], 0, 0
    if deg >= 1:
        _lc, fac = R(phi[: deg + 1]).factor()
        for g, mult in fac:
            directions += g.degree()
            if mult > 1:
                repeated += 1
            if g.degree() == 1:
                if mult == 1:
                    places += 1
                    continue
                r = -g.coeffs()[0]
                h = {(i + j - m, j): c for (i, j), c in f.items()}     # y = x y1
                pl, dl, lab = resolve_point(_shift_y(h, r), R, depth + 1)
                places += pl
                sub_labels.append(lab)
                if dl is None:
                    exact = False
                else:
                    delta += dl
            elif mult > 1:
                exact = False                                          # repeated, not rational
    if deg < m:                                                        # the direction x = 0
        directions += 1
        if m - deg == 1:
            places += 1
        else:
            repeated += 1
            h = {(i, i + j - m): c for (i, j), c in f.items()}         # x = x1 y
            pl, dl, lab = resolve_point(h, R, depth + 1)
            places += pl
            sub_labels.append(lab)
            if dl is None:
                exact = False
            else:
                delta += dl
    if repeated == 0:
        label = "A_1" if m == 2 else f"ordinary {m}-fold"
    elif m == 2 and exact and len(sub_labels) == 1:
        sub = sub_labels[0]
        if sub == "smooth":
            label = "A_2"
        elif sub.startswith("A_"):
            label = f"A_{int(sub[2:]) + 2}"
        else:
            label = "other"
    else:
        label = "other"
    return places, (delta if exact else None), label


def _is_singular(f) -> bool:
    return (0, 0) not in f and (1, 0) not in f and (0, 1) not in f


def _fibre_tools(p, k, coeffs, fibre):
    if fibre == "T":
        A = [list(row) for row in coeffs]                                   # A[i][j]: T^i V^j
    else:
        A = [[coeffs[i][j] for i in range(len(coeffs))] for j in range(len(coeffs[0]))]
    ctx = fq_default_ctx(p, k, var="a")
    R = fq_default_poly_ctx(ctx)
    nb, nw = len(A), len(A[0])
    cols = [R([A[i][j] % p for i in range(nb)]) for j in range(nw)]        # in the base variable
    charts = {
        (False, False): A,
        (False, True): [row[::-1] for row in A],                           # W -> 1/W
        (True, False): A[::-1],                                            # B -> 1/B
        (True, True): [row[::-1] for row in A[::-1]],
    }
    return A, ctx, R, cols, charts


def _point_contribution(ctx, R, charts, p, b, w, b_inf=False):
    """(number of rational places, delta, label) at the point (b, w) of the plane model;
    w is None for W = infinity, and b_inf marks B = infinity (then b is ignored)."""
    A = charts[(b_inf, w is None)]
    f = _local_equation(ctx, A, p, ctx(0) if b_inf else b, ctx(0) if w is None else w)
    if not _is_singular(f):
        return 1, 0, "smooth"
    return resolve_point(f, R)


def _count_chunk(args):
    """Contribution of the base points with codes in [lo, hi) to #X(F_{p^k}).

    Stateless: a base point is processed only if its code is the smallest in
    its Frobenius orbit, and is then weighted by the orbit length.
    """
    p, k, coeffs, fibre, lo, hi = args
    A, ctx, R, cols, charts = _fibre_tools(p, k, coeffs, fibre)
    nb, nw = len(A), len(A[0])
    total = singular = 0

    def fibre_count(cs, b, b_inf):
        cnt, multiple = _projective_root_count(R, cs)
        sing = 0
        for w, _mult in multiple:
            places, _delta, label = _point_contribution(ctx, R, charts, p, b, w, b_inf)
            if label != "smooth":
                cnt += places - 1
                sing += 1
        return cnt, sing

    for n in range(lo, hi):
        b = _element(ctx, p, k, n)
        size, x, minimal = 1, b, True
        for _ in range(k - 1):
            x = x.frobenius()
            c = _code(x, p, k)
            if c == n:
                break
            if c < n:
                minimal = False
                break
            size += 1
        if not minimal:
            continue
        cnt, sing = fibre_count([col.evaluate(b) for col in cols], b, False)
        total += size * cnt
        singular += size * sing
    if lo == 0:                                       # the fibre over base = infinity
        lead = [ctx(A[nb - 1][j] % p) for j in range(nw)]
        cnt, sing = fibre_count(lead, ctx(0), True)
        total += cnt
        singular += sing
    return total, singular


def count_points(p: int, k: int, coeffs=None, fibre: str = "T", return_nodes: bool = False,
                 processes: int = 1, chunk: int = 250_000):
    """#X(F_{p^k}): the number of F_{p^k}-points of the normalization of the plane model.

    fibre = "T": sum over t of the points in the fibre of the degree 23 map t.
    fibre = "V": sum over v of the points in the fibre of the degree 4 map v
                 (quartic fibre polynomials; much faster, and an independent check).

    Every repeated root of a fibre polynomial is examined.  A smooth point of the
    plane model counts once.  A singular point counts as many times as there are
    rational places above it, found by resolving the germ (resolve_point): 2 or 0
    for a node, 1 for a cusp, 2 or 0 for a tacnode, and so on.  The count is exact
    whenever the reduction of the model is reduced; that it has geometric genus 4
    (so that the normalization is the reduction of X) is certified separately by
    singular_locus.

    With processes > 1 the base points are split into chunks handled by
    short-lived worker processes (this also bounds memory use).
    """
    coeffs = data.F() if coeffs is None else coeffs
    q = p**k
    jobs = [(p, k, coeffs, fibre, lo, min(lo + chunk, q)) for lo in range(0, q, chunk)]
    if processes > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(processes, maxtasksperchild=1) as pool:
            parts = pool.map(_count_chunk, jobs, chunksize=1)
    else:
        parts = [_count_chunk(j) for j in jobs]
    total = sum(t for t, _ in parts)
    singular = sum(nc for _, nc in parts)
    return (total, singular) if return_nodes else total


def singular_locus(p: int, coeffs, t_polys) -> dict:
    """All singular points of the plane model mod p over the algebraic closure.

    t_polys: integer polynomials (ascending coefficients) whose roots contain the
    T-coordinates of every singular point; T = infinity is always examined.  For the
    model F of X these are T, T^2 + 23 and h62 (the discriminant identity).

    Returns {"delta": total delta invariant, "types": {label: number of geometric
    points}, "irreducible": bool}.  The geometric genus of the reduction is then
    (a-1)(b-1) - delta for a model of bidegree (a, b), provided it is absolutely
    irreducible; `irreducible` is True when a fibre certifies that (a fibre polynomial
    of full degree that is irreducible over F_p; absolute irreducibility follows
    because the two degrees 4 and 23 are coprime).
    """
    from collections import Counter
    types, delta = Counter(), 0
    seen_factors = set()
    work = []                                           # (degree d of t, factor) ; None = infinity
    for poly in t_polys:
        up = nmod_poly([c % p for c in poly], p)
        if up.degree() < 1:
            continue
        for g, _m in up.factor()[1]:
            key = tuple(int(c) for c in g.coeffs())
            if key not in seen_factors:
                seen_factors.add(key)
                work.append((g.degree(), key))
    work.append((1, None))
    for d, key in work:
        # the fibre over one root t of the factor; its d conjugates behave alike
        A, ctx, R, cols, charts = _fibre_tools(p, d, coeffs, "T")
        if key is None:
            b, b_inf = ctx(0), True
            cs = [ctx(A[len(A) - 1][j] % p) for j in range(len(A[0]))]
        else:
            b, b_inf = R(list(key)).roots()[0][0], False
            cs = [col.evaluate(b) for col in cols]
        full = len(cs) - 1
        deg = full
        while deg >= 0 and cs[deg].is_zero():
            deg -= 1
        if deg < 0:
            raise BadFibre("fibre polynomial vanishes identically")
        found = []                                      # (number of geometric points, delta, label)
        degrees = set()                                 # degrees e > 1 of repeated factors
        if deg >= 1:
            for g, m in R(cs[: deg + 1]).factor()[1]:
                if m > 1 and g.degree() == 1:
                    places, dl, label = _point_contribution(ctx, R, charts, p, b, -g.coeffs()[0], b_inf)
                    found.append((d, dl, label))
                elif m > 1:
                    degrees.add(g.degree())
        if full - deg > 1:
            places, dl, label = _point_contribution(ctx, R, charts, p, b, None, b_inf)
            found.append((d, dl, label))
        for e in sorted(degrees):
            # repeated roots of degree e over F_p(t): work in F_{p^(d e)}, one root t2 of the factor
            A2, ctx2, R2, cols2, charts2 = _fibre_tools(p, d * e, coeffs, "T")
            b2 = R2(list(key)).roots()[0][0]
            cs2 = [col.evaluate(b2) for col in cols2]
            deg2 = max(j for j in range(len(cs2)) if not cs2[j].is_zero())
            done = set()
            for r, m in R2(cs2[: deg2 + 1]).roots():
                if m == 1 or r in done:
                    continue
                orbit, x = [], r
                while x not in orbit:
                    orbit.append(x)
                    for _ in range(d):                  # the generator of Gal over F_p(t2)
                        x = x.frobenius()
                done.update(orbit)
                if len(orbit) != e:
                    continue                            # belongs to another degree
                places, dl, label = _point_contribution(ctx2, R2, charts2, p, b2, r, False)
                found.append((d * e, dl, label))
        for mult, dl, label in found:
            if label == "smooth":
                continue
            if dl is None:
                raise BadFibre(f"singular point of type '{label}' with a non-rational repeated direction")
            delta += mult * dl
            types[label] += mult
    # Irreducibility.  If some fibre F(T, v0), v0 in F_{p^k}, is irreducible of full degree in T
    # and the coefficients of F as a polynomial in T have no common factor in V, then F is
    # irreducible over F_{p^k}, hence over F_p; it is then absolutely irreducible because its
    # conjugate factors over the algebraic closure would have equal bidegrees dividing (4, 23).
    content = nmod_poly([0], p)
    for i in range(len(coeffs)):
        content = content.gcd(nmod_poly([c % p for c in coeffs[i]], p))
    irreducible = False
    if content.degree() == 0:
        for k in (1, 3, 5, 7):
            A, ctx, R, cols, charts = _fibre_tools(p, k, coeffs, "V")
            for n in range(min(p**k, 400)):
                v = _element(ctx, p, k, n)
                poly = R([col.evaluate(v) for col in cols])
                if poly.degree() == len(A[0]) - 1 and poly.is_irreducible():
                    irreducible = True
                    break
            if irreducible:
                break
    return {"delta": delta, "types": dict(sorted(types.items())), "irreducible": irreducible}


def _resultant(f, g, one):
    """Resultant of two univariate polynomials over a finite field (Euclid)."""
    res = one
    while True:
        n, m = f.degree(), g.degree()
        if m < 0:
            return one - one
        if m == 0:
            return res * g.leading_coefficient() ** n
        r = divmod(f, g)[1]
        dr = r.degree()
        if dr < 0:
            return one - one
        res = res * g.leading_coefficient() ** (n - dr)
        if (n * m) % 2:
            res = -res
        f, g = g, r


def resultant_T(p: int, coeffs) -> list[int]:
    """Res_V(F, dF/dV) mod p as a polynomial in T (ascending residues), with the formal
    degrees (n, n-1) in V.  Its roots are the T-coordinates of all points of the model
    at which the projection to the T-line is not etale, including those at V = infinity;
    in particular of all singular points with finite T.  Evaluation and interpolation
    over an extension field, so that small p are allowed."""
    n_t, n_v = len(coeffs) - 1, len(coeffs[0]) - 1
    if n_v % p == 0:
        raise BadFibre("the characteristic divides the degree in V")
    need = n_t * (2 * n_v - 1) + 1
    k = 1
    while p**k < need + n_t + 8:
        k += 1
    ctx = fq_default_ctx(p, k, var="a")
    R = fq_default_poly_ctx(ctx)
    cols = [R([coeffs[i][j] % p for i in range(n_t + 1)]) for j in range(n_v + 1)]
    if all(c % p == 0 for c in (coeffs[i][n_v] for i in range(n_t + 1))):
        raise BadFibre("the degree in V drops mod p")
    xs, ys, n = [], [], 0
    while len(xs) < need:
        if n >= p**k:
            raise BadFibre("not enough interpolation points")
        t = _element(ctx, p, k, n)
        n += 1
        f = R([c.evaluate(t) for c in cols])
        if f.degree() < n_v:
            continue                                   # the leading coefficient vanishes here
        xs.append(t)
        ys.append(_resultant(f, f.derivative(), ctx(1)))
    x = R([0, 1])
    master = R([1])
    for t in xs:
        master = master * (x - R([t]))
    total = R([0])
    for t, y in zip(xs, ys):
        if y.is_zero():
            continue
        part = divmod(master, x - R([t]))[0]
        total = total + part * R([y / part.evaluate(t)])
    out = []
    for c in total.coeffs():
        code = _code(c, p, k)
        if code >= p:
            raise BadFibre("interpolated resultant is not defined over the prime field")
        out.append(code)
    return out


def reduction_certificate(p: int, coeffs=None, genus: int = 4) -> dict:
    """Certificate that the normalization of the plane model mod p is the good reduction
    of the curve: the model mod p is absolutely irreducible and the delta invariants of
    its singular points add up to (a-1)(b-1) - genus.  (Then the normalization of the
    integral model is smooth; see the paper, Lemma on reduction.)"""
    coeffs = data.F() if coeffs is None else coeffs
    try:
        res = resultant_T(p, coeffs)
    except BadFibre as exc:
        return {"certified": False, "reason": str(exc)}
    if not any(res):
        return {"certified": False, "reason": "the discriminant vanishes identically mod p"}
    loc = singular_locus(p, coeffs, [res])
    arithmetic = (len(coeffs) - 2) * (len(coeffs[0]) - 2)
    loc["arithmetic_genus"] = arithmetic
    loc["geometric_genus"] = arithmetic - loc["delta"]
    loc["certified"] = bool(loc["irreducible"] and loc["geometric_genus"] == genus)
    return loc


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
