"""The Nielsen class Ni(2A, 23A, 23B) of M23 and its real structure.

Conventions follow [HJLPPZ26]: right action, triples (g1, g2, g3) with
g1 g2 g3 = 1, g1 in 2A, g2 in 23A, g3 in 23B, where 23A is *defined* as the
class of the published g2.  Inner classes are normalised to have second entry
equal to the published g2; they are then parametrised by the involutions a
with (a g2)^-1 in 23B, modulo conjugation by <g2> = C(g2).

Complex conjugation acts on the class through
    (g1, g2, g3) -> (g1^-1, g3^-1, g2^-1)        [HJLPPZ26, Section 3.1].
"""
from __future__ import annotations

from functools import lru_cache

from . import m23
from .perm import conj, fixed_points, from_cycles, identity, inv, mul, mul_many, power

N = 23


def _canonical(a):
    """Smallest <g2>-conjugate of the involution a."""
    best, cur = a, a
    for _ in range(N - 1):
        cur = conj(cur, m23.G2)
        if cur < best:
            best = cur
    return best


@lru_cache(maxsize=None)
def nielsen_class() -> list[tuple]:
    """Canonical representatives (g1, g2, g3) of the inner classes, sorted."""
    sign_a = m23.sign_of_23_element(m23.G2)
    good = []
    for a in m23.involutions():
        prod = mul(a, m23.G2)
        if m23.signature(prod) != (0, 0, 0):
            continue                      # a g2 is not a 23-cycle
        # g3 = (a g2)^-1 in 23B  <=>  a g2 in 23A
        if m23.sign_of_23_element(prod) == sign_a:
            good.append(a)
    assert len(good) % N == 0
    reps = sorted({_canonical(a) for a in good})
    assert len(reps) * N == len(good), "C(g2) does not act freely"
    return [(a, m23.G2, inv(mul(a, m23.G2))) for a in reps]


def normalise(triple) -> tuple:
    """Conjugate a triple with g1 g2 g3 = 1 and g2 in 23A to canonical form."""
    g1, g2, g3 = triple
    assert mul_many(g1, g2, g3) == identity(N)
    h = m23.conjugator_in_group(g2, m23.G2)
    if h is None:
        raise ValueError("second entry is not in class 23A")
    a = _canonical(conj(g1, h))
    return (a, m23.G2, inv(mul(a, m23.G2)))


def index_of(triple) -> int:
    return nielsen_class().index(normalise(triple))


def complex_conjugate(triple) -> tuple:
    g1, g2, g3 = triple
    return (inv(g1), inv(g3), inv(g2))


def conjugation_permutation() -> list[int]:
    """Image of each class under complex conjugation."""
    return [index_of(complex_conjugate(t)) for t in nielsen_class()]


def real_structure(triple):
    """The involution c0 with c0 g1 c0 = g1^-1, c0 g2 c0 = g3^-1, c0 g3 c0 = g2^-1.

    Returns None if the cover is not real.  For a real cover, c0 is the action
    of complex conjugation on the fibre over a real base point in (0, sqrt 23).
    """
    g1, g2, g3 = triple
    h = m23.conjugator_in_group(g2, inv(g3))
    if h is None:
        return None
    for k in range(N):
        c0 = mul(power(g2, k), h)
        if conj(g1, c0) == inv(g1):
            assert conj(g2, c0) == inv(g3) and conj(g3, c0) == inv(g2)
            assert mul(c0, c0) == identity(N) and c0 != identity(N)
            return c0
    return None


def real_components(triple, c0) -> int:
    """Number of connected components of X(R) for a real cover.

    Real points over t in (0, sqrt 23) are Fix(c0) =: A; over t < 0 (fibre
    identified along the upper half circle) they are Fix(c0 g1) =: B.  Arcs of
    X(R) join x in A to x^g3 in B through t = infinity, and through t = 0 they
    join x_A to x_B when g1 fixes x, and x to x^g1 inside A or inside B
    otherwise (a fold at a real ramification point).
    """
    g1, g2, g3 = triple
    a_pts = fixed_points(c0)
    b_pts = fixed_points(mul(c0, g1))
    adj: dict = {("A", x): [] for x in a_pts}
    adj.update({("B", x): [] for x in b_pts})

    def edge(u, v):
        adj[u].append(v)
        adj[v].append(u)

    for x in a_pts:
        assert g3[x] in b_pts
        edge(("A", x), ("B", g3[x]))
    for x in a_pts:
        if g1[x] == x:
            assert x in b_pts
            edge(("A", x), ("B", x))
        elif x < g1[x]:
            assert g1[x] in a_pts
            edge(("A", x), ("A", g1[x]))
    for x in b_pts:
        if g1[x] != x and x < g1[x]:
            assert g1[x] in b_pts
            edge(("B", x), ("B", g1[x]))
    assert all(len(v) == 2 for v in adj.values())
    seen, comps = set(), 0
    for start in adj:
        if start in seen:
            continue
        comps += 1
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u])
    return comps


# ---------------------------------------------------------------------------
# Cross-references to the published tables
# ---------------------------------------------------------------------------

def _c(*cyc):
    return from_cycles([cyc])


# seven_nielsen_class_representatives.m in github.com/shaowuz/m23isgalois
# (commit f3866d1): entries <x, y, z> with z*y*x = 1 and y fixed.
HJLPPZ_FILE_Y = _c(1, 2, 11, 10, 16, 9, 6, 3, 23, 19, 20, 14, 21, 17, 4, 8, 22, 5, 18, 15, 13, 7, 12)
HJLPPZ_FILE = [
    (from_cycles([(1, 11), (2, 23), (3, 8), (4, 16), (5, 21), (7, 20), (15, 19), (18, 22)]),
     _c(1, 2, 3, 4, 10, 11, 12, 7, 19, 18, 8, 6, 9, 16, 17, 21, 22, 5, 14, 20, 13, 15, 23)),
    (from_cycles([(3, 17), (5, 11), (7, 18), (8, 16), (9, 21), (12, 19), (14, 22), (15, 23)]),
     _c(1, 12, 23, 18, 13, 15, 3, 21, 16, 4, 17, 6, 9, 14, 8, 10, 11, 22, 20, 19, 7, 5, 2)),
    (from_cycles([(1, 4), (2, 14), (5, 23), (6, 21), (8, 11), (9, 12), (10, 19), (17, 20)]),
     _c(1, 17, 19, 11, 4, 12, 16, 10, 23, 22, 8, 2, 20, 21, 9, 7, 13, 15, 18, 5, 3, 6, 14)),
    (from_cycles([(1, 19), (2, 12), (3, 10), (5, 11), (6, 20), (7, 23), (9, 22), (13, 17)]),
     _c(1, 23, 13, 21, 14, 20, 9, 8, 4, 17, 15, 18, 5, 2, 7, 3, 11, 22, 16, 10, 6, 19, 12)),
    (from_cycles([(1, 14), (2, 19), (4, 10), (5, 23), (7, 20), (8, 13), (12, 18), (15, 21)]),
     _c(1, 20, 13, 4, 11, 2, 23, 22, 8, 15, 14, 12, 5, 3, 6, 9, 16, 10, 17, 21, 18, 7, 19)),
    (from_cycles([(2, 22), (4, 21), (5, 15), (6, 7), (8, 9), (10, 23), (12, 18), (14, 17)]),
     _c(1, 12, 5, 18, 7, 9, 4, 14, 21, 17, 20, 19, 23, 11, 2, 8, 16, 10, 3, 6, 13, 15, 22)),
    (from_cycles([(1, 18), (2, 13), (4, 19), (5, 17), (6, 14), (9, 11), (12, 22), (16, 20)]),
     _c(1, 5, 21, 14, 9, 2, 15, 18, 12, 8, 4, 23, 3, 6, 20, 10, 11, 16, 19, 17, 22, 7, 13)),
]

# [Janik26, Table 1]: representatives (g2; g3) with common first entry g1.
JANIK_TABLE = [
    (_c(1, 13, 19, 12, 4, 18, 10, 23, 16, 2, 15, 3, 20, 21, 7, 5, 14, 6, 11, 9, 22, 8, 17),
     _c(1, 17, 3, 19, 13, 11, 6, 14, 21, 7, 5, 20, 8, 18, 16, 2, 4, 12, 15, 23, 10, 22, 9)),
    (_c(1, 8, 17, 10, 19, 14, 21, 11, 5, 6, 4, 13, 20, 22, 2, 18, 3, 15, 12, 23, 9, 16, 7),
     _c(1, 20, 13, 16, 9, 2, 18, 23, 12, 19, 10, 17, 3, 22, 7, 4, 6, 21, 14, 15, 8, 11, 5)),
    (_c(1, 3, 21, 22, 8, 15, 20, 9, 5, 4, 6, 19, 23, 18, 11, 16, 13, 14, 12, 17, 7, 10, 2),
     _c(1, 23, 15, 3, 11, 22, 5, 9, 7, 17, 12, 14, 13, 4, 21, 8, 18, 2, 10, 20, 19, 6, 16)),
    (_c(1, 12, 15, 21, 2, 10, 8, 18, 9, 4, 14, 17, 19, 7, 13, 5, 20, 16, 11, 22, 6, 23, 3),
     _c(1, 8, 10, 23, 6, 18, 3, 2, 5, 13, 20, 21, 19, 17, 14, 16, 7, 15, 12, 11, 4, 9, 22)),
    (_c(1, 22, 9, 13, 20, 3, 10, 6, 18, 14, 17, 21, 19, 11, 4, 5, 8, 12, 16, 7, 23, 15, 2),
     _c(1, 23, 20, 13, 9, 18, 6, 10, 8, 21, 17, 14, 22, 11, 15, 2, 19, 5, 16, 12, 3, 7, 4)),
    (_c(1, 9, 4, 22, 2, 13, 10, 16, 7, 18, 6, 20, 5, 15, 3, 17, 21, 12, 19, 11, 8, 14, 23),
     _c(1, 2, 18, 20, 6, 22, 16, 10, 13, 23, 14, 3, 19, 12, 5, 7, 4, 9, 11, 15, 21, 17, 8)),
    (_c(1, 18, 3, 12, 13, 14, 10, 15, 9, 22, 5, 7, 11, 2, 16, 20, 8, 17, 23, 19, 6, 4, 21),
     _c(1, 5, 18, 11, 20, 4, 6, 15, 10, 14, 13, 12, 8, 7, 21, 16, 23, 17, 3, 22, 9, 19, 2)),
]


def hjlppz_file_indices() -> list[int]:
    """Class index of each triple of the HJLPPZ data file (paper form (x, z, y))."""
    out = []
    for x, z in HJLPPZ_FILE:
        y = HJLPPZ_FILE_Y
        assert mul_many(z, y, x) == identity(N)
        out.append(index_of((x, z, y)))
    return out


def janik_table_indices() -> list[tuple[int, str]]:
    """Class index of each row of [Janik26, Table 1], with the reading used.

    Janik's table is stated in a convention in which the published HJLPPZ
    cycles are inverted.  We try the direct reading g1 g2 g3 = 1 first; if the
    second entry is then in 23B, we invert all entries, which gives a triple
    (g3^-1, g2^-1, g1^-1)-style product-one relation g3' g2' g1' = 1, i.e. the
    triple (g1', g3', g2') up to rotation in our convention.
    """
    out = []
    for g2, g3 in JANIK_TABLE:
        g1 = m23.G1
        if mul_many(g1, g2, g3) == identity(N):
            try:
                out.append((index_of((g1, g2, g3)), "direct"))
                continue
            except ValueError:
                pass
            # second entry in 23B: the mirror passport; conjugate by t -> -t,
            # i.e. braid to (g1, g2 g3 g2^-1, g2)
            out.append((index_of((g1, mul_many(g2, g3, inv(g2)), g2)), "direct, braided"))
        else:
            a, b, c = inv(g1), inv(g2), inv(g3)
            assert mul_many(c, b, a) == identity(N)
            # c b a = 1  =>  a c' b' = 1 with ... use rotation: (a, c^a..., ) simply test
            t = (a, conj(c, a), conj(b, a)) if mul_many(a, conj(c, a), conj(b, a)) == identity(N) else None
            if t is None:
                t = (a, mul_many(inv(a), c, a), mul_many(inv(a), b, a))
            out.append((index_of(t), "inverted"))
    return out
