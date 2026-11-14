"""M23 as the permutation group <g1, g2> of [HJLPPZ26, Section 3.1].

Everything here is derived from the two published generators: the stabilizer
chain (hence the order), the Steiner system S(4,7,23) they preserve, a full
enumeration of the 10,200,960 elements with a census of cycle types, and class
representatives.
"""
from __future__ import annotations

from functools import lru_cache

from . import golay
from .perm import (act_set, cycle_type, cycles_of, fixed_points, from_cycles, identity,
                   inv, mul, mul_many, order, power)

N = 23
ORDER = 10_200_960

# [HJLPPZ26, Section 3.1]; right action, g1 g2 g3 = 1.
G1 = from_cycles([(1, 11), (2, 23), (3, 8), (4, 16), (5, 21), (7, 20), (15, 19), (18, 22)])
G2 = from_cycles([(1, 2, 3, 4, 10, 11, 12, 7, 19, 18, 8, 6, 9, 16, 17, 21, 22, 5, 14, 20, 13, 15, 23)])
G3 = from_cycles([(1, 2, 11, 10, 16, 9, 6, 3, 23, 19, 20, 14, 21, 17, 4, 8, 22, 5, 18, 15, 13, 7, 12)])

# Cycle types on 23 points of the 17 classes (12 distinct types).
CLASS_TYPES = {
    "1A": "1^23", "2A": "1^7 2^8", "3A": "1^5 3^6", "4A": "1^3 2^2 4^4", "5A": "1^3 5^4",
    "6A": "1 2^2 3^2 6^2", "7AB": "1^2 7^3", "8A": "1 2 4 8^2", "11AB": "1 11^2",
    "14AB": "2 7 14", "15AB": "3 5 15", "23AB": "23",
}


class StabChain:
    """Deterministic Schreier-Sims stabilizer chain for a small permutation group."""

    def __init__(self, gens, n: int = N):
        self.n = n
        self.base: list[int] = []
        self.sgens: list[list[tuple]] = []      # sgens[i] generate the i-th stabilizer
        self.trans: list[dict[int, tuple]] = []  # trans[i][pt] = u with base[i]^u = pt
        for g in gens:
            self._add(tuple(g), 0)
        self._close()

    def _moved_point(self, g):
        for i, j in enumerate(g):
            if i != j:
                return i
        return None

    def _extend_base(self, g):
        self.base.append(self._moved_point(g))
        self.sgens.append([])
        self.trans.append({})

    def _add(self, g, level):
        """Add g as a strong generator at all levels <= the first base point it moves."""
        if g == identity(self.n):
            return
        j = 0
        while j < len(self.base) and g[self.base[j]] == self.base[j]:
            j += 1
        if j == len(self.base):
            self._extend_base(g)
        for i in range(level, j + 1):
            self.sgens[i].append(g)
            self._orbit(i)

    def _orbit(self, i):
        b = self.base[i]
        trans = {b: identity(self.n)}
        queue = [b]
        while queue:
            pt = queue.pop()
            for s in self.sgens[i]:
                img = s[pt]
                if img not in trans:
                    trans[img] = mul(trans[pt], s)
                    queue.append(img)
        self.trans[i] = trans

    def sift(self, g, level: int = 0):
        g = tuple(g)
        for i in range(level, len(self.base)):
            pt = g[self.base[i]]
            if pt not in self.trans[i]:
                return g, i
            g = mul(g, inv(self.trans[i][pt]))
        return g, len(self.base)

    def _close(self):
        i = len(self.base) - 1
        while i >= 0:
            restart = None
            for pt, u in list(self.trans[i].items()):
                for s in self.sgens[i]:
                    sch = mul_many(u, s, inv(self.trans[i][s[pt]]))
                    res, lvl = self.sift(sch, i + 1)
                    if res != identity(self.n):
                        self._add(res, i + 1)
                        restart = max(lvl, i + 1) if lvl < len(self.base) else len(self.base) - 1
                        break
                if restart is not None:
                    break
            i = restart if restart is not None else i - 1

    @property
    def orbit_lengths(self) -> list[int]:
        return [len(t) for t in self.trans]

    @property
    def order(self) -> int:
        out = 1
        for t in self.trans:
            out *= len(t)
        return out

    def __contains__(self, g) -> bool:
        res, _ = self.sift(g)
        return res == identity(self.n)


@lru_cache(maxsize=None)
def chain() -> StabChain:
    c = StabChain([G1, G2])
    assert c.order == ORDER, c.order
    return c


@lru_cache(maxsize=None)
def steiner() -> frozenset:
    """The S(4,7,23) preserved by <g1,g2>: the orbit of Fix(g1)."""
    start = fixed_points(G1)
    seen, queue = {start}, [start]
    while queue:
        h = queue.pop()
        for g in (G1, G2):
            k = act_set(h, g)
            if k not in seen:
                seen.add(k)
                queue.append(k)
    system = frozenset(seen)
    assert golay.is_steiner_4_7_23(system)
    return system


def cyclic_labelling(sigma) -> list[int]:
    """lab[x] = i such that x = x0^(sigma^i), where x0 = point 0; sigma a 23-cycle."""
    lab, x = [0] * N, 0
    for i in range(N):
        lab[x] = i
        x = sigma[x]
    assert x == 0 and sorted(lab) == list(range(N))
    return lab


def sign_of_23_element(sigma, system=None) -> int:
    """+1 or -1 according to which cyclic Golay code the cycle of sigma induces.

    Label the points by Z/23 so that sigma is i -> i+1.  The Steiner system
    becomes shift invariant, hence equals the heptad system of G^+ or of G^-
    (Lemma "two cyclic codes").  This sign separates the classes 23A and 23B.
    """
    system = steiner() if system is None else system
    return golay.cyclic_sign(golay.transport(system, cyclic_labelling(sigma)))


def enumerate_elements():
    """All 10,200,960 elements as a uint8 numpy array of shape (ORDER, 23)."""
    import numpy as np
    c = chain()
    elems = np.arange(N, dtype=np.uint8)[None, :]
    for level in reversed(range(len(c.base))):
        t = np.array(list(c.trans[level].values()), dtype=np.uint8)
        # (e * u)[x] = u[e[x]]
        elems = t[:, elems].reshape(-1, N)
    return elems


def census() -> dict[tuple[int, int, int], int]:
    """Number of elements for each value of (Fix g, Fix g^2, Fix g^3).

    These three numbers separate the 12 cycle types that occur in M23.
    """
    import numpy as np
    elems = enumerate_elements()
    assert elems.shape == (ORDER, N)
    idx = np.arange(N, dtype=np.uint8)[None, :]
    sq = np.take_along_axis(elems, elems, axis=1)
    cu = np.take_along_axis(sq, elems, axis=1)
    f1 = (elems == idx).sum(axis=1)
    f2 = (sq == idx).sum(axis=1)
    f3 = (cu == idx).sum(axis=1)
    key = f1 * 10000 + f2 * 100 + f3
    vals, counts = np.unique(key, return_counts=True)
    return {(int(v) // 10000, int(v) // 100 % 100, int(v) % 100): int(c)
            for v, c in zip(vals, counts)}


SIGNATURES = {  # (Fix g, Fix g^2, Fix g^3) -> cycle-type label
    (23, 23, 23): "1A", (7, 23, 7): "2A", (5, 5, 23): "3A", (3, 7, 3): "4A", (3, 3, 3): "5A",
    (1, 5, 7): "6A", (2, 2, 2): "7AB", (1, 3, 1): "8A", (1, 1, 1): "11AB", (0, 2, 0): "14AB",
    (0, 0, 3): "15AB", (0, 0, 0): "23AB",
}

# |C_G(g)|; split types consist of two classes of equal size (g and g^-1).
CENTRALIZER_ORDERS = {"1A": ORDER, "2A": 2688, "3A": 180, "4A": 32, "5A": 15, "6A": 12,
                      "7AB": 14, "8A": 8, "11AB": 11, "14AB": 14, "15AB": 15, "23AB": 23}


def signature(p) -> tuple[int, int, int]:
    p2 = mul(p, p)
    p3 = mul(p2, p)
    return (len(fixed_points(p)), len(fixed_points(p2)), len(fixed_points(p3)))


def label(p) -> str:
    return SIGNATURES[signature(p)]


@lru_cache(maxsize=None)
def class_representatives() -> dict[str, tuple]:
    """One element per cycle type: powers of a deterministic walk in g1, g2."""
    import random
    rng = random.Random(23)
    reps: dict[str, tuple] = {"1A": identity(N)}
    p = identity(N)
    while len(reps) < len(SIGNATURES):
        p = mul(p, rng.choice((G1, G2, G3)))
        q = p
        for _ in range(order(p) - 1):
            reps.setdefault(label(q), q)
            q = mul(q, p)
    return reps


def cycle_index_data() -> list[tuple[str, int, tuple[int, ...]]]:
    """(label, number of elements, cycle type) for the 12 cycle types."""
    reps = class_representatives()
    out = []
    for name, cent in CENTRALIZER_ORDERS.items():
        mult = 2 if name.endswith("AB") else 1
        out.append((name, mult * ORDER // cent, cycle_type(reps[name])))
    assert sum(n for _, n, _ in out) == ORDER
    return out


def involutions() -> list[tuple]:
    """The 3795 involutions (a single class, 2A)."""
    seen, queue = {G1}, [G1]
    while queue:
        a = queue.pop()
        for g in (G1, G2):
            b = mul_many(inv(g), a, g)
            if b not in seen:
                seen.add(b)
                queue.append(b)
    assert len(seen) == 3795
    return sorted(seen)


def conjugator_in_group(a, b):
    """An element h of M23 with h^-1 a h = b, for 23-cycles a, b; None if not conjugate.

    A conjugator in S23 is determined by the image of one point, so there are 23
    candidates; membership in M23 is decided with the stabilizer chain.
    """
    c = chain()
    for start in range(N):
        h = [0] * N
        x, y = 0, start
        for _ in range(N):
            h[x] = y
            x, y = a[x], b[y]
        h = tuple(h)
        if h in c:
            return h
    return None


__all__ = [name for name in dir() if not name.startswith("_")]
_ = (cycles_of, order, power)
