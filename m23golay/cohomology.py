"""Mod 2 modules of M23 attached to the Golay code, and H^1 of three-point covers.

Right modules throughout.  P = F_2^23 is the permutation module, P0 its
even-weight part, G0 the even-weight Golay code (dimension 11, also the dual
code of the Golay code), and Q = P0/G0, which is dual to G0.

For a triple (g1, g2, g3) with g1 g2 g3 = 1 and a module M, a 1-cocycle of
the free group <gamma1, gamma2, gamma3 | gamma1 gamma2 gamma3 = 1> is given by
(c1, c2) in M^2, with  c3 = c1.g2 g3 + c2.g3  and  z(ab) = z(a).b + z(b).
Coboundaries are (a(1+g1), a(1+g2)).

  parabolic  :  c_i in M(1 + g_i) for i = 1,2,3    -> H^1(P^1, j_* M)
  orbifold   :  c_i (1 + g_i + ... + g_i^(e_i - 1)) = 0  -> H^1(Delta(e1,e2,e3), M)

For M = P0 the parabolic group is H^1(X, F_2) = Jac(X)[2] (dimension 2g).
"""
from __future__ import annotations

from . import golay, m23
from .f2 import Span, kernel
from .perm import act_mask, identity, inv, mul, order

N = 23


class Module:
    """A subquotient sub/quo of F_2^23 with the right permutation action."""

    def __init__(self, sub_basis, quo_basis=(), name="M"):
        self.name = name
        quo = Span(quo_basis)
        self._quo_dim = quo.dim
        ordered = quo.basis()
        full = Span(ordered)
        for v in sub_basis:
            if full.add(v):
                ordered.append(v)
        # rebuild so that 'solve' returns coordinates in the order `ordered`
        self._span = Span()
        self._ordered = []
        for v in ordered:
            if self._span.add(v):
                self._ordered.append(v)
        assert self._span.dim == len(self._ordered)
        # Span.solve returns a mask over *added* vectors; all were independent
        self.dim = len(self._ordered) - self._quo_dim
        self._cache = {}

    def coords(self, v: int) -> int:
        """Coordinates of the class of an ambient vector v (must lie in sub)."""
        combo = self._span.solve(v)
        if combo is None:
            raise ValueError("vector not in module")
        return combo >> self._quo_dim

    def lift(self, x: int) -> int:
        """An ambient representative of the element with coordinates x."""
        out, i = 0, 0
        while x:
            if x & 1:
                out ^= self._ordered[self._quo_dim + i]
            x >>= 1
            i += 1
        return out

    def matrix(self, g) -> tuple:
        """Rows: image of each basis element under g."""
        g = tuple(g)
        if g not in self._cache:
            self._cache[g] = tuple(
                self.coords(act_mask(self._ordered[self._quo_dim + i], g)) for i in range(self.dim))
        return self._cache[g]


def apply(x: int, mat) -> int:
    out, i = 0, 0
    while x:
        if x & 1:
            out ^= mat[i]
        x >>= 1
        i += 1
    return out


def mat_mul(a, b) -> tuple:
    """Matrix of 'first a, then b'."""
    return tuple(apply(row, b) for row in a)


def mat_add(a, b) -> tuple:
    return tuple(x ^ y for x, y in zip(a, b))


def mat_id(n) -> tuple:
    return tuple(1 << i for i in range(n))


def fixed_dim(mat) -> int:
    n = len(mat)
    return len(kernel([row ^ (1 << i) for i, row in enumerate(mat)]))


def standard_modules(system=None) -> dict[str, Module]:
    system = m23.steiner() if system is None else system
    g0 = golay.even_subcode(system).basis()
    p0 = [1 | (1 << i) for i in range(1, N)]
    return {
        "G0": Module(g0, name="G0"),
        "Q": Module(p0, g0, name="Q = P0/G0"),
        "P0": Module(p0, name="P0"),
    }


def fixed_space_table(system=None) -> dict[str, dict[str, int]]:
    """dim M^g for class representatives and M in {G0, Q, P0}; 'defect' = 2 dim G0^g - dim P0^g."""
    mods = standard_modules(system)
    out = {}
    for name, g in sorted(m23.class_representatives().items()):
        row = {k: fixed_dim(m.matrix(g)) for k, m in mods.items()}
        row["defect"] = row["G0"] + row["Q"] - row["P0"]
        out[name] = row
    return out


class TripleCohomology:
    """H^1 of the three-point cover with monodromy `triple`, coefficients in `module`."""

    def __init__(self, module: Module, triple):
        self.M = module
        self.triple = tuple(map(tuple, triple))
        g1, g2, g3 = self.triple
        assert mul(mul(g1, g2), g3) == identity(N)
        d = self.d = module.dim
        self.mat = [module.matrix(g) for g in self.triple]
        self.one_plus = [mat_add(m, mat_id(d)) for m in self.mat]
        self.norm = []
        for g, m in zip(self.triple, self.mat):
            acc, pw = mat_id(d), mat_id(d)
            for _ in range(order(g) - 1):
                pw = mat_mul(pw, m)
                acc = mat_add(acc, pw)
            self.norm.append(acc)
        self.g2g3 = mat_mul(self.mat[1], self.mat[2])
        # coboundaries
        self.B = Span(self.pack(apply(1 << i, self.one_plus[0]), apply(1 << i, self.one_plus[1]))
                      for i in range(d))
        self.images = [Span(rows) for rows in self.one_plus]
        self.Zpar = self._solutions(lambda c: tuple(self.images[i].reduce(c[i]) for i in range(3)))
        self.Zorb = self._solutions(lambda c: tuple(apply(c[i], self.norm[i]) for i in range(3)))

    # cocycles are stored as (c1 | c2 << d)
    def pack(self, c1, c2) -> int:
        return c1 | (c2 << self.d)

    def unpack(self, z) -> tuple[int, int, int]:
        c1, c2 = z & ((1 << self.d) - 1), z >> self.d
        c3 = apply(c1, self.g2g3) ^ apply(c2, self.mat[2])
        return c1, c2, c3

    def _solutions(self, conditions) -> Span:
        d = self.d
        imgs = []
        for i in range(2 * d):
            vals = conditions(self.unpack(1 << i))
            packed = 0
            for j, v in enumerate(vals):
                packed |= v << (j * d)
            imgs.append(packed)
        return Span(kernel(imgs))

    @property
    def dim_parabolic(self) -> int:
        assert all(b in self.Zpar for b in self.B.basis())
        return self.Zpar.dim - self.B.dim

    @property
    def dim_orbifold(self) -> int:
        assert all(b in self.Zorb for b in self.B.basis())
        return self.Zorb.dim - self.B.dim

    def complement_basis(self, Z: Span) -> list[int]:
        """Cocycles in Z whose classes form a basis of Z/B."""
        s = Span(self.B.basis())
        out = []
        for v in Z.basis():
            if s.add(v):
                out.append(v)
        return out

    def conjugation_matrix(self, c0, Z: Span | None = None):
        """Matrix of complex conjugation on Z/B (default: parabolic classes).

        z'(gamma) = z(kappa gamma).c0 with kappa(gamma1) = gamma1^-1,
        kappa(gamma2) = gamma3^-1, kappa(gamma3) = gamma2^-1.
        """
        Z = self.Zpar if Z is None else Z
        g1, g2, g3 = self.triple
        m_c0 = self.M.matrix(c0)
        m1 = mat_mul(self.M.matrix(inv(g1)), m_c0)
        m3 = mat_mul(self.M.matrix(inv(g3)), m_c0)
        basis = self.complement_basis(Z)
        s = Span(self.B.basis())
        for v in basis:
            s.add(v)
        nb = self.B.dim
        rows = []
        for v in basis:
            c1, c2, c3 = self.unpack(v)
            img = self.pack(apply(c1, m1), apply(c3, m3))
            assert img in Z
            combo = s.solve(img)
            rows.append(combo >> nb)
        return tuple(rows)


def lagrangian_report(triple, c0=None, system=None) -> dict:
    """Dimensions of the Golay pieces of Jac(X)[2] and, for real covers, of their real parts."""
    mods = standard_modules(system)
    out = {}
    for key, m in mods.items():
        h = TripleCohomology(m, triple)
        entry = {"parabolic": h.dim_parabolic, "orbifold": h.dim_orbifold}
        if c0 is not None:
            mat = h.conjugation_matrix(c0)
            assert mat_mul(mat, mat) == mat_id(len(mat)), "conjugation is not an involution"
            entry["real_part"] = fixed_dim(mat)
        out[key] = entry
    return out
