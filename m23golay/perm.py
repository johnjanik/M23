"""Permutations of {0,...,n-1} in the RIGHT-action convention.

A permutation is a tuple p with p[x] = x^p (the image of x).  Products are read
left to right, as in Magma/GAP and in [HJLPPZ26, footnote 3]:

    mul(p, q)[x] = q[p[x]]        i.e.  x^(pq) = (x^p)^q.

All cycle notation coming from the papers is 1-indexed; internally points are
0-indexed (paper point k is internal point k-1).
"""
from __future__ import annotations

from typing import Iterable, Sequence

Perm = tuple


def identity(n: int = 23) -> Perm:
    return tuple(range(n))


def mul(p: Sequence[int], q: Sequence[int]) -> Perm:
    """First p, then q."""
    return tuple(q[i] for i in p)


def mul_many(*ps: Sequence[int]) -> Perm:
    out = ps[0]
    for p in ps[1:]:
        out = mul(out, p)
    return tuple(out)


def inv(p: Sequence[int]) -> Perm:
    out = [0] * len(p)
    for i, j in enumerate(p):
        out[j] = i
    return tuple(out)


def power(p: Sequence[int], k: int) -> Perm:
    n = len(p)
    if k < 0:
        p, k = inv(p), -k
    result, base = identity(n), tuple(p)
    while k:
        if k & 1:
            result = mul(result, base)
        base = mul(base, base)
        k >>= 1
    return result


def conj(p: Sequence[int], h: Sequence[int]) -> Perm:
    """h^-1 p h (the right-action conjugate p^h)."""
    return mul_many(inv(h), p, h)


def from_cycles(cycles: Iterable[Iterable[int]], n: int = 23) -> Perm:
    """Build a permutation from 1-indexed disjoint cycles."""
    p = list(range(n))
    for cyc in cycles:
        cyc = [c - 1 for c in cyc]
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            p[a] = b
    assert sorted(p) == list(range(n)), "cycles are not disjoint"
    return tuple(p)


def cycles_of(p: Sequence[int]) -> list[list[int]]:
    """Cycles (0-indexed), fixed points included, each starting at its minimum."""
    seen, out = [False] * len(p), []
    for i in range(len(p)):
        if not seen[i]:
            cyc, j = [], i
            while not seen[j]:
                seen[j] = True
                cyc.append(j)
                j = p[j]
            out.append(cyc)
    return out


def cycle_type(p: Sequence[int]) -> tuple[int, ...]:
    """Sorted cycle lengths, e.g. (1,1,1,1,1,1,1,2,2,2,2,2,2,2,2) for class 2A."""
    return tuple(sorted(len(c) for c in cycles_of(p)))


def type_string(ct: Sequence[int]) -> str:
    out = []
    for length in sorted(set(ct)):
        m = list(ct).count(length)
        out.append(f"{length}^{m}" if m > 1 else f"{length}")
    return " ".join(out)


def order(p: Sequence[int]) -> int:
    from math import lcm
    return lcm(*cycle_type(p))


def fixed_points(p: Sequence[int]) -> frozenset[int]:
    return frozenset(i for i, j in enumerate(p) if i == j)


def to_cycle_string(p: Sequence[int]) -> str:
    """1-indexed cycle notation without fixed points."""
    parts = ["(" + " ".join(str(x + 1) for x in c) + ")" for c in cycles_of(p) if len(c) > 1]
    return "".join(parts) or "()"


def act_set(s: Iterable[int], p: Sequence[int]) -> frozenset[int]:
    return frozenset(p[i] for i in s)


def act_mask(v: int, p: Sequence[int]) -> int:
    """Right action on a bitmask v in F_2^n:  (v.p)[x^p] = v[x]."""
    out, i = 0, 0
    while v:
        if v & 1:
            out |= 1 << p[i]
        v >>= 1
        i += 1
    return out
