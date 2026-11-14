"""Counting invariants: Molien series, orbits on subsets, permutation characters."""
from __future__ import annotations

import itertools
from fractions import Fraction

from . import m23
from .perm import act_set, cycle_type


def _series_product(types_with_counts, order, degree, squarefree=False):
    """(1/|G|) sum_g prod over cycles of 1/(1 - t^len)  [or (1 + t^len) if squarefree]."""
    total = [Fraction(0)] * (degree + 1)
    for _, count, ct in types_with_counts:
        ser = [Fraction(0)] * (degree + 1)
        ser[0] = Fraction(1)
        for length in ct:
            if squarefree:
                for d in range(degree, length - 1, -1):
                    ser[d] += ser[d - length]
            else:
                for d in range(length, degree + 1):
                    ser[d] += ser[d - length]
        for d in range(degree + 1):
            total[d] += count * ser[d]
    out = [x / order for x in total]
    assert all(x.denominator == 1 for x in out)
    return [int(x) for x in out]


def molien(degree: int = 12) -> list[int]:
    """dim of the degree-d part of Q[x_1..x_23]^{M23}, d = 0..degree."""
    return _series_product(m23.cycle_index_data(), m23.ORDER, degree)


def partitions_at_most(degree: int, parts: int = 23) -> list[int]:
    """Number of partitions of d into at most `parts` parts = dim of symmetric invariants."""
    p = [1] + [0] * degree
    for k in range(1, parts + 1):
        for d in range(k, degree + 1):
            p[d] += p[d - k]
    return p


def subset_orbits() -> list[int]:
    """Number of M23-orbits on k-subsets of 23 points, k = 0..23."""
    return _series_product(m23.cycle_index_data(), m23.ORDER, 23, squarefree=True)


def squarefree_symmetric() -> list[int]:
    return [1] * 24


def permutation_characters() -> dict[str, dict[str, int]]:
    """Fixed-point counts of class representatives on several M23-sets."""
    system = m23.steiner()
    sets = {
        "points": [frozenset([i]) for i in range(23)],
        "duads": [frozenset(c) for c in itertools.combinations(range(23), 2)],
        "heptads": list(system),
        "triads": [frozenset(c) for c in itertools.combinations(range(23), 3)],
    }
    flags = [(i, h) for h in system for i in h]
    out = {}
    for name, g in sorted(m23.class_representatives().items()):
        row = {k: sum(1 for s in v if act_set(s, g) == s) for k, v in sets.items()}
        row["flags"] = sum(1 for i, h in flags if g[i] == i and act_set(h, g) == h)
        out[name] = row
    return out


def cycle_type_on(blocks, g) -> tuple[int, ...]:
    """Cycle type of g acting on a family of subsets."""
    seen, out = set(), []
    for b in blocks:
        if b in seen:
            continue
        n, c = 0, b
        while c not in seen:
            seen.add(c)
            c = act_set(c, g)
            n += 1
        out.append(n)
    return tuple(sorted(out))


def gassmann_duads_heptads() -> dict[str, bool]:
    """For each cycle type: do duads and heptads have the same cycle type?"""
    system = list(m23.steiner())
    duads = [frozenset(c) for c in itertools.combinations(range(23), 2)]
    return {name: cycle_type_on(system, g) == cycle_type_on(duads, g)
            for name, g in sorted(m23.class_representatives().items())}


def subdegrees() -> dict[str, list[int]]:
    """Orbit lengths of a block stabiliser on blocks: (1,42,210) for duads, (1,112,140) for heptads."""
    system = list(m23.steiner())
    h0 = system[0]
    hept = sorted([sum(1 for h in system if len(h & h0) == k) for k in (7, 1, 3)])
    duad = sorted([1, 2 * 21, 21 * 20 // 2])
    return {"heptads": hept, "duads": duad}


_ = cycle_type
