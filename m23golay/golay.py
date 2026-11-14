"""The two cyclic binary Golay codes of length 23 and their Steiner systems.

Over F_2,  z^23 - 1 = (z - 1) g_plus(z) g_minus(z)  with

    g_plus (z) = z^11 + z^9 + z^7 + z^6 + z^5 + z + 1      (0xAE3)
    g_minus(z) = z^11 + z^10 + z^6 + z^5 + z^4 + z^2 + 1   (0xC75)

the two being reciprocal.  The cyclic codes they generate are the two binary
quadratic-residue codes; each is a perfect [23,12,7] Golay code, and i -> -i
exchanges them.  Coordinates are indexed by Z/23, and the cyclic shift
i -> i+1 is an automorphism of both.

The weight-7 supports ("heptads") of either code form a Steiner system
S(4,7,23).  A system is represented as a frozenset of frozensets.
"""
from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache
from math import comb

from .f2 import Span

N = 23
G_PLUS = 0xAE3
G_MINUS = 0xC75
WEIGHT_DISTRIBUTION = {0: 1, 7: 253, 8: 506, 11: 1288, 12: 1288, 15: 506, 16: 253, 23: 1}


def _polmul_mod(a: int, b: int) -> int:
    """Product in F_2[z]/(z^23 - 1)."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a >> N:
            a = (a & ((1 << N) - 1)) | 1
    return out


def reciprocal(g: int, deg: int = 11) -> int:
    return sum(((g >> i) & 1) << (deg - i) for i in range(deg + 1))


def codewords(gen: int = G_PLUS) -> list[int]:
    """All 4096 codewords as 23-bit masks (bit i = coordinate i)."""
    out = []
    for message in range(1 << 12):
        w = 0
        for i in range(12):
            if message >> i & 1:
                w ^= gen << i
        out.append(w)
    return out


@lru_cache(maxsize=None)
def heptads(sign: int = +1) -> frozenset[frozenset[int]]:
    """The 253 heptads of the cyclic Golay code G^+ (sign=+1) or G^- (sign=-1)."""
    gen = G_PLUS if sign > 0 else G_MINUS
    return frozenset(
        frozenset(i for i in range(N) if w >> i & 1)
        for w in codewords(gen) if w.bit_count() == 7
    )


def mask(s) -> int:
    out = 0
    for i in s:
        out |= 1 << i
    return out


def code_span(system) -> Span:
    """F_2-span of the heptad indicator vectors (the Golay code, dimension 12)."""
    return Span(mask(h) for h in system)


def even_subcode(system) -> Span:
    """The even-weight subcode G_0 (dimension 11) = dual of the Golay code."""
    hs = sorted(mask(h) for h in system)
    return Span(hs[0] ^ h for h in hs[1:])


def is_steiner_4_7_23(system) -> bool:
    if len(system) != 253 or any(len(h) != 7 for h in system):
        return False
    counts = Counter(q for h in system for q in itertools.combinations(sorted(h), 4))
    return len(counts) == comb(23, 4) and set(counts.values()) == {1}


def five_subsets(system) -> frozenset[frozenset[int]]:
    """The 5313 five-subsets lying in a heptad (the support of J_5)."""
    return frozenset(frozenset(s) for h in system for s in itertools.combinations(sorted(h), 5))


def block_through(system):
    """Dictionary 4-subset -> the unique heptad containing it."""
    return {frozenset(q): h for h in system for q in itertools.combinations(sorted(h), 4)}


def transport(system, perm) -> frozenset[frozenset[int]]:
    """Image of a system under a point map i -> perm[i]."""
    return frozenset(frozenset(perm[i] for i in h) for h in system)


def cyclic_sign(system) -> int:
    """+1 / -1 if `system` (on Z/23, shift-invariant) is the G^+ / G^- system."""
    if system == heptads(+1):
        return +1
    if system == heptads(-1):
        return -1
    raise ValueError("not one of the two cyclic Golay systems")


def verify_basic_facts() -> dict:
    """Machine check of Section 2 of the research note, for both cyclic codes."""
    report = {}
    assert _polmul_mod(_polmul_mod(G_PLUS, G_MINUS), 0b11) == 0, "(z+1) g+ g- != z^23 - 1"
    assert reciprocal(G_PLUS) == G_MINUS
    for sign, gen in ((+1, G_PLUS), (-1, G_MINUS)):
        words = codewords(gen)
        assert len(set(words)) == 4096
        wd = dict(sorted(Counter(w.bit_count() for w in words).items()))
        assert wd == WEIGHT_DISTRIBUTION, wd
        hs = heptads(sign)
        assert is_steiner_4_7_23(hs)
        shift = [(i + 1) % N for i in range(N)]
        assert transport(hs, shift) == hs, "not cyclic"
        assert all(sum(i in h for h in hs) == 77 for i in range(N))
        assert sum(0 in h and 1 in h for h in hs) == 21
        assert sum(0 in h and 1 in h and 2 in h for h in hs) == 5
        inter = Counter(len(a & b) for a, b in itertools.combinations(hs, 2))
        assert set(inter) == {1, 3}, inter
        assert len(five_subsets(hs)) == 253 * comb(7, 5) == 5313
        assert code_span(hs).dim == 12 and even_subcode(hs).dim == 11
        # perfect code: radius-3 balls partition F_2^23
        assert 2 ** 12 * sum(comb(23, j) for j in range(4)) == 2 ** 23
        report[sign] = {"weights": wd, "intersections": dict(inter)}
    neg = [(-i) % N for i in range(N)]
    assert transport(heptads(+1), neg) == heptads(-1)
    assert heptads(+1) != heptads(-1)
    report["pairs_of_heptads"] = {"meet_in_1": 253 * 112 // 2, "meet_in_3": 253 * 140 // 2}
    assert report[+1]["intersections"] == {1: 253 * 112 // 2, 3: 253 * 140 // 2}
    return report
