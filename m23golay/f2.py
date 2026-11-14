"""Linear algebra over F_2 with Python integers as bit vectors."""
from __future__ import annotations

from typing import Iterable


class Span:
    """A subspace of F_2^N kept in reduced echelon form.

    Each basis vector is stored together with a combination mask recording how
    it was produced from the vectors passed to :meth:`add`, which gives
    coordinates and solutions of linear systems for free.
    """

    def __init__(self, vectors: Iterable[int] = ()):
        self.pivots: dict[int, tuple[int, int]] = {}  # pivot bit -> (vector, combo)
        self._order: list[int] = []                   # pivot bits, descending
        self.n_added = 0
        for v in vectors:
            self.add(v)

    def _reduce(self, v: int, combo: int = 0) -> tuple[int, int]:
        """Full reduction: the result has no pivot bit set (linear, canonical)."""
        for hb in self._order:
            if v >> hb & 1:
                pv, pc = self.pivots[hb]
                v ^= pv
                combo ^= pc
        return v, combo

    def _insert(self, r: int, combo: int) -> None:
        hb = r.bit_length() - 1
        self.pivots[hb] = (r, combo)
        self._order.append(hb)
        self._order.sort(reverse=True)

    def add(self, v: int) -> bool:
        """Insert v; return True if it enlarged the span."""
        idx = self.n_added
        self.n_added += 1
        r, combo = self._reduce(v, 1 << idx)
        if r == 0:
            return False
        self._insert(r, combo)
        return True

    @property
    def dim(self) -> int:
        return len(self.pivots)

    def __contains__(self, v: int) -> bool:
        return self._reduce(v)[0] == 0

    def reduce(self, v: int) -> int:
        """Canonical representative of v modulo the span."""
        return self._reduce(v)[0]

    def solve(self, v: int) -> int | None:
        """Mask of added vectors summing to v, or None if v is not in the span."""
        r, combo = self._reduce(v)
        return combo if r == 0 else None

    def basis(self) -> list[int]:
        return [self.pivots[k][0] for k in sorted(self.pivots, reverse=True)]


def rank(vectors: Iterable[int]) -> int:
    return Span(vectors).dim


def kernel(images: list[int]) -> list[int]:
    """Basis of {x : sum_{i in x} images[i] = 0}, x as a mask over the indices."""
    s = Span()
    out = []
    for i, v in enumerate(images):
        r, combo = s._reduce(v, 1 << i)
        if r == 0:
            out.append(combo)
        else:
            s._insert(r, combo)
    return out


def popcount(v: int) -> int:
    return v.bit_count()
