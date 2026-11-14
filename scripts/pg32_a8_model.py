#!/usr/bin/env python3
"""The exceptional isomorphism GL_4(2) = A_8 in the cyclic (Singer) model of PG(3,2).

Points of PG(3,2) are the exponents 0..14 of a root x of a primitive quartic Q over F_2
(the nonzero elements of F_16 = F_2[x]/(Q)); three points i, j, k are collinear when
x^i + x^j + x^k = 0.  There are 35 lines and 56 spreads (5 pairwise disjoint lines).
Call two spreads adjacent when they share exactly one line; this graph is the Johnson
graph J(8,3), so the spreads are the 3-subsets of a set of 8 "letters":
  * the maximal cliques of size 6 are the 28 pairs of letters;
  * a letter is a set of 7 such cliques that pairwise share a spread; the union of the
    7 cliques is the family of 21 spreads "containing" the letter.
GL_4(2) permutes the 8 letters as A_8.

This is the combinatorial input for the octic resolvent of the A_8 field (see
docs/RUSTMATH_HANDOFF.md, step A5).  Writes data/pg32_a8_model.json for both primitive
quartics x^4+x+1 and x^4+x^3+1.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import m23  # noqa: E402


def model(q_mask):
    # powers of x in F_2[x]/(Q), as bitmasks
    pw, v = [], 1
    for _ in range(15):
        pw.append(v)
        v <<= 1
        if v & 16:
            v ^= q_mask
    assert v == 1 and len(set(pw)) == 15, "Q is not primitive"
    idx = {val: i for i, val in enumerate(pw)}
    lines = sorted({tuple(sorted((i, j, idx[pw[i] ^ pw[j]]))) for i in range(15) for j in range(i + 1, 15)})
    assert len(lines) == 35
    # spreads: 5 pairwise disjoint lines
    spreads = []

    def extend(chosen, used, start):
        if len(chosen) == 5:
            spreads.append(tuple(chosen))
            return
        first = min(set(range(15)) - used)
        for li in range(len(lines)):
            L = lines[li]
            if first in L and not (set(L) & used):
                extend(chosen + [li], used | set(L), li + 1)

    extend([], set(), 0)
    assert len(spreads) == 56
    sset = [set(s) for s in spreads]
    adj = [[len(sset[a] & sset[b]) == 1 for b in range(56)] for a in range(56)]
    assert all(sum(row) == 15 for row in adj)                    # J(8,3) has valency 3*5 = 15
    # cliques of size 6: for adjacent a, b the common neighbours together with a, b
    cliques = set()
    for a in range(56):
        for b in range(a + 1, 56):
            if adj[a][b]:
                common = [c for c in range(56) if adj[a][c] and adj[b][c]]
                for extra in itertools.combinations(common, 4):
                    cl = (a, b) + extra
                    if all(adj[u][w] for u, w in itertools.combinations(cl, 2)):
                        cliques.add(tuple(sorted(cl)))
    cliques = sorted(cliques)
    assert len(cliques) == 28, len(cliques)
    meet = [[len(set(c) & set(d)) == 1 for d in cliques] for c in cliques]
    # letters: sets of 7 cliques pairwise meeting in one spread
    letters = set()
    for c in range(28):
        nb = [d for d in range(28) if meet[c][d]]
        assert len(nb) == 12                                     # a pair meets 2*6 other pairs
        for d in nb:
            star = [c, d] + [e for e in nb if meet[d][e] and
                             all(meet[e][f] for f in (c, d))]
            # the common neighbours of c and d are the 5 other pairs through the shared letter
            # plus the third side of the triangle; remove the one not meeting the others
            cand = [e for e in star[2:] if sum(meet[e][f] for f in star[2:] if f != e) >= 4]
            full = tuple(sorted([c, d] + cand))
            if len(full) == 7 and all(meet[u][w] for u, w in itertools.combinations(full, 2)):
                letters.add(full)
    letters = sorted(letters)
    assert len(letters) == 8, len(letters)
    families = [sorted(set().union(*[set(cliques[c]) for c in L])) for L in letters]
    assert all(len(f) == 21 for f in families)
    assert all(sum(s in f for f in families) == 3 for s in range(56))     # a spread = 3 letters

    # action of GL_4(2) on the letters: Singer cycle i -> i+1, Frobenius i -> 2i, one transvection
    def on_letters(point_perm):
        line_perm = {li: lines.index(tuple(sorted(point_perm[p] for p in lines[li]))) for li in range(35)}
        sp_index = {frozenset(s): k for k, s in enumerate(spreads)}
        sp_perm = [sp_index[frozenset(line_perm[li] for li in s)] for s in spreads]
        fam_index = {frozenset(f): k for k, f in enumerate(families)}
        return tuple(fam_index[frozenset(sp_perm[s] for s in f)] for f in families)

    singer = [(i + 1) % 15 for i in range(15)]
    frob = [(2 * i) % 15 for i in range(15)]
    # transvection v -> v + lambda(v) u with u = 1, lambda = coefficient of x^3
    trans = [idx[pw[i] ^ (1 if pw[i] & 8 else 0)] for i in range(15)]
    gens = [on_letters(g) for g in (singer, frob, trans)]
    order = m23.StabChain(gens, n=8).order
    assert order == 20160, order

    def ctype(p):
        seen, out = set(), []
        for i in range(8):
            if i not in seen:
                n, j = 0, i
                while j not in seen:
                    seen.add(j)
                    j = p[j]
                    n += 1
                out.append(n)
        return sorted(out)

    assert ctype(gens[0]) == [3, 5] and ctype(gens[2]) == [2, 2, 2, 2]
    return {"points": "exponents 0..14 of x in F_2[x]/(Q)", "lines": lines,
            "spreads": [list(s) for s in spreads], "letter_families": families,
            "singer_cycle_on_letters": list(gens[0]), "frobenius_on_letters": list(gens[1]),
            "transvection_on_letters": list(gens[2]), "group_order_on_letters": order}


out = {"x^4+x+1": model(0b10011), "x^4+x^3+1": model(0b11001)}
(Path(__file__).resolve().parent.parent / "data" / "pg32_a8_model.json").write_text(json.dumps(out) + "\n")
print("PG(3,2): 35 lines, 56 spreads, 28 cliques, 8 letters; GL_4(2) acts on the letters with order 20160;")
print("Singer cycle has type 3 5 and a transvection has type 2^4 on the 8 letters.  Wrote data/pg32_a8_model.json")
