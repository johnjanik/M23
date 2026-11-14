#!/usr/bin/env python3
"""The fifteen lifts of the HJLPPZ triple to the group 2^11:M23, as permutations of 46 points.

For the rational class of Ni(2A,23A,23B), each nonzero element w of the Golay
Lagrangian W = H^1(P^1, j_* G_0) is a parabolic cocycle (c1, c2, c3) with values in
the even Golay code G_0.  It defines the triple
    g~_i :  (x, e) -> (x^{g_i}, e + c_i[x^{g_i}])        on  Omega x F_2  (46 points),
with g~_1 g~_2 g~_3 = 1, cycle types (1^14 2^16, 23^2, 23^2), generating G_0:M23 of
order 2^11 * 10200960.  The corresponding cover Z_w -> P^1 has degree 46 and genus 7
and is the etale double cover of X attached to w.  Galois permutes the fifteen covers
through GL_4(F_2) = A_8.

Point (x, e) is numbered 2x + e (0-based).  Writes data/lifts_2_11_M23.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import cohomology as coh  # noqa: E402
from m23golay import m23, nielsen  # noqa: E402
from m23golay.perm import cycle_type, identity, mul_many, type_string  # noqa: E402

N = 23
triple = nielsen.nielsen_class()[nielsen.hjlppz_file_indices()[0]]      # the rational class
g1, g2, g3 = triple
G0 = coh.standard_modules()["G0"]
H = coh.TripleCohomology(G0, triple)
basis = H.complement_basis(H.Zpar)
assert len(basis) == 4


def lift(g, c):
    """Permutation of 46 points for (g, c), c an ambient vector in F_2^23 (bitmask)."""
    out = [0] * 46
    for x in range(N):
        y = g[x]
        for e in (0, 1):
            out[2 * x + e] = 2 * y + (e ^ (c >> y & 1))
    return tuple(out)


def genus(perms, degree):
    ram = sum(degree - len(set(map(tuple, [sorted(c) for c in __cycles(p)]))) for p in perms)
    return (ram - 2 * degree + 2) // 2


def __cycles(p):
    seen, out = set(), []
    for i in range(len(p)):
        if i not in seen:
            cyc, j = [], i
            while j not in seen:
                seen.add(j)
                cyc.append(j)
                j = p[j]
            out.append(cyc)
    return out


records = []
for m in range(1, 16):
    z = 0
    for i in range(4):
        if m >> i & 1:
            z ^= basis[i]
    c1, c2, c3 = (G0.lift(c) for c in H.unpack(z))
    t = (lift(g1, c1), lift(g2, c2), lift(g3, c3))
    assert mul_many(*t) == identity(46)
    types = [type_string(cycle_type(p)) for p in t]
    assert types == ["1^14 2^16", "23^2", "23^2"], types
    assert genus(t, 46) == 7
    # blocks {2x, 2x+1} are preserved and the action on blocks is the original triple
    for p, g in zip(t, triple):
        assert all(p[2 * x] // 2 == g[x] == p[2 * x + 1] // 2 for x in range(N))
    records.append({"w": m, "g1": list(t[0]), "g2": list(t[1]), "g3": list(t[2])})

# group order for the first lift (Schreier-Sims on 46 points)
chain = m23.StabChain([records[0]["g1"], records[0]["g2"]], n=46)
order = chain.order
print("order of <g~1, g~2> for w = 1:", order, "= 2^11 * |M23|:", order == 2**11 * m23.ORDER)
assert order == 2**11 * m23.ORDER
# the fifteen triples are pairwise inequivalent: distinct cohomology classes by construction;
# as a sanity check the triples themselves are distinct
assert len({(tuple(r["g1"]), tuple(r["g2"])) for r in records}) == 15
out = {"description": "Fifteen generating triples of 2^11:M23 <= S_46 lifting the rational triple of Ni(2A,23A,23B); "
                      "points numbered 2x+e, 0-based; right action; g1*g2*g3 = 1",
       "passport": ["1^14 2^16", "23^2", "23^2"], "degree": 46, "genus": 7,
       "group_order": order, "base_triple": [list(g) for g in triple], "lifts": records}
(Path(__file__).resolve().parent.parent / "data" / "lifts_2_11_M23.json").write_text(json.dumps(out) + "\n")
print("wrote data/lifts_2_11_M23.json with", len(records), "triples")
