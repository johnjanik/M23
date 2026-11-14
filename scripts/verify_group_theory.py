#!/usr/bin/env python3
"""Machine verification of every group-theoretic statement used in the paper.

Pure Python (numpy only for the enumeration of the 10,200,960 elements).
Writes data/group_facts.json, from which the tables of the paper are generated.
"""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import cohomology as coh  # noqa: E402
from m23golay import golay, invariants, m23, nielsen  # noqa: E402
from m23golay.f2 import Span  # noqa: E402
from m23golay.perm import (act_mask, act_set, cycle_type, fixed_points, identity, inv, mul,  # noqa: E402
                           mul_many, power, type_string)

OUT = Path(__file__).resolve().parent.parent / "data" / "group_facts.json"
facts = {}


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(f"verification failed: {name}")


# --- Section 2: Golay codes ------------------------------------------------
golay.verify_basic_facts()
check("both cyclic codes are [23,12,7] with the stated weight distribution; heptads form S(4,7,23)", True)

# --- M23 from the published generators ---------------------------------------
c = m23.chain()
check("|<g1,g2>| = 10200960 (Schreier-Sims)", c.order == m23.ORDER)
check("g1 g2 g3 = 1", mul_many(m23.G1, m23.G2, m23.G3) == identity(23))
S = m23.steiner()
check("orbit of Fix(g1) is a Steiner system S(4,7,23)", golay.is_steiner_4_7_23(S))
cen = m23.census()
check("census: 12 cycle types with the class sizes of M23",
      all(cnt == (2 if m23.SIGNATURES[s].endswith("AB") else 1) * m23.ORDER // m23.CENTRALIZER_ORDERS[m23.SIGNATURES[s]]
          for s, cnt in cen.items()) and len(cen) == 12)
invs = m23.involutions()
fixsets = Counter(fixed_points(a) for a in invs)
check("3795 involutions; each fixes a heptad pointwise; each heptad arises from 15 involutions",
      len(invs) == 3795 and set(fixsets) == set(S) and set(fixsets.values()) == {15})

# --- two cyclic codes --------------------------------------------------------
sg2, sg3 = m23.sign_of_23_element(m23.G2), m23.sign_of_23_element(m23.G3)
qr = {pow(k, 2, 23) for k in range(1, 23)}
check("sign(g2^k) = (k/23) sign(g2) for k = 1..22",
      all(m23.sign_of_23_element(power(m23.G2, k)) == (sg2 if k in qr else -sg2) for k in range(1, 23)))
check("sign(g2) = -1, sign(g3) = +1", (sg2, sg3) == (-1, +1))
facts["signs"] = {"g2 (23A of HJLPPZ)": sg2, "g3 (23B of HJLPPZ)": sg3}
neg = [(-i) % 23 for i in range(23)]
check("i -> -i exchanges the two cyclic Steiner systems, i -> 2i preserves each",
      golay.transport(golay.heptads(1), neg) == golay.heptads(-1)
      and golay.transport(golay.heptads(1), [(2 * i) % 23 for i in range(23)]) == golay.heptads(1))

# --- mod 2 modules -------------------------------------------------------------
G0 = golay.even_subcode(S)
basis = G0.basis()
words = []
for m in range(1, 1 << 11):
    v = 0
    for i in range(11):
        if m >> i & 1:
            v ^= basis[i]
    words.append(v)
by_weight = {}
for v in words:
    by_weight.setdefault(v.bit_count(), []).append(v)
check("G0 has nonzero weights 8, 12, 16 (506, 1288, 253 words)",
      {w: len(vs) for w, vs in by_weight.items()} == {8: 506, 12: 1288, 16: 253})
check("each weight class spans G0, so G0 is an irreducible F2[M23]-module",
      all(Span(vs).dim == 11 for vs in by_weight.values()))
check("G0 is self-orthogonal: G0 . G0 = 0", all((a & b).bit_count() % 2 == 0 for a in basis for b in basis))
code = golay.code_span(S)
check("perfect code: every vector is within distance 3 of exactly one codeword (sphere packing)",
      2**12 * (1 + 23 + 253 + 1771) == 2**23)
table = coh.fixed_space_table()
check("dim P0^g = dim G0^g + dim Q^g for every class (exactness of invariants)",
      all(v["defect"] == 0 for v in table.values()))
check("dim G0^g = dim Q^g for every class", all(v["G0"] == v["Q"] for v in table.values()))
img = Span(v ^ act_mask(v, m23.G1) for v in basis)
elts = set()
ib = img.basis()
for m in range(1 << len(ib)):
    v = 0
    for i in range(len(ib)):
        if m >> i & 1:
            v ^= ib[i]
    elts.add(v)
facts["im(1+g1) weights"] = dict(Counter(v.bit_count() for v in elts))
check("im(1+g1) on G0 is the [8,4,4] code on the 8 transpositions: weights {0:1, 8:14, 16:1}",
      facts["im(1+g1) weights"] == {0: 1, 8: 14, 16: 1})

# restriction of the g1-fixed even codewords to the heptad Fix(g1): the simplex code
from m23golay.f2 import kernel  # noqa: E402
fix_g1 = fixed_points(m23.G1)
fixmask = sum(1 << i for i in fix_g1)
restr = set()
kb = []
for mcombo in kernel([v ^ act_mask(v, m23.G1) for v in basis]):
    v = 0
    for i in range(11):
        if mcombo >> i & 1:
            v ^= basis[i]
    kb.append(v)
for m in range(1 << len(kb)):
    v = 0
    for i in range(len(kb)):
        if m >> i & 1:
            v ^= kb[i]
    restr.add(v & fixmask)
lines = [frozenset(i for i in fix_g1 if not (x >> i & 1)) for x in restr if x]
check("G0^{g1} restricted to Fix(g1) is the simplex code [7,3,4]; complements of its words are the lines of a Fano plane",
      len(restr) == 8 and all(x.bit_count() == 4 for x in restr if x) and len(set(lines)) == 7
      and all(sum(1 for l in lines if a in l and b in l) == 1 for a, b in itertools.combinations(sorted(fix_g1), 2)))

# --- theta characteristics and a Lagrangian (general symplectic lemma, genus 4) --------
# Theta characteristics = quadratic forms q on V = F_2^8 with polar form e; parity = Arf.
def _theta_lagrangian_check(g=4):
    n = 2 * g
    def e(x, y):
        return sum(((x >> i) & (y >> (i + g)) ^ (x >> (i + g)) & (y >> i)) & 1 for i in range(g)) & 1
    def q0(x):
        return sum((x >> i) & (x >> (i + g)) & 1 for i in range(g)) & 1
    forms = {c: [q0(x) ^ e(c, x) for x in range(1 << n)] for c in range(1 << n)}     # q_c = q0 + e(c, .)
    arf = {c: int(sum(1 for v in f if v == 0) != (1 << (n - 1)) + (1 << (g - 1))) for c, f in forms.items()}
    assert sum(arf.values()) == (1 << (g - 1)) * ((1 << g) - 1)                      # 120 odd
    # a Lagrangian on which q0 does not vanish: the graph {(x, Ax)} of a symmetric matrix A
    A = [0b0011, 0b0101, 0b1110, 0b0100]          # rows of a symmetric 4x4 matrix with diagonal (1,0,1,0)
    assert all((A[i] >> j & 1) == (A[j] >> i & 1) for i in range(g) for j in range(g))
    W = set()
    for x in range(1 << g):
        y = 0
        for i in range(g):
            if bin(A[i] & x).count("1") & 1:
                y |= 1 << i
        W.add(x | (y << g))
    assert len(W) == 1 << g and all(e(a, b) == 0 for a in W for b in W)
    cosets = {}
    for c, f in forms.items():
        key = tuple(f[w] for w in sorted(W))          # the restriction q_c | W (linear on W)
        cosets.setdefault(key, []).append(arf[c])
    profile = sorted((len(v), sum(v)) for v in cosets.values())
    special = [v for k, v in cosets.items() if not any(k)]
    return profile, special
profile, special = _theta_lagrangian_check()
check("theta characteristics modulo a Lagrangian W (g = 4): 16 cosets of 16; the forms vanishing on W "
      "are 16 even characteristics; each other coset has 8 odd and 8 even",
      profile == [(16, 0)] + [(16, 8)] * 15 and len(special) == 1 and sum(special[0]) == 0)

# --- class table ---------------------------------------------------------------
system = list(S)
duads = [frozenset(x) for x in itertools.combinations(range(23), 2)]
pc = invariants.permutation_characters()
classes = []
for name, count, ct in m23.cycle_index_data():
    g = m23.class_representatives()[name]
    on_h, on_d = invariants.cycle_type_on(system, g), invariants.cycle_type_on(duads, g)
    classes.append({"class": name, "elements": count, "centralizer": m23.CENTRALIZER_ORDERS[name],
                    "cycle_type": type_string(ct), "fixed_points": pc[name]["points"],
                    "fixed_duads": pc[name]["duads"], "fixed_heptads": pc[name]["heptads"],
                    "fixed_triads": pc[name]["triads"], "fixed_flags": pc[name]["flags"],
                    "cycle_type_on_heptads": type_string(on_h), "cycle_type_on_duads": type_string(on_d),
                    "dim_G0_fixed": table[name]["G0"], "dim_P0_fixed": table[name]["P0"]})
    check(f"class {name}: same cycle type on duads and heptads", on_h == on_d)
facts["classes"] = classes
check("permutation characters of duads and heptads coincide (Gassmann)",
      all(r["fixed_duads"] == r["fixed_heptads"] for r in classes))
check("triads and flags are NOT Gassmann equivalent", any(r["fixed_triads"] != r["fixed_flags"] for r in classes))
sub = invariants.subdegrees()
check("subdegrees: duads (1,42,210), heptads (1,112,140): the two actions are inequivalent",
      sub == {"heptads": [1, 112, 140], "duads": [1, 42, 210]})
facts["subdegrees"] = sub

# --- invariants ----------------------------------------------------------------
mol, sym = invariants.molien(14), invariants.partitions_at_most(14)
facts["molien"] = {"M23": mol, "S23": sym, "excess": [a - b for a, b in zip(mol, sym)]}
check("M23-invariants are symmetric in degrees <= 4; exactly one new invariant in degree 5",
      facts["molien"]["excess"][:6] == [0, 0, 0, 0, 0, 1])
facts["subset_orbits"] = invariants.subset_orbits()
check("M23 has 2 orbits on 5-subsets, of sizes 5313 and 28336",
      facts["subset_orbits"][5] == 2 and len(golay.five_subsets(S)) == 5313)

# --- decomposition group at the rational branch point ---------------------------
import numpy as np  # noqa: E402
E = m23.enumerate_elements()
g1a = np.array(m23.G1, dtype=np.uint8)
C = [tuple(int(x) for x in row) for row in E[(g1a[E] == E[:, g1a]).all(axis=1)]]
check("|C(g1)| = 2688", len(C) == 2688)


def orbit_lengths(objs, group):
    objs, out = set(objs), []
    while objs:
        o = objs.pop()
        orb, queue = {o}, [o]
        while queue:
            a = queue.pop()
            for g in group:
                b = act_set(a, g)
                if b not in orb:
                    orb.add(b)
                    queue.append(b)
        objs -= orb
        out.append(len(orb))
    return sorted(out)


facts["C(g1) orbits"] = {"points": orbit_lengths([frozenset([i]) for i in range(23)], C),
                         "duads": orbit_lengths(duads, C), "heptads": orbit_lengths(system, C)}
check("C(g1) fixes exactly one heptad and no duad",
      facts["C(g1) orbits"] == {"points": [7, 16], "duads": [8, 21, 112, 112], "heptads": [1, 28, 112, 112]})

# --- Nielsen class and real structure -----------------------------------------
ni = nielsen.nielsen_class()
check("|Ni(2A,23A,23B)| = 7", len(ni) == 7)
conj = nielsen.conjugation_permutation()
check("complex conjugation: 3 fixed classes and 2 transpositions",
      sorted(Counter(len(set([i, j])) for i, j in enumerate(conj)).items()) == [(1, 3), (2, 4)]
      and all(conj[conj[i]] == i for i in range(7)))
hj = nielsen.hjlppz_file_indices()
jk = [i for i, _ in nielsen.janik_table_indices()]
check("the seven triples of the HJLPPZ data file represent the seven classes", sorted(hj) == list(range(7)))
check("the seven rows of [Janik26, Table 1] represent the seven classes", sorted(jk) == list(range(7)))
ll = nielsen.lelay_indices()
check("the seven classes of [LeLay26] represent the seven classes; his real classes 3, 6, 7 are ours, "
      "and his distinguished class 6 is the rational one",
      sorted(ll) == list(range(7)) and {ll[2], ll[5], ll[6]} == {i for i, j in enumerate(conj) if i == j}
      and ll[5] == hj[0])
rows = []
for i, t in enumerate(ni):
    c0 = nielsen.real_structure(t)
    rep = coh.lagrangian_report(t, c0)
    row = {"index": i, "hjlppz_file_row": hj.index(i) + 1, "janik_orbit": jk.index(i),
           "lelay_id": ll.index(i) + 1,
           "conjugate": conj[i], "real": c0 is not None,
           "dim_W": rep["G0"]["parabolic"], "dim_Wdual": rep["Q"]["parabolic"],
           "dim_J2": rep["P0"]["parabolic"], "dim_H1_orbifold_G0": rep["G0"]["orbifold"]}
    if c0 is not None:
        r = nielsen.real_components(t, c0)
        row.update({"real_components": r, "fix_c0_cap_fix_g1": len(fixed_points(c0) & fixed_points(t[0])),
                    "dim_J2_real": rep["P0"]["real_part"], "dim_W_real": rep["G0"]["real_part"]})
        check(f"class {i}: dim J(R)[2] = g + r - 1", rep["P0"]["real_part"] == 4 + r - 1)
    rows.append(row)
    check(f"class {i}: dim W = 4 = g, dim Jac[2] = 8", row["dim_W"] == 4 and row["dim_J2"] == 8)
facts["nielsen"] = rows
rational = hj[0]
check("the rational cover (first HJLPPZ triple, Janik orbit 1) is real with r = 2; the other two real covers have r = 1",
      rows[rational]["real"] and rows[rational]["real_components"] == 2 and rows[rational]["janik_orbit"] == 1
      and sorted(r["real_components"] for r in rows if r["real"]) == [1, 1, 2])
check("complex conjugation acts on W as a transvection for the rational cover (dim W^c = 3)",
      rows[rational]["dim_W_real"] == 3)
facts["rational_class"] = rational

OUT.write_text(json.dumps(facts, indent=1) + "\n")
print(f"\nall checks passed; wrote {OUT}")
