#!/usr/bin/env python3
"""Exact sizes and real points of all three-point Nielsen classes of M23 that can
carry a cover defined over Q.

Input: data/m23_three_point_passports.json (class vectors and structure constants n
from the character table, made by scripts/gap/m23_passports.g).  For each class
vector this script computes, by enumeration in the permutation group:

  * survivors  = #{a in C1 : a b in C3^-1}  for a fixed b in C2; must equal n |C(b)|
                 (an independent check of the character-table value and of the
                 separation of the classes D, D^-1 by the trace on the Golay code);
  * size       = number of inner classes of GENERATING triples.  A subgroup of M23
                 containing an element of order not in {1, 11, 23} is M23 if and only
                 if it is transitive on the 23 points (all maximal subgroups other
                 than 23:11 are intransitive);
  * real       = number of classes fixed by complex conjugation:
        type "rational" (three real branch points): classes of (a, b) for which an
            involution inverts both a and b;
        type "conjugate_pair" (classes R, D, D^-1 at 0 and a pair of complex conjugate
            points): involutions c with a := c b c b^-1 in R and <a, b> = M23,
            divided by |C(b)|.

Writes data/m23_three_point_nielsen.json.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import golay, m23  # noqa: E402
from m23golay.perm import inv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N = 23
E = m23.enumerate_elements()
IDX = np.arange(N, dtype=np.uint8)[None, :]


def signatures(X):
    X2 = np.take_along_axis(X, X, axis=1)
    X3 = np.take_along_axis(X2, X, axis=1)
    return ((X == IDX).sum(axis=1).astype(np.int32) * 10000 + (X2 == IDX).sum(axis=1) * 100
            + (X3 == IDX).sum(axis=1))


def sig_code(name):
    s = next(sig for sig, lab in m23.SIGNATURES.items() if lab == name)
    return s[0] * 10000 + s[1] * 100 + s[2]


SIG = signatures(E)
CLASS = {name: E[SIG == sig_code(name)] for name in m23.SIGNATURES.values()}
INVOLUTIONS = CLASS["2A"]

# systematic basis of the even Golay code: v_k has a 1 at pivot[k] and 0 at the other pivots
basis = golay.even_subcode(m23.steiner()).basis()
rows = list(basis)
pivots = []
for k in range(len(rows)):
    piv = rows[k].bit_length() - 1
    pivots.append(piv)
    for j in range(len(rows)):
        if j != k and rows[j] >> piv & 1:
            rows[j] ^= rows[k]
V = np.array([[(r >> x) & 1 for x in range(N)] for r in rows], dtype=np.uint8)      # 11 x 23
PIV = np.array(pivots)


def golay_trace(X):
    """tr(x | G_0) in F_2 for each row x of X:  sum_k v_k[x^-1(pivot_k)]."""
    Xinv = np.argsort(X, axis=1)
    return V[np.arange(len(PIV))[None, :], Xinv[:, PIV]].sum(axis=1) & 1


def transitive(A, b):
    """Is <a, b> transitive on 23 points, for each row a of A?"""
    n = len(A)
    reach = np.zeros((n, N), dtype=bool)
    reach[:, 0] = True
    Ainv = np.argsort(A, axis=1)
    binv = np.argsort(np.array(b))
    for _ in range(N):
        new = reach | np.take_along_axis(reach, Ainv, axis=1) | reach[:, binv]
        if (new == reach).all():
            break
        reach = new
    return reach.all(axis=1)


def rep(name):
    """A representative; for a split type this fixes which class is called D."""
    key = name[:-1].upper() + ("AB" if name[:-1] in ("7", "11", "14", "15", "23") else "A")
    return m23.class_representatives()[key], key


def centralizer_order(key):
    return m23.CENTRALIZER_ORDERS[key]


STEINER = [frozenset(h) for h in m23.steiner()]


def cyclic_design(x):
    """Class invariant for elements of order 11: label each 11-cycle of x by Z/11 through x
    and record the positions of the heptads on it.  The family is translation invariant,
    and replacing x by x^k multiplies it by 1/k; it is preserved exactly by the squares."""
    from collections import Counter
    from m23golay.perm import cycles_of
    cyc = cycles_of(x)
    length = max(len(c) for c in cyc)
    fam = Counter()
    for c in cyc:
        if len(c) != length:
            continue
        pos, y = {}, c[0]
        for i in range(length):
            pos[y] = i
            y = x[y]
        for h in STEINER:
            s_ = tuple(sorted(pos[v] for v in h if v in pos))
            if len(s_) >= 3:
                fam[s_] += 1
    return frozenset(fam.items())


def in_class_of(X, b, key):
    """Rows of X conjugate to b: same cycle type and, for the split types, the same
    trace on the even Golay code (7, 14, 15, 23) or the same cyclic design (11)."""
    ok = signatures(X) == sig_code(key)
    if key == "11AB":
        ref = cyclic_design(b)
        sub = np.flatnonzero(ok)
        keep = np.array([cyclic_design(tuple(int(v) for v in X[i])) == ref for i in sub], dtype=bool)
        ok[sub] = keep
    elif key.endswith("AB"):
        tb = int(golay_trace(np.array([b], dtype=np.uint8))[0])
        sub = np.flatnonzero(ok)
        ok[sub] = golay_trace(X[sub]) == tb
    return ok


# the invariants separate each pair D, D^-1
for key in ("7AB", "14AB", "15AB", "23AB"):
    g = np.array([m23.class_representatives()[key], inv(m23.class_representatives()[key])], dtype=np.uint8)
    t = golay_trace(g)
    assert t[0] != t[1], key
g11 = m23.class_representatives()["11AB"]
assert cyclic_design(g11) != cyclic_design(inv(g11))

spec = json.loads((ROOT / "data" / "m23_three_point_passports.json").read_text())
out = []
for row in spec["passports"]:
    c1, c2, c3 = row["classes"]
    n = Fraction(row["structure_constant"])
    if row["type"] == "conjugate_pair":
        b, bkey = rep(c2)
        barr = np.array(b, dtype=np.uint8)
        A = CLASS[c1.upper()]
        P = barr[A]                                    # a b
        ok = in_class_of(P, b, bkey)                   # a b in D  <=>  g3 = (a b)^-1 in D^-1
        cent = centralizer_order(bkey)
        surv = A[ok]
        gen = transitive(surv, b)
        # real classes: involutions c with a = c b c b^-1 in R and <a, b> = M23
        binv = np.array(inv(b), dtype=np.uint8)
        Aa = binv[INVOLUTIONS][:, :]                   # placeholder shape
        cbc = INVOLUTIONS[np.arange(len(INVOLUTIONS))[:, None], barr[INVOLUTIONS]]   # c b c: x -> c[b[c[x]]]
        Aa = np.take_along_axis(binv[None, :].repeat(len(INVOLUTIONS), 0), cbc, axis=1)  # (c b c) b^-1: x -> binv[cbc[x]]
        okr = signatures(Aa) == sig_code(c1.upper())
        real_pairs = int(transitive(Aa[okr], b).sum())
    else:
        # put the class with the smallest centraliser in the middle
        names = sorted([c1, c2, c3], key=lambda c: centralizer_order(c.upper()))
        b, bkey = rep(names[0])
        barr = np.array(b, dtype=np.uint8)
        cent = centralizer_order(bkey)
        A = CLASS[names[1].upper()]
        P = barr[A]
        ok = signatures(P) == sig_code(names[2].upper())
        surv = A[ok]
        gen = transitive(surv, b)
        # real classes: involutions c inverting b, and a = c u (u an involution) in the class of a
        cb = INVOLUTIONS[np.arange(len(INVOLUTIONS))[:, None], barr[INVOLUTIONS]]       # c b c
        inverting = INVOLUTIONS[(cb == np.array(inv(b), dtype=np.uint8)[None, :]).all(axis=1)]
        real_pairs = 0
        for c in inverting:
            Aa = INVOLUTIONS[:, c]                      # a = c u : x -> u[c[x]]
            oka = signatures(Aa) == sig_code(names[1].upper())
            Aa = Aa[oka]
            Pa = barr[Aa]
            okp = signatures(Pa) == sig_code(names[2].upper())
            real_pairs += int(transitive(Aa[okp], b).sum())
    assert len(surv) == n * cent, (row["classes"], len(surv), n, cent)
    assert int(gen.sum()) % cent == 0 and real_pairs % cent == 0
    size, real = int(gen.sum()) // cent, real_pairs // cent
    assert real <= size and (size - real) % 2 == 0
    rec = {"type": row["type"], "classes": [c.upper() for c in row["classes"]], "genus23": row["genus23"],
           "structure_constant": row["structure_constant"], "size": size, "real": real}
    out.append(rec)
    print(f"{row['type']:15s} {'-'.join(rec['classes']):12s} g={rec['genus23']} n={row['structure_constant']:>8s} "
          f"size={size:6d} real={real}", flush=True)

chk = {tuple(r["classes"]): r for r in out}
assert chk[("2A", "23A", "23B")]["size"] == 7 and chk[("2A", "23A", "23B")]["real"] == 3
ref = json.loads((ROOT / "data" / "passports_C_23A_23B.json").read_text())
for cls, r in ref.items():
    assert chk[(cls, "23A", "23B")]["size"] == r["size"] and chk[(cls, "23A", "23B")]["real"] == r["real_classes"], cls
(ROOT / "data" / "m23_three_point_nielsen.json").write_text(json.dumps(
    {"description": "inner Nielsen classes of generating triples of M23 for the rational three-point class vectors; "
                    "'real' = classes fixed by complex conjugation", "passports": out}, indent=0) + "\n")
print("all checks passed;", len(out), "class vectors")
