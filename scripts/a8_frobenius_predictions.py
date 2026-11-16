#!/usr/bin/env python3
"""Classes of GL_4(2) = A_8 and the Frobenius predictions for the field K_W.

Part 1 enumerates GL_4(2) (20160 elements) in the model of data/pg32_a8_model.json and
records, for each of its 14 conjugacy classes: the characteristic polynomial on F_2^4, the
order, the cycle type on the 15 points of PG(3,2), and the cycle type on the 8 letters.
The cycle type on the 15 hyperplanes equals the one on points (g and its transpose are
conjugate in GL_n(q), and the class of g^-1 has the cycle type of g).

Part 2 reads the Frobenius table of X (data/zeta_X.json for p <= 79, and
data/rustmath_wpf_tables.json for the rows computed in RustMath) and lists, for every prime,
the characteristic polynomial Q_p of Frobenius on W up to reversal and the cycle types that a
polynomial g8 (degree 8) or g15 (degree 15) for K_W may have modulo p.

Part 3 uses the order of Jac(X)(F_p).  The 2-rank of Jac(X)(F_p) is the dimension of the
kernel of Frobenius - 1 on Jac(X)[2], and it is at most the 2-adic valuation v of
#Jac(X)(F_p) = P_p(1).  When v = 1 and x+1 divides P_p mod 2, Frobenius has a single Jordan
block for the eigenvalue 1 on Jac(X)[2], of size 2m where (x+1)^m exactly divides Q_p, and its
restriction to W is a single block of size m (the invariant subspaces of one block form a
chain).  That decides the order of Frobenius on Jac(X)[2] and, for m >= 2, the class on W.

Writes data/a8_frobenius_predictions.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q_MASK = 0b10011  # x^4 + x + 1, the first model of data/pg32_a8_model.json


def poly_str(mask: int) -> str:
    terms = []
    for e in range(mask.bit_length() - 1, -1, -1):
        if mask >> e & 1:
            terms.append("1" if e == 0 else "x" if e == 1 else f"x^{e}")
    return "+".join(terms) if terms else "0"


def pmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def pdivmod(a: int, b: int):
    q = 0
    db = b.bit_length()
    while a.bit_length() >= db:
        s = a.bit_length() - db
        q |= 1 << s
        a ^= b << s
    return q, a


IRREDUCIBLES = [0b11, 0b111, 0b1011, 0b1101, 0b10011, 0b11001, 0b11111]


def factor_str(mask: int) -> str:
    out = []
    for f in IRREDUCIBLES:
        e = 0
        while True:
            q, r = pdivmod(mask, f)
            if r:
                break
            mask, e = q, e + 1
        if e:
            out.append(f"({poly_str(f)})" + (f"^{e}" if e > 1 else ""))
    assert mask == 1
    return "".join(out)


def reverse(mask: int) -> int:
    n = mask.bit_length()
    return int(format(mask, f"0{n}b")[::-1], 2)


def parse_poly(s: str) -> int:
    """Parse a product of factors such as '(x+1)^2(x^3+x+1)' or a sum 'x^4+x+1' over F_2."""
    s = s.replace(" ", "").replace("*", "")
    if "(" not in s:
        s = "(" + s + ")"
    out, i = 1, 0
    while i < len(s):
        assert s[i] == "(", s
        j = s.index(")", i)
        m = 0
        for t in s[i + 1:j].split("+"):
            e = 0 if t == "1" else 1 if t == "x" else int(t[2:])
            m ^= 1 << e
        i, e = j + 1, 1
        if i < len(s) and s[i] == "^":
            k = i + 1
            while k < len(s) and s[k].isdigit():
                k += 1
            e, i = int(s[i + 1:k]), k
        for _ in range(e):
            out = pmul(out, m)
    return out


def ctype(perm) -> str:
    seen, lens = [False] * len(perm), []
    for i in range(len(perm)):
        if not seen[i]:
            n, j = 0, i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                n += 1
            lens.append(n)
    c = Counter(lens)
    return " ".join(f"{k}^{c[k]}" if c[k] > 1 else str(k) for k in sorted(c))


def charpoly(cols) -> int:
    """Characteristic polynomial of the 4x4 matrix over F_2 with columns cols (bit masks):
    det(x - M) by expansion over permutations, with entries in F_2[x] as bit masks."""
    from itertools import permutations
    total = 0
    for sigma in permutations(range(4)):
        term = 1
        for j in range(4):
            i = sigma[j]
            entry = (cols[j] >> i & 1) ^ (0b10 if i == j else 0)
            if entry == 0:
                term = 0
                break
            term = pmul(term, entry)
        total ^= term
    return total


def main():
    model = json.loads((ROOT / "data" / "pg32_a8_model.json").read_text())[poly_str(Q_MASK)]
    # points: exponent e <-> the vector x^e mod Q
    vec = [1]
    for _ in range(14):
        v = vec[-1] << 1
        if v & 0b10000:
            v ^= Q_MASK
        vec.append(v)
    exp_of = {v: e for e, v in enumerate(vec)}
    lines = [tuple(l) for l in model["lines"]]
    line_index = {frozenset(l): i for i, l in enumerate(lines)}
    spreads = [tuple(s) for s in model["spreads"]]
    spread_index = {frozenset(s): i for i, s in enumerate(spreads)}
    families = [frozenset(f) for f in model["letter_families"]]
    family_index = {f: i for i, f in enumerate(families)}

    def point_perm(cols):
        def apply(v):
            r = 0
            for j in range(4):
                if v >> j & 1:
                    r ^= cols[j]
            return r
        return tuple(exp_of[apply(vec[e])] for e in range(15))

    def on_letters(pp):
        lp = [line_index[frozenset(pp[a] for a in l)] for l in lines]
        sp = [spread_index[frozenset(lp[a] for a in s)] for s in spreads]
        return tuple(family_index[frozenset(sp[a] for a in f)] for f in families)

    # generators: multiplication by x (Singer cycle) and a transvection e0 -> e0 + e1
    gens = [(vec[1], vec[2], vec[3], vec[4]), (0b0011, 0b0010, 0b0100, 0b1000)]

    def mul(a, b):  # first a, then b (columns are images of the basis)
        def apply(cols, v):
            r = 0
            for j in range(4):
                if v >> j & 1:
                    r ^= cols[j]
            return r
        return tuple(apply(b, a[j]) for j in range(4))

    identity = (1, 2, 4, 8)
    seen, frontier = {identity}, [identity]
    while frontier:
        nxt = []
        for g in frontier:
            for s in gens:
                h = mul(g, s)
                if h not in seen:
                    seen.add(h)
                    nxt.append(h)
        frontier = nxt
    assert len(seen) == 20160, len(seen)

    classes = Counter()
    for g in seen:
        pp = point_perm(g)
        order, h = 1, g
        while h != identity:
            h, order = mul(h, g), order + 1
        fixed = sum(1 for e in range(15) if pp[e] == e) + 1
        classes[(charpoly(g), order, ctype(pp), ctype(on_letters(pp)), fixed.bit_length() - 1)] += 1
    assert len(classes) == 14 and sum(classes.values()) == 20160
    rows = [{"charpoly": poly_str(cp), "factored": factor_str(cp), "order": o,
             "cycle_type_15": c15, "cycle_type_8": c8, "dim_fixed": fd, "class_size": n}
            for (cp, o, c15, c8, fd), n in sorted(classes.items(), key=lambda kv: (kv[0][1], kv[0][0]))]

    by_pair = defaultdict(list)
    for (cp, o, c15, c8, fd), n in classes.items():
        by_pair[min(cp, reverse(cp))].append((o, c15, c8, fd))

    # Part 2: the primes
    primes = {}
    zx = json.loads((ROOT / "data" / "zeta_X.json").read_text())
    rm = json.loads((ROOT / "data" / "rustmath_wpf_tables.json").read_text())
    recount = json.loads((ROOT / "data" / "rustmath_crosscheck.json").read_text())
    recounted = {int(p) for p, r in recount.items() if r.get("equal_to_rustmath")}
    owner = {int(r["p"]) for r in zx if r.get("status") == "ok"}
    owner_P = {int(r["p"]): r["P"] for r in zx if r.get("status") == "ok"}
    for r in rm["X"]:
        if "Q" not in r:
            continue
        p = int(r["p"])
        q = parse_poly(r["Q"])
        pm2 = parse_poly(r["P_mod_2"])
        assert pmul(q, reverse(q)) == pm2, (p, r["Q"], r["P_mod_2"])
        key = min(q, reverse(q))
        opts = sorted(set(by_pair[key]))
        a1, a2, a3, a4 = r["a"]
        P = [1, a1, a2, a3, a4, p * a3, p * p * a2, p ** 3 * a1, p ** 4]
        if p in owner_P:
            assert owner_P[p] == P, p
        order_J = sum(P)
        v2 = (order_J & -order_J).bit_length() - 1
        m = 0
        rest = q
        while True:
            quo, rem = pdivmod(rest, 0b11)
            if rem:
                break
            rest, m = quo, m + 1
        odd = {1: 1, 0b111: 3, 0b1011: 7, 0b1101: 7, 0b10011: 15, 0b11001: 15, 0b11111: 5}[rest]
        row = {
            "Q_or_reverse": [poly_str(key), poly_str(reverse(key))],
            "factored": factor_str(key),
            "order_of_Jac_Fp": order_J,
            "v2_order_of_Jac_Fp": v2,
            "counted_by": ("both" if p in owner or p in recounted else "RustMath only"),
        }
        if m == 0:
            # no eigenvalue 1: P mod 2 is squarefree here, Frobenius is semisimple on Jac[2]
            assert pmul(q, reverse(q)) == pm2 and rest in (0b10011, 0b11001, 0b11111)
            if rest != 0b11111:
                row["order_on_Jac2"] = odd
        elif v2 == 1:
            opts = [o for o in opts if o[3] == 1]
            unipotent = 1
            while unipotent < 2 * m:
                unipotent *= 2
            row["order_on_Jac2"] = odd * unipotent
            row["decided_by"] = "v2(#Jac(F_p)) = 1: one Jordan block for the eigenvalue 1"
        row["cycle_types_8"] = sorted({c8 for _, _, c8, _ in opts})
        row["cycle_types_15"] = sorted({c15 for _, c15, _, _ in opts})
        primes[p] = row
    out = {
        "description": "Classes of GL_4(2) = A_8 and the cycle types allowed for Frobenius in K_W "
                       "(8 letters, and 15 points or hyperplanes), from P_p mod 2 = Q Q^rev.",
        "classes": rows,
        "primes": {str(p): primes[p] for p in sorted(primes)},
    }
    (ROOT / "data" / "a8_frobenius_predictions.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'order':>5} {'size':>5}  {'charpoly':24} {'on 8 letters':12} {'on 15 points':14} dim fixed")
    for r in rows:
        print(f"{r['order']:>5} {r['class_size']:>5}  {r['factored']:24} {r['cycle_type_8']:12} "
              f"{r['cycle_type_15']:14} {r['dim_fixed']}")
    print()
    print(f"{'p':>4}  {'Q_p up to reversal':24} v2 {'ord J[2]':>8}  {'on 8 letters':24} counted by")
    for p in sorted(primes):
        r = primes[p]
        c8 = " or ".join(r["cycle_types_8"]) if len(r["cycle_types_8"]) <= 2 else f"{len(r['cycle_types_8'])} unipotent types"
        print(f"{p:>4}  {r['factored']:24} {r['v2_order_of_Jac_Fp']:>2} {str(r.get('order_on_Jac2', '?')):>8}  {c8:24} {r['counted_by']}")


if __name__ == "__main__":
    main()
