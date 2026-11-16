#!/usr/bin/env python3
"""The image of Galois on Jac(X)[2], and Frobenius orbits on the 120 tritangent planes.

Linear algebra over F_2 behind two statements.

1. The stabilizer P of the Lagrangian W in Sp_8(F_2) is R : GL(W), where the unipotent radical
   R is the space of symmetric tensors in W (x) W (dimension 10).  The tensors w (x) w span it,
   and GL(W) is transitive on them.  So a subgroup of P that maps onto GL(W) and contains one
   element 1 + w (x) w is all of P.  For X such an element is Frobenius^7 at p = 17 (also 73,
   127): #Jac(X)(F_p) is 2 times an odd number while (x+1)^2 exactly divides P_p mod 2, so
   Frobenius has one Jordan block of size 2 for the eigenvalue 1 and order 14 on Jac(X)[2].
   The orders are in data/a8_frobenius_predictions.json (scripts/a8_frobenius_predictions.py).

   The orbits of P on the 256 theta characteristics are also listed: the 16 even ones that
   vanish on W, the other 120 even ones, and the 120 odd ones.

2. The orbit lengths of such a Frobenius on the 120 odd theta characteristics (the tritangent
   planes of the canonical model) and on the 136 even ones, next to the semisimple alternative
   and the order 15 case used in the RustMath handoff.

Model: J[2] = W + W^dual with coordinates (x, y), pairing <(x,y),(x',y')> = x.y' + x'.y.  The
quadratic forms with this polar form are q_{a,b}(x,y) = x.y + a.x + b.y, with Arf invariant a.b
and restriction a to W.
"""
from itertools import product
from collections import Counter

def dot(u, v): return sum(i & j for i, j in zip(u, v)) & 1
def matvec(M, v): return tuple(sum(M[i][j] & v[j] for j in range(4)) & 1 for i in range(4))
def matmul(A, B): return [[sum(A[i][k] & B[k][j] for k in range(4)) & 1 for j in range(4)] for i in range(4)]
def transpose(A): return [[A[j][i] for j in range(4)] for i in range(4)]
def inverse(A):
    n = 4; M = [list(A[i]) + [int(i == j) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c]); M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c]: M[r] = [a ^ b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]

def companion(coeffs):  # monic x^k + c_{k-1} x^{k-1} + ... + c_0, coeffs = [c_0..c_{k-1}]
    k = len(coeffs); M = [[0] * k for _ in range(k)]
    for i in range(1, k): M[i][i - 1] = 1
    for i in range(k): M[i][k - 1] = coeffs[i]
    return M

def block(*blocks):
    n = sum(len(b) for b in blocks); M = [[0] * n for _ in range(n)]; o = 0
    for b in blocks:
        for i in range(len(b)):
            for j in range(len(b)): M[o + i][o + j] = b[i][j]
        o += len(b)
    return M

vecs = list(product((0, 1), repeat=4))
points = [(x, y) for x in vecs for y in vecs]

def orbits(g, label):
    # g: function on (x, y); forms as value tables are heavy, so act on (a, b) through q o g^-1 by fitting
    ginv = {g(v): v for v in points}
    def q(a, b, v): return (dot(v[0], v[1]) + dot(a, v[0]) + dot(b, v[1])) & 1
    table = {}
    for a in vecs:
        for b in vecs:
            table[tuple(q(a, b, v) for v in points)] = (a, b)
    def act(ab):
        a, b = ab
        return table[tuple(q(a, b, ginv[v]) for v in points)]
    odd = [(a, b) for a in vecs for b in vecs if dot(a, b) == 1]
    even = [(a, b) for a in vecs for b in vecs if dot(a, b) == 0]
    out = {}
    for name, S in (("odd", odd), ("even", even)):
        seen, lens = set(), []
        for s in S:
            if s in seen: continue
            n, t = 0, s
            while t not in seen:
                seen.add(t); t = act(t); n += 1
            lens.append(n)
        out[name] = dict(sorted(Counter(lens).items()))
    # order of g
    n, cur = 1, {v: g(v) for v in points}
    while any(cur[v] != v for v in points):
        cur = {v: g(cur[v]) for v in points}; n += 1
    print(f"{label}: order on J[2] = {n}; orbits on the 120 odd: {out['odd']}; on the 136 even: {out['even']}")

def frob(A, N=None):
    """x -> A x + N y, y -> A^-T y: preserves W = {y = 0} and the pairing when A^-1 N is symmetric."""
    AinvT = transpose(inverse(A))
    def g(v):
        x, y = v
        x2 = matvec(A, x)
        if N is not None: x2 = tuple(i ^ j for i, j in zip(x2, matvec(N, y)))
        return (x2, matvec(AinvT, y))
    # check symplectic
    for v in points[::7]:
        for w in points[::11]:
            gv, gw = g(v), g(w)
            assert (dot(v[0], w[1]) + dot(w[0], v[1])) & 1 == (dot(gv[0], gw[1]) + dot(gw[0], gv[1])) & 1
    return g

one = [[1]]
c3 = companion([1, 1, 0])        # x^3 + x + 1
c2 = companion([1, 1])           # x^2 + x + 1
c4 = companion([1, 1, 0, 0])     # x^4 + x + 1
E11 = [[1 if (i, j) == (0, 0) else 0 for j in range(4)] for i in range(4)]

A7 = block(one, c3)              # (x+1)(x^3+x+1), fixed vector e_0
orbits(frob(A7), "type (x+1)(cubic), semisimple (order 7)      ")
orbits(frob(A7, E11), "type (x+1)(cubic), one Jordan block (order 14) ")
orbits(frob(c4), "type x^4+x+1 (order 15)                       ")

# Statement 1: the tensors w (x) w span the 10-dimensional space of symmetric tensors.
rows = []
for w in vecs:
    if any(w):
        rows.append(sum((w[i] & w[j]) << (4 * i + j) for i in range(4) for j in range(4)))
basis = []
for r in rows:
    for b in basis:
        r = min(r, r ^ b)
    if r:
        basis.append(r)
print(f"span of the 15 tensors w (x) w: dimension {len(basis)} (symmetric tensors: 10)")
assert len(basis) == 10

# The orbits of the full stabilizer P on the 256 quadratic forms (theta characteristics).
def form_action(g):
    ginv = {g(v): v for v in points}
    def q(a, b, v): return (dot(v[0], v[1]) + dot(a, v[0]) + dot(b, v[1])) & 1
    table = {tuple(q(a, b, v) for v in points): (a, b) for a in vecs for b in vecs}
    return {(a, b): table[tuple(q(a, b, ginv[v]) for v in points)] for a in vecs for b in vecs}

ident = [[int(i == j) for j in range(4)] for i in range(4)]
shear = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
gens = [form_action(frob(c4)), form_action(frob(shear)), form_action(frob(ident, E11))]
seen, sizes = set(), []
for start in gens[0]:
    if start in seen:
        continue
    orbit, frontier = {start}, [start]
    while frontier:
        nxt = []
        for f in frontier:
            for g in gens:
                h = g[f]
                if h not in orbit:
                    orbit.add(h)
                    nxt.append(h)
        frontier = nxt
    seen |= orbit
    a, b = start
    sizes.append((len(orbit), "odd" if dot(a, b) else "even", "zero on W" if not any(a) else "nonzero on W"))
print("orbits of the stabilizer of W on the 256 forms:", sorted(sizes))
assert sorted(n for n, _, _ in sizes) == [16, 120, 120]
