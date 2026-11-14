#!/usr/bin/env python3
"""Count the connected components of X(R) directly from the equation F(T,V) = 0.

For real t != 0 the fibre has exactly 7 real points.  Sort the real roots of
F(t, .) on the circle P^1(R) and follow them as t runs over
(0, +inf) u (-inf, 0).  The cyclic order changes only
  * at a real node with real branches (a crunode): two adjacent roots cross;
  * at a real zero of the leading coefficient: one root passes through infinity.
At t = 0 the fibre is g7 * g8^2: a simple real root continues from t > 0 to
t < 0, and at a real root of g8 two real roots merge on one side of 0.
The number of cycles of the resulting matching is the number of ovals.

This is independent of all group theory; nielsen.real_components predicts 2
for the rational cover.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mpmath as mp  # noqa: E402
from m23golay import data  # noqa: E402

mp.mp.dps = 90
F = data.F()
h62 = [int(c) for c in json.loads((Path(__file__).resolve().parent.parent / "data" / "h62.json").read_text())["h62"]]


def fibre(t):
    return [sum(mp.mpf(F[i][j]) * t**i for i in range(5)) for j in range(24)]   # ascending in V


def real_roots(poly_asc, tol=mp.mpf(10) ** -30):
    cs = list(poly_asc)
    while abs(cs[-1]) == 0:
        cs.pop()
    roots = mp.polyroots(cs[::-1], maxsteps=500, extraprec=400)
    scale = max(1, max(abs(r) for r in roots))
    return sorted(r.real for r in roots if abs(r.imag) < tol * scale)


def partial(t, v, dt, dv):
    total = mp.mpf(0)
    for i in range(dt, 5):
        for j in range(dv, 24):
            c = F[i][j]
            if c:
                ci = mp.mpf(1)
                for a in range(dt):
                    ci *= (i - a)
                for a in range(dv):
                    ci *= (j - a)
                total += c * ci * t ** (i - dt) * v ** (j - dv)
    return total


events = []          # (t, kind, payload)
# crunodes
node_ts = real_roots([mp.mpf(c) for c in h62])
ncrun = nacn = 0
for t0 in node_ts:
    delta = mp.mpf(10) ** -12 * max(1, abs(t0))
    # the double root: the real root of d/dV F(t0, V) that is also a root of F(t0, V)
    fv = fibre(t0)
    dfv = [k * c for k, c in enumerate(fv)][1:]
    cands = real_roots(dfv)
    v0 = min(cands, key=lambda v: abs(mp.polyval(fv[::-1], v)) / (1 + abs(v)) ** 23)
    hess = partial(t0, v0, 1, 1) ** 2 - partial(t0, v0, 2, 0) * partial(t0, v0, 0, 2)
    if hess > 0:
        ncrun += 1
        events.append((t0, "cross", v0))
    else:
        nacn += 1
# poles of v
lc = [mp.mpf(F[i][23]) for i in range(5)]
for t0 in real_roots(lc):
    events.append((t0, "pole", None))

pos_events = sorted([e for e in events if e[0] > 0], key=lambda e: e[0])
neg_events = sorted([e for e in events if e[0] < 0], key=lambda e: e[0])


def apply_events(labels, evs):
    """labels[k] = label of the k-th smallest real root; update across events (t increasing)."""
    for t0, kind, v0 in evs:
        d = mp.mpf(10) ** -9 * max(1, abs(t0))
        before, after = real_roots(fibre(t0 - d)), real_roots(fibre(t0 + d))
        assert len(before) == len(after) == 7, (len(before), len(after))
        if kind == "cross":
            order = sorted(range(7), key=lambda k: abs(before[k] - v0))[:2]
            a, b = sorted(order)
            assert b == a + 1, "crossing roots are not adjacent"
            labels[a], labels[b] = labels[b], labels[a]
        else:
            big = max(abs(x) for x in before)
            if abs(before[-1]) == big and before[-1] > 0 and after[0] < 0 and abs(after[0]) == max(abs(x) for x in after):
                labels = [labels[-1]] + labels[:-1]          # top root wraps to the bottom
            elif abs(before[0]) == big and before[0] < 0 and after[-1] > 0:
                labels = labels[1:] + [labels[0]]            # bottom root wraps to the top
            else:
                raise RuntimeError("could not classify pole event")
    return labels


eps = mp.mpf(10) ** -7
big = mp.mpf(10) ** 9
# roots at t = 0: simple real roots (from g7) and real double roots (from g8)
f0 = fibre(mp.mpf(0))
_r0 = real_roots(f0, tol=mp.mpf(10) ** -20)
r0 = []                                    # distinct real roots (double roots appear twice numerically)
for v in _r0:
    if not r0 or abs(v - r0[-1]) > mp.mpf(10) ** -15:
        r0.append(v)


def limit_class(v):
    """Index of the real root of F(0, .) closest to v."""
    return min(range(len(r0)), key=lambda k: abs(r0[k] - v))


start = real_roots(fibre(eps))            # t = 0+
labels = list(range(7))                   # label = sorted position at t = 0+
labels = apply_events(labels, pos_events) # now at t = +big
top = real_roots(fibre(big))
bot = real_roots(fibre(-big))
# across t = infinity: the 7 real roots of F_inf vary continuously in 1/t; match by value
finf = real_roots([mp.mpf(F[4][j]) for j in range(24)])
assert len(finf) == 7
to_inf_top = [min(range(7), key=lambda k: abs(finf[k] - v)) for v in top]
to_inf_bot = [min(range(7), key=lambda k: abs(finf[k] - v)) for v in bot]
assert sorted(to_inf_top) == sorted(to_inf_bot) == list(range(7))
labels_bot = [None] * 7
for k in range(7):
    labels_bot[to_inf_bot.index(to_inf_top[k])] = labels[k]
labels_end = apply_events(labels_bot, neg_events)     # now at t = 0-
end = real_roots(fibre(-eps))

# build the graph
adj = {("A", k): [] for k in range(7)}
adj.update({("B", k): [] for k in range(7)})


def edge(u, v):
    adj[u].append(v)
    adj[v].append(u)


for k in range(7):                       # through infinity
    edge(("A", labels_end[k]), ("B", k))
ca = [limit_class(v) for v in start]
cb = [limit_class(v) for v in end]
for cls in set(ca) | set(cb):
    ia = [k for k in range(7) if ca[k] == cls]
    ib = [k for k in range(7) if cb[k] == cls]
    if len(ia) == 1 and len(ib) == 1:
        edge(("A", ia[0]), ("B", ib[0]))             # unramified real point over 0
    elif len(ia) == 2 and not ib:
        edge(("A", ia[0]), ("A", ia[1]))             # fold on the positive side
    elif len(ib) == 2 and not ia:
        edge(("B", ib[0]), ("B", ib[1]))             # fold on the negative side
    else:
        raise RuntimeError(f"unexpected local structure at t = 0: {ia} {ib}")
assert all(len(v) == 2 for v in adj.values())
seen, comps = set(), 0
for s in adj:
    if s in seen:
        continue
    comps += 1
    stack = [s]
    while stack:
        u = stack.pop()
        if u not in seen:
            seen.add(u)
            stack.extend(adj[u])
print(f"real nodes: {len(node_ts)} ({ncrun} crunodes, {nacn} acnodes); real poles of v: {len(real_roots(lc))}")
print(f"real roots of F(0,V): {len(r0)} distinct; through-points: {sum(1 for c in set(ca) if ca.count(c)==1)}; folds: +side {sum(1 for c in set(ca) if ca.count(c)==2)}, -side {sum(1 for c in set(cb) if cb.count(c)==2)}")
print("number of connected components of X(R):", comps)
