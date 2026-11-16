# Golay-code structure of the Mathieu M23 cover

Working repository for a manuscript that builds on

* X. Huang, B. Jackson, K.-H. Lee, B. Poonen, R. Pries, S. Zhang,
  *The Mathieu group M23 is a Galois group over Q*, arXiv:2608.08538v2 ("HJLPPZ");
* J. A. Janik, *Real structure and the Galois fixed point in the Nielsen class
  (2,23A,23B) of M23* (`refs/`);
* the research note `notes/M23_Golay_Invariants_Research_Note.tex` on Golay invariants.

The manuscript is `paper/golay_m23.tex` (PDF: `paper/golay_m23.pdf`).

## Results in one page

| | statement | status |
|---|---|---|
| A | For any degree 23 cover X -> P^1 with monodromy in M23, the even Golay code cuts out a Galois-stable Lagrangian W in Jac(X)[2]. The proof uses that the Golay code is perfect (unique coset leaders). | proved |
| B | For the HJLPPZ curve (genus 4), Galois acts on W = F_2^4 through all of GL_4(F_2) = A_8; End(Jac X) = Z; complex conjugation is a transvection on W; the cover has 15 lifts to 2^11:M23 with the same inertia classes, conjugate over Q. X has good reduction at 5, 7, 11 and every other prime of the table, so the A_8 field is unramified outside {2, 3, 23}. | computed, with a reduction certificate at each prime; reproduced independently in RustMath |
| C | Roots in Frobenius order at a prime of irreducibility carry one of exactly two Steiner systems (the two cyclic Golay codes). Certified p-adic labelling. J5(f1) = 524492, J5(f2) = 61916; degree 253 heptad resolvents; J5 of the regular family is n(t)/c23(t) with n an explicit integer quartic. | proved + computed, checked at independent primes |
| D | Duad and heptad stabilizers are Gassmann equivalent: arithmetically equivalent non-isomorphic fields of degree 253, and two genus 46 curves over Q with isogenous Jacobians, one with a rational point above t = 0 and one without. 253 is the smallest such index in M23. | proved (group theory) + verified on f1, f2 |
| E | The real roots of an M23-polynomial form a heptad. Of the three real covers in the Nielsen class, the rational one is the only one with disconnected real locus (2 ovals vs 1). | group theory + numerical check on F(T,V) |
| F | All 57 rational three-point class vectors of M23: exact Nielsen class sizes and real points. Every vector containing 8A has no real point; (2A,14A,14B) has 17 covers of genus 2, 3 real. | enumeration, checked against the character table |

What is not claimed: no new regular M23-extension of Q(t), and no explanation of
why the Hurwitz scheme has a rational point.

## Relation to the research note

Every claim of the note was checked by code (`scripts/verify_group_theory.py`):
the code parameters and weight distribution, the Steiner counts 253/77/21/5, the
5313 five-subsets, the stabilizers of J5 and J7, the first non-symmetric
invariant in degree 5 (Molien series: 0,0,0,0,0,1,3,8,18,...), the genus
computations, the heptad resolvent, and the answer 1+6 to the orbit question.
The note's own conclusion stands: covariants such as the gradient of J7 only
give Tschirnhaus transforms (the explicit transform for f2 is in
`data/tschirnhaus_f2.json`; its coefficients have a 125-digit denominator).
The note's phases 1 to 5 are carried out here; phase 6 (other passports) is not.
The manuscript keeps the note's material as Section 8 and puts the new
structural results first.

## Layout

```
m23golay/     library: permutations, Golay codes, M23 from the HJLPPZ generators,
              Nielsen class and real structure, mod 2 cohomology, p-adic labelling,
              point counting on F(T,V) = 0
scripts/      verification and data-generation scripts (see below)
data/         computed data (JSON): group facts, L-polynomials, resolvents, invariants
paper/        manuscript; tables are generated from data/ by scripts/make_tables.py
notes/        the original research note
refs/         the author's earlier paper
tests/        fast regression tests
```

## Reproducing

Requirements: Python 3.11+, `numpy`, `python-flint`, `mpmath` (and `pytest`);
a LaTeX installation for the paper. PARI/GP and GAP were used only for
cross-checks.

```bash
pip install numpy python-flint mpmath pytest
```

```bash
scripts/fetch_external.sh
```

The second command clones the HJLPPZ data repository into `external/` (it is not
redistributed here). Then:

| command | what it does | time |
|---|---|---|
| `python3 -m pytest -q` | regression tests | 1 s |
| `python3 scripts/verify_group_theory.py` | all group-theoretic statements; writes `data/group_facts.json` | 10 s |
| `python3 scripts/verify_curve_model.py` | discriminant identity over Z; reduction certificate (singular points, delta sum 62, irreducibility) and both fibrations for p <= 31 | 1 min |
| `python3 scripts/zeta_X.py 79 4` | reduction certificate, exact counts and L-polynomials for p <= 79, mod 2 and mod 23 tests; writes `data/zeta_X.json` | about 30 min on 4 cores |
| `python3 scripts/crosscheck_rustmath.py 4 83 97 107` | recounts rows of RustMath's extended table (`data/rustmath_wpf_tables.json`) | about 40 min |
| `python3 scripts/zeta_L.py 43 2` | the same for the six covers over L at primes of degree one; reproduces the published counts 45 and 52 at 53; writes `data/zeta_L.json` | 15 min |
| `python3 scripts/verify_resolvents.py` | Gassmann patterns, irreducibility, real heptad for f1, f2 | 5 s |
| `python3 scripts/real_locus.py` | counts the ovals of X(R) from the equation | 15 s |
| `python3 scripts/golay_invariant_family.py` | J5 of the regular family; writes `data/J5_family.json` | 2 min |
| `python3 scripts/passports_23.py` | the six class vectors (C,23A,23B): sizes, real covers, ovals; writes `data/passports_C_23A_23B.json` | 2.5 min |
| `python3 scripts/passports_all.py` | all 57 rational three-point class vectors of M23: exact Nielsen class sizes and real points; writes `data/m23_three_point_nielsen.json` | 6 min |
| `python3 scripts/lifts_2_11_M23.py` | the fifteen lifts of the triple to 2^11:M23 in S_46; writes `data/lifts_2_11_M23.json` | 1 s |
| `python3 scripts/pg32_a8_model.py` | PG(3,2) spreads and the 8 letters of GL_4(2) = A_8; writes `data/pg32_a8_model.json` | 5 s |
| `python3 scripts/a8_frobenius_predictions.py` | the 14 classes of GL_4(2) = A_8 and, for every counted prime, the cycle types allowed for Frobenius in the A8 field and the order of Frobenius on Jac(X)[2] where it is decided; writes `data/a8_frobenius_predictions.json` | 5 s |
| `python3 scripts/jac2_frobenius_orbits.py` | linear algebra over F_2 for the image of Galois on Jac(X)[2] and the Frobenius orbits on the 120 tritangent planes | 5 s |
| `python3 scripts/make_tables.py` | regenerates `paper/tables/*.tex` | instant |

The degree 253 resolvents in `data/golay_resolvents.json` are produced by
`m23golay.arith.heptad_resolvent` and `duad_resolvent` (about 15 s each).

## Handoff to RustMath

`docs/RUSTMATH_HANDOFF.md` specifies three work packages for the RustMath project, with
reference values: the Frobenius table, the three-point class vectors of M23 (Phase 6 of the
research note), and a pipeline for the explicit A8 field through tritangent planes.
`docs/RUSTMATH_REQUESTS.md` (2026-11-16) is the follow-up: changes wanted in what RustMath has
built, what the exploratory work still needs, and facts found since the handoff. One of them is
not in the manuscript yet: the point counts at 17 and 73 show that the image of Galois on
Jac(X)[2] is the full stabilizer of W in Sp_8(F_2), a group 2^10:A_8.

## Sources checked

The manuscript's attributions were checked against Le Lay, *Arithmetic and geometry of an M23
Hurwitz scheme* (2026), and Zarhin's two papers on hyperelliptic Jacobians. Overlaps with Le Lay
(the two cyclic Steiner systems, the Fano plane of the fibre above 0, the real points of that
fibre) are now attributed in the paper, and his numbering of the seven covers is in the class
table. `scripts/gap/` holds the GAP cross-checks.

## Branches

`main` is the stable branch, and the one other repositories should reference. Work in progress is
on `claude/blissful-knuth-qla7ym` and reaches `main` by pull request.

## Conventions

Permutations act on the right, as in HJLPPZ (`x^(gh) = (x^g)^h`), with their
published `g1, g2, g3` and `g1 g2 g3 = 1`; 23A and 23B are the classes of `g2`
and `g3`. The two cyclic Golay codes are generated by
`z^11+z^9+z^7+z^6+z^5+z+1` (sign +) and its reciprocal (sign -); in this
normalization 23A has sign - and 23B has sign +.
