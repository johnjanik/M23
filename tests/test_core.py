"""Fast regression tests (seconds).  Run: python3 -m pytest -q"""
import json
from pathlib import Path

from m23golay import arith, cohomology as coh, data, golay, invariants, m23, nielsen
from m23golay.perm import identity, mul_many

ROOT = Path(__file__).resolve().parent.parent


def test_golay_codes():
    golay.verify_basic_facts()


def test_group_order_and_relation():
    assert m23.chain().order == 10_200_960
    assert mul_many(m23.G1, m23.G2, m23.G3) == identity(23)
    assert golay.is_steiner_4_7_23(m23.steiner())


def test_signs_separate_23A_23B():
    assert m23.sign_of_23_element(m23.G2) == -1
    assert m23.sign_of_23_element(m23.G3) == +1


def test_nielsen_class():
    assert len(nielsen.nielsen_class()) == 7
    conj = nielsen.conjugation_permutation()
    assert sum(i == j for i, j in enumerate(conj)) == 3


def test_golay_lagrangian_dimensions():
    for t in nielsen.nielsen_class():
        rep = coh.lagrangian_report(t)
        assert rep["G0"]["parabolic"] == 4 and rep["Q"]["parabolic"] == 4 and rep["P0"]["parabolic"] == 8


def test_invariants():
    assert invariants.molien(6) == [1, 1, 2, 3, 5, 8, 14]
    assert invariants.subset_orbits()[:8] == [1, 1, 1, 1, 1, 2, 3, 4]
    assert all(invariants.gassmann_duads_heptads().values())


def test_steiner_labelling_f2():
    lab = arith.steiner_labelling(data.F2)
    assert abs(lab.t5[lab.sign]) <= lab.bound_t5 < abs(lab.t5[-lab.sign])
    assert arith.golay_invariants(data.F2, lab) == {"J5": 61916, "J7": -83030}


def test_stored_invariants_agree():
    stored = json.loads((ROOT / "data" / "golay_resolvents.json").read_text())
    assert stored["f1"]["J5"] == 524492 and stored["f2"]["J5"] == 61916
    assert len(stored["f2"]["R253_heptad"]) == 254


def _external_available():
    return (data.EXTERNAL / "polynomial_F.m").exists()


def test_point_counts_and_golay_splitting_at_7():
    import pytest
    if not _external_available():
        pytest.skip("HJLPPZ data not fetched")
    from m23golay import curve
    counts = [curve.count_points(7, k, fibre="V") for k in (1, 2, 3, 4)]
    assert counts == [11, 69, 383, 2433]
    assert [curve.count_points(7, k, fibre="T") for k in (1, 2, 3)] == counts[:3]
    P = curve.frobenius_polynomial(7, counts)
    assert curve.is_weil_polynomial(7, P) and curve.golay_splitting(P)["splits"]


def test_point_counts_match_published_values_at_53():
    """[HJLPPZ26, Prop. 3.14]: 45 and 52 points over F_53 at two primes of L above 53."""
    import pytest
    if not (data.EXTERNAL / "polynomial_calF.m").exists():
        pytest.skip("HJLPPZ data not fetched")
    from flint import nmod_poly
    from m23golay import curve
    roots = [int(r) for r, _ in nmod_poly([c % 53 for c in data.L_POLY], 53).roots()]
    counts = sorted(curve.count_points(53, 1, coeffs=data.calF_mod(53, s0), fibre="V")
                    for s0 in roots)
    assert counts == [45, 52, 57]


def test_three_point_table_anchor_rows():
    rows = {tuple(r["classes"]): r for r in
            json.loads((ROOT / "data" / "m23_three_point_nielsen.json").read_text())["passports"]}
    assert len(rows) == 57
    assert (rows[("2A", "23A", "23B")]["size"], rows[("2A", "23A", "23B")]["real"]) == (7, 3)
    assert (rows[("2A", "14A", "14B")]["size"], rows[("2A", "14A", "14B")]["real"]) == (17, 3)
    assert rows[("2A", "7A", "7B")]["size"] == 0
    assert all(r["real"] == 0 for k, r in rows.items() if "8A" in k)


def test_real_covers_from_involutions():
    """Real classes of (2A,23A,23B) are the involutions c with c g2 c g2^-1 in 2A, modulo <g2>."""
    from m23golay.perm import inv, mul_many
    good = [c for c in m23.involutions()
            if m23.label(mul_many(c, m23.G2, c, inv(m23.G2))) == "2A"]
    assert len(good) == 3 * 23


def test_lifts_and_pg32_data():
    lifts = json.loads((ROOT / "data" / "lifts_2_11_M23.json").read_text())
    assert len(lifts["lifts"]) == 15 and lifts["group_order"] == 2**11 * 10_200_960
    pg = json.loads((ROOT / "data" / "pg32_a8_model.json").read_text())
    for model in pg.values():
        assert len(model["lines"]) == 35 and len(model["spreads"]) == 56
        assert len(model["letter_families"]) == 8 and model["group_order_on_letters"] == 20160


def _germ(terms, p=31, k=1):
    from flint import fq_default_ctx, fq_default_poly_ctx
    ctx = fq_default_ctx(p, k, var="a")
    return {key: ctx(c) for key, c in terms.items()}, fq_default_poly_ctx(ctx)


def test_resolution_of_plane_curve_germs():
    """(places, delta, label) for standard singularities over F_31 (3 is a nonsquare mod 31)."""
    from m23golay.curve import resolve_point
    cases = [
        ({(0, 2): 1, (2, 0): -1}, (2, 1, "A_1")),                # y^2 = x^2
        ({(0, 2): 1, (2, 0): -3}, (0, 1, "A_1")),                # y^2 = 3 x^2
        ({(0, 2): 1, (3, 0): -1}, (1, 1, "A_2")),                # y^2 = x^3
        ({(0, 2): 1, (4, 0): -1}, (2, 2, "A_3")),                # y^2 = x^4
        ({(0, 2): 1, (4, 0): -3}, (0, 2, "A_3")),                # y^2 = 3 x^4
        ({(0, 2): 1, (5, 0): -1}, (1, 2, "A_4")),                # y^2 = x^5
        ({(2, 0): 1, (0, 4): -1}, (2, 2, "A_3")),                # x^2 = y^4 (tangent x = 0)
        ({(2, 1): 1, (1, 2): -1, (0, 4): 1}, (3, 3, "ordinary 3-fold")),      # xy(x - y) + y^4
        ({(3, 0): 1, (1, 2): -3, (0, 4): 1}, (1, 3, "ordinary 3-fold")),      # x(x^2 - 3y^2) + y^4
    ]
    for terms, expected in cases:
        f, R = _germ(terms)
        assert resolve_point(f, R) == expected, (terms, resolve_point(f, R))


def test_exact_counts_at_the_non_nodal_primes():
    import pytest
    if not _external_available():
        pytest.skip("HJLPPZ data not fetched")
    from m23golay import curve
    # p = 5: 60 nodes and one tacnode; both fibrations
    assert [curve.count_points(5, k, fibre="V") for k in (1, 2, 3, 4)] == [7, 37, 169, 561]
    assert [curve.count_points(5, k, fibre="T") for k in (1, 2, 3)] == [7, 37, 169]
    cert = curve.reduction_certificate(5)
    assert cert["certified"] and cert["types"] == {"A_1": 60, "A_3": 1}
    # p = 13: 61 nodes and one cusp
    assert curve.reduction_certificate(13)["types"] == {"A_1": 61, "A_2": 1}
    assert [curve.count_points(13, k, fibre="V") for k in (1, 2, 3)] == [11, 161, 2135]
    # the cover over L at (31, s - 12): a tacnode and an ordinary triple point
    A = data.calF_mod(31, 12)
    assert [curve.count_points(31, k, coeffs=A, fibre="V") for k in (1, 2)] == [43, 1017]
    assert curve.reduction_certificate(31, A)["types"] == {"A_1": 57, "A_3": 1, "ordinary 3-fold": 1}
    # models that must be refused
    assert not curve.reduction_certificate(3)["certified"]
    assert not curve.reduction_certificate(11, data.calF_mod(11, 2))["certified"]


def test_irreducibility_is_not_claimed_for_a_product_of_conjugates():
    """N(TV - alpha), alpha in F_169 \\ F_13, is irreducible over F_13 with irreducible
    fibres, and splits over F_169.  The certificate must not call it absolutely irreducible."""
    from m23golay.curve import resultant_T, singular_locus
    p = 13
    # alpha a root of x^2 - 2 (2 is a nonsquare mod 13): trace 0, norm -2
    coeffs = [[(-2) % p, 0, 0], [0, 0, 0], [0, 0, 1]]          # T^2 V^2 - 2
    loc = singular_locus(p, coeffs, [resultant_T(p, coeffs)])
    assert loc["irreducible"] is False
