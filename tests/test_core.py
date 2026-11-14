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
    counts = sorted(curve.count_points(53, 1, coeffs=data.calF_mod(53, s0), fibre="V", strict=False)
                    for s0 in roots)
    assert counts == [45, 52, 57]
