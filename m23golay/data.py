"""Published polynomials of [HJLPPZ26].

The univariate examples are printed in the paper and reproduced here.  The
bivariate models F, F_small (over Q) and calF (over the sextic field L) are
read from a local clone of the authors' repository
https://github.com/shaowuz/m23isgalois (see scripts/fetch_external.sh); they
are not redistributed in this repository.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = Path(os.environ.get("M23ISGALOIS", ROOT / "external" / "m23isgalois"))

# [HJLPPZ26, Example 1.2]: specialisation T = 2 after polredbest.  Ascending coefficients.
F1 = [-3150159884154, 19169943578802, -46185312415788, 53599151839311, -28142323927002,
      1949980486716, 3961650395556, -1023887308293, -194398310718, 35123826654, 3700476348,
      10550424369, 290615166, -1333395744, -42732528, 67672923, 1545462, -1808490, 18400,
      26151, -1150, -184, 0, 1]

# [HJLPPZ26, Example 3.17]: specialisation T = -3703/947 after polredbest.
F2 = [1243077066, -11217790920, 38754121124, -67357061907, 63691532334, -26037806834,
      -5728074376, 9451874955, -2463757240, 83587888, -436105484, 320807726, -71999476,
      17344116, -9392096, 2223732, -361974, 127420, -21620, 1679, -598, 46, 0, 1]

# [HJLPPZ26, Example 3.22]: the M22 polynomial (degree 22).
F3 = [212425572022, 8150987444096, 19223398109406, 12770814291390, -5309597045808,
      -12865403779026, -8595497273752, -2898168029276, -396850048164, 48162643482,
      44873894586, 13957937472, 222611762, -1334363150, -104404974, 57499538, 1612816,
      -878184, -3160, 4948, -171, -10, 1]

FIELD_DISCRIMINANTS = {"f1": 2**36 * 3**18 * 23**30, "f2": 2**44 * 7**8 * 23**24}

# The sextic field of moduli of the other six covers [HJLPPZ26, Section 3.7].
L_POLY = [4, 5, 1, -3, -3, -1, 1]

_TERM = re.compile(r"([+-]?\d+)\*T\^(\d+)\*[VW]\^(\d+)")


def _read_bivariate(name: str) -> list[list[int]]:
    path = EXTERNAL / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/fetch_external.sh or set M23ISGALOIS")
    coeffs = [[0] * 24 for _ in range(5)]
    for c, i, j in _TERM.findall(path.read_text()):
        coeffs[int(i)][int(j)] += int(c)
    return coeffs


@lru_cache(maxsize=None)
def F() -> list[list[int]]:
    """F[i][j] = coefficient of T^i V^j in the bidegree (4,23) model of X."""
    f = _read_bivariate("polynomial_F.m")
    assert max(len(str(abs(c))) for row in f for c in row) == 63
    return f


@lru_cache(maxsize=None)
def F_small() -> list[list[int]]:
    """F_small[i][j] = coefficient of T^i W^j (V = (37W - 40)/(49W + 32))."""
    return _read_bivariate("polynomial_F_small.m")


def specialise(coeffs, q_num: int, q_den: int = 1) -> list[int]:
    """Integer coefficients (ascending in V) of q_den^4 F(q_num/q_den, V)."""
    return [sum(coeffs[i][j] * q_num**i * q_den**(4 - i) for i in range(5)) for j in range(24)]


_CALF_TERM = re.compile(r"\(((?:\(-?\d+\)\*s\^\d\+?)+)\)\*T\^(\d+)\*V\^(\d+)")
_CALF_COEF = re.compile(r"\((-?\d+)\)\*s\^(\d)")


@lru_cache(maxsize=None)
def calF() -> dict:
    """calF[(i, j)] = coefficients in the power basis 1, s, ..., s^5 of L of the
    coefficient of T^i V^j in the model of the covers defined over L."""
    path = EXTERNAL / "polynomial_calF.m"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/fetch_external.sh")
    txt = path.read_text()
    body = txt[txt.index("calF :="):txt.index("assert")]
    out = {}
    for inner, i, j in _CALF_TERM.findall(body):
        cs = [0] * 6
        for c, e in _CALF_COEF.findall(inner):
            cs[int(e)] = int(c)
        out[(int(i), int(j))] = cs
    assert len(out) == 120
    return out


def calF_mod(p: int, s0: int) -> list[list[int]]:
    """Reduction of calF at the degree one prime (p, s - s0) of L."""
    cf = calF()
    return [[sum(c * pow(s0, e, p) for e, c in enumerate(cf.get((i, j), [0] * 6))) % p
             for j in range(24)] for i in range(5)]
