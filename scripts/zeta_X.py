#!/usr/bin/env python3
"""L-polynomials of the genus 4 curve X at primes of good reduction of the plane model.

For each prime p: N_k = #X(F_{p^k}) for k = 1..4, the Frobenius polynomial
P_p(T), a Weil-polynomial check, the factorisation of P_p mod 2 and the
candidates Q with P_p = Q Q^rev (mod 2) (Golay Lagrangian), and the mod 23
test P_p(chi(p)) = 0 for the 23-torsion point [b' - c'].

Usage: zeta_X.py [max_prime] [processes] [min_prime]   (existing results for other primes are kept)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import curve  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "zeta_X.json"


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def legendre(a, p):
    return 1 if pow(a % p, (p - 1) // 2, p) == 1 else -1


def work(p):
    note = "nodal reduction of the plane model"
    try:
        counts = [curve.count_points(p, k, fibre="V", processes=PROCS) for k in (1, 2, 3, 4)]
    except curve.BadFibre:
        # a non-nodal singularity mod p: count it as unibranch and rely on the Weil test
        note = "non-nodal singularity mod p, counted as unibranch; validated by the Weil test only"
        try:
            counts = [curve.count_points(p, k, fibre="V", strict=False, processes=PROCS) for k in (1, 2, 3, 4)]
        except Exception as exc:
            return {"p": p, "status": "skipped", "reason": repr(exc)}
    try:
        P = curve.frobenius_polynomial(p, counts)
        weil = curve.is_weil_polynomial(p, P)
    except Exception as exc:
        return {"p": p, "status": "skipped", "reason": repr(exc), "counts": counts}
    if not weil:
        return {"p": p, "status": "skipped", "reason": "not a Weil polynomial", "counts": counts}
    chi = legendre(-23, p)
    val = lambda x: sum(c * x ** (8 - i) for i, c in enumerate(P))
    split = curve.golay_splitting(P)
    return {
        "p": p, "status": "ok", "model": note, "counts": counts, "P": P, "chi_minus23": chi,
        "P(chi) mod 23": val(chi) % 23, "P(chi p) mod 23": val(chi * p) % 23,
        "P mod 2": [[curve.bits_to_str(f), m] for f, m in split["P_mod_2_factors"]],
        "Q candidates": [curve.bits_to_str(q) for q in split["Q_candidates"]],
        "golay_split": split["splits"],
    }


PROCS = 1

if __name__ == "__main__":
    # usage: zeta_X.py [max_prime] [processes] [min_prime]
    bound = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    PROCS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    primes = [p for p in range(start, bound + 1) if is_prime(p) and p != 23]
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    results = [r for r in results if r["p"] not in primes]
    for p in primes:                  # primes one at a time; each count uses PROCS processes
        r = work(p)
        results.append(r)
        print(json.dumps(r), flush=True)
        OUT.write_text(json.dumps(sorted(results, key=lambda r: r["p"]), indent=1) + "\n")
