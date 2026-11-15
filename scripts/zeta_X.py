#!/usr/bin/env python3
"""L-polynomials of the genus 4 curve X from its plane model F(T,V) = 0.

For each prime p:
  certificate   the model mod p is absolutely irreducible and the delta invariants of its
                singular points add up to 62, so that its normalization is the good
                reduction of X (curve.reduction_certificate);
  counts        N_k = #X(F_{p^k}), k = 1..4, exactly: singular points are resolved by
                blowing up (curve.count_points);
  P_p           the Frobenius polynomial, with the Weil check;
  mod 23        P_p(chi(p)) = P_p(chi(p) p) = 0 mod 23 (the class [b' - c'] of order 23);
  mod 2         the factorization of P_p mod 2 and the quartics Q with P_p = Q Q^rev
                (the Golay Lagrangian).

Usage:
  zeta_X.py MAX [PROCS [MIN]]     recompute the primes MIN..MAX (existing rows are kept)
  zeta_X.py --primes 5,13,31      recompute the listed primes
  zeta_X.py --certify             add the certificate to every stored row without recounting
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m23golay import curve  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "zeta_X.json"
PROCS = 4


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def legendre(a, p):
    return 1 if pow(a % p, (p - 1) // 2, p) == 1 else -1


def derived(p, counts):
    P = curve.frobenius_polynomial(p, counts)
    if not curve.is_weil_polynomial(p, P):
        raise ValueError("not a Weil polynomial")
    chi = legendre(-23, p)
    val = lambda x: sum(c * x ** (8 - i) for i, c in enumerate(P))
    split = curve.golay_splitting(P)
    return {
        "P": P, "chi_minus23": chi,
        "P(chi) mod 23": val(chi) % 23, "P(chi p) mod 23": val(chi * p) % 23,
        "P mod 2": [[curve.bits_to_str(f), m] for f, m in split["P_mod_2_factors"]],
        "Q candidates": [curve.bits_to_str(q) for q in split["Q_candidates"]],
        "golay_split": split["splits"],
    }


def certificate(p):
    cert = curve.reduction_certificate(p)
    if not cert["certified"]:
        return cert
    return {"certified": True, "singular_points": cert["types"], "delta": cert["delta"]}


def work(p):
    cert = certificate(p)
    if not cert["certified"]:
        return {"p": p, "status": "refused", "reason": cert.get("reason", "geometric genus of the model mod p is not 4"),
                "certificate": cert}
    counts = [curve.count_points(p, k, fibre="V", processes=PROCS) for k in (1, 2, 3, 4)]
    row = {"p": p, "status": "ok", "certificate": cert, "counts": counts}
    row.update(derived(p, counts))
    return row


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else []


def save(rows):
    OUT.write_text(json.dumps(sorted(rows, key=lambda r: r["p"]), indent=1) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    rows = load()
    if args and args[0] == "--certify":
        for r in rows:
            if r["status"] == "ok":
                r["certificate"] = certificate(r["p"])
                r.pop("model", None)
                print(r["p"], r["certificate"], flush=True)
        save(rows)
        sys.exit(0)
    if args and args[0] == "--primes":
        primes = [int(a) for a in args[1].split(",")]
        if len(args) > 2:
            PROCS = int(args[2])
    else:
        bound = int(args[0]) if args else 50
        PROCS = int(args[1]) if len(args) > 1 else 4
        start = int(args[2]) if len(args) > 2 else 5
        primes = [p for p in range(start, bound + 1) if is_prime(p)]
    for p in primes:
        r = work(p)
        rows = [x for x in rows if x["p"] != p] + [r]
        print(json.dumps(r), flush=True)
        save(rows)
