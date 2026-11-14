#!/usr/bin/env python3
"""Generate the LaTeX tables of the paper from the JSON files in data/."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
facts = json.loads((DATA / "group_facts.json").read_text())
zeta = json.loads((DATA / "zeta_X.json").read_text())
res = json.loads((DATA / "golay_resolvents.json").read_text())


def tex_type(s: str) -> str:
    return "$" + re.sub(r"\^(\d+)", r"^{\1}", s).replace(" ", "\\,") + "$"


def num(n: int) -> str:
    s = f"{abs(n):,}".replace(",", "\\,")
    return ("-" if n < 0 else "") + s


# --- classes -----------------------------------------------------------------
rows = []
for r in facts["classes"]:
    name = r["class"].replace("AB", "A,B")
    rows.append(f"${name}$ & {tex_type(r['cycle_type'])} & ${num(r['elements'])}$ & ${r['fixed_heptads']}$ & "
                f"{tex_type(r['cycle_type_on_heptads'])} & ${r['dim_G0_fixed']}$ & ${r['dim_P0_fixed']}$\\\\")
(OUT / "classes.tex").write_text(r"""\begin{table}[ht]
\centering\footnotesize
\begin{tabular}{@{}llrrlrr@{}}
\toprule
class & on $23$ points & elements & fixed & on duads $=$ on heptads & $\dim\cG_0^{\,g}$ & $\dim P_0^{\,g}$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Cycle types in $\M$. ``Elements'' counts both classes for the split types; ``fixed'' is the number of fixed duads, equal to the number of fixed heptads. The cycle types on duads and on heptads coincide (Proposition~\ref{prop:gassmann}), and $\dim P_0^{\,g}=2\dim\cG_0^{\,g}$ (Corollary~\ref{cor:fixed}).}
\label{tab:classes}
\end{table}
""")

# --- Nielsen class -------------------------------------------------------------
rows = []
for r in facts["nielsen"]:
    real = "real" if r["real"] else f"conj.\\ to {r['conjugate']}"
    field = "$\\QQ$" if r["index"] == facts["rational_class"] else "$L$"
    extra = (f"${r['real_components']}$ & ${r['dim_J2_real']}$ & ${r['dim_W_real']}$" if r["real"]
             else "-- & -- & --")
    rows.append(f"${r['index']}$ & ${r['hjlppz_file_row']}$ & ${r['janik_orbit']}$ & {real} & {field} & "
                f"${r['dim_W']}$ & {extra}\\\\")
(OUT / "nielsen.tex").write_text(r"""\begin{table}[ht]
\centering\small
\begin{tabular}{@{}cccllcccc@{}}
\toprule
class & \cite{HJLPPZ26} file & \cite{Janik26} orbit & conjugation & moduli & $\dim W_\cG$ & $r$ & $\dim\Jac[2]^{c}$ & $\dim W_\cG^{\,c}$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{The seven classes of $\Ni(2A,23A,23B)$. Columns 2 and 3 give the position in the data file of \cite{HJLPPZ26} and the orbit label of \cite[Table~1]{Janik26}. For real covers, $r$ is the number of components of $X(\RR)$ and the last two columns are the dimensions of the fixed spaces of complex conjugation (Theorem~\ref{thm:real}).}
\label{tab:nielsen}
\end{table}
""")

# --- Frobenius -----------------------------------------------------------------
TYPES = {  # Q up to reversal -> (order of semisimple part, possible cycle types in A8)
    frozenset({"x^4+x+1", "x^4+x^3+1"}): ("15", "$3\\,5$"),
    frozenset({"x^4+x^3+x^2+1", "x^4+x^2+x+1"}): ("7", "$1\\,7$"),
    frozenset({"x^4+x^3+x^2+x+1"}): ("5", "$1^{3}\\,5$"),
    frozenset({"x^4+x^3+x+1"}): ("3", "$1^{2}\\,3^{2}$ or $2\\,6$"),
    frozenset({"x^4+x^2+1"}): ("3", "$1^{5}\\,3$ or $1\\,2^{2}\\,3$"),
    frozenset({"x^4+1"}): ("1", "$1^8$, $2^4$, $1^42^2$, $4^2$ or $1^2\\,2\\,4$"),
}


def poly_tex(s: str) -> str:
    return "$" + re.sub(r"\^(\d+)", r"^{\1}", s) + "$"


def factored(fs) -> str:
    out = []
    for f, m in fs:
        f = re.sub(r"\^(\d+)", r"^{\1}", f)
        out.append(f"({f})" + (f"^{{{m}}}" if m > 1 else ""))
    return "$" + "".join(out) + "$"


rows, counts, starred = [], {}, []
for r in zeta:
    if r["status"] != "ok":
        continue
    assert r["golay_split"] and r["P(chi) mod 23"] == 0 and r["P(chi p) mod 23"] == 0
    key = frozenset(r["Q candidates"])
    order, cyc = TYPES[key]
    counts[order] = counts.get(order, 0) + 1
    star = "" if r["model"].startswith("nodal") else "$^*$"
    if star:
        starred.append(r["p"])
    a = r["P"][1:5]
    rows.append(f"${r['p']}${star} & $({', '.join(str(x) for x in a)})$ & {factored(r['P mod 2'])} & "
                f"{poly_tex(sorted(r['Q candidates'])[0])} & ${order}$ & {cyc}\\\\")
(OUT / "frobenius.tex").write_text(r"""\begin{table}[ht]
\centering\footnotesize
\begin{tabular}{@{}rlllcl@{}}
\toprule
$p$ & $(a_1,a_2,a_3,a_4)$ & $P_p \bmod 2$ & $Q_p$ & order & type in $A_8$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Frobenius on $\Jac(X)$ and on the Golay Lagrangian $W$. Here $P_p(T)=T^8+a_1T^7+a_2T^6+a_3T^5+a_4T^4+pa_3T^3+p^2a_2T^2+p^3a_1T+p^4$, $Q_p$ is given up to reversal, ``order'' is the order of the semisimple part of $\rho_W(\Frob_p)$, and the last column lists the possible cycle types on $8$ letters. An asterisk marks primes at which the plane model has a non-nodal singularity (Section~\ref{sec:computations}).}
\label{tab:frobenius}
\end{table}
""")

# --- invariants ----------------------------------------------------------------
rows = []
for name, label in (("f1", "f_1"), ("f2", "f_2")):
    d = res[name]
    sign = "+" if d["sign_at_prime"] > 0 else "-"
    digits = max(len(str(abs(int(c)))) for c in d["R253_heptad"])
    ddig = max(len(str(abs(int(c)))) for c in d["R253_duad"])
    rows.append(f"${label}$ & ${d['labelling_prime']}$ & ${sign}$ & ${num(d['T5'])}$ & ${num(d['J5'])}$ & "
                f"${num(d['J7'])}$ & ${digits}$ & ${ddig}$ & ${d['verified_at_second_prime'][0]}$\\\\")
(OUT / "invariants.tex").write_text(r"""\begin{table}[ht]
\centering\small
\begin{tabular}{@{}lccrrrccc@{}}
\toprule
 & $p$ & $\varepsilon(\Frob_p)$ & $T_5$ & $J_5$ & $J_7$ & digits of $R_f$ & digits of $D_f$ & check\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Steiner labelling and Golay invariants of the polynomials of \cite[Examples 1.2, 3.17]{HJLPPZ26}. ``Digits'' is the size of the largest coefficient of the heptad resolvent $R_f$ and of the duad resolvent $D_f$; ``check'' is the second prime at which everything was recomputed.}
\label{tab:invariants}
\end{table}
""")

# --- Molien --------------------------------------------------------------------
m = facts["molien"]
n = 13
head = " & ".join(f"${d}$" for d in range(n))
(OUT / "molien.tex").write_text(r"""\begin{table}[ht]
\centering\small
\begin{tabular}{@{}l""" + "r" * n + r"""@{}}
\toprule
degree & """ + head + r"""\\
\midrule
$\M$-invariants & """ + " & ".join(f"${x}$" for x in m["M23"][:n]) + r"""\\
symmetric & """ + " & ".join(f"${x}$" for x in m["S23"][:n]) + r"""\\
difference & """ + " & ".join(f"${x}$" for x in m["excess"][:n]) + r"""\\
\bottomrule
\end{tabular}
\caption{Dimensions of the spaces of homogeneous invariants of $\M$ and of $S_{23}$ in $\QQ[x_1,\dots,x_{23}]$. The numbers of orbits of $\M$ on $k$-subsets, $k=0,\dots,11$, are $""" + ",".join(str(x) for x in facts["subset_orbits"][:12]) + r"""$.}
\label{tab:molien}
\end{table}
""")
print("tables written; Frobenius classes by order of semisimple part:", counts, "; starred primes:", starred)
