"""Attribution table: every programme's excess return split into a beta tilt and selection, with the
confidence attaching to each part (paper Appendix C, Table C.3).

    python paper/make_attribution_table.py [results/timing_models_extended.csv]

Writes paper/table_attribution_rows.tex defining \\AttributionRows. For each programme, from the Jensen
regression d_t = alpha + (beta - 1) l_t + e_t of the log excess return on the benchmark log return: the mean
excess return and the framework's confidence Phi(t); beta with its standard error; the tilt (beta - 1)
times the benchmark's mean return, with the confidence that it is positive, Phi(t_{beta-1}) (the
benchmark's mean is positive for every programme); the selection alpha with the confidence that it is
positive, Phi(t_alpha). Excess = tilt + selection exactly (least squares with an intercept).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "timing_models_extended.csv")
P = pd.read_csv(src, index_col=0).sort_values("t_raw", ascending=False)
assert (P["mean_l"] > 0).all(), "a negative benchmark mean would flip the sign of the tilt's confidence"
chk = P["mean_d"] - (P["J_a"] + P["J_beta_term"])
assert np.abs(chk).max() < 1e-6, "excess must equal tilt + selection"


def bp(x):
    return "$%+d$" % round(1e4 * x)


def pct(p):
    return "%d" % round(100 * p)


lines = []
for f, r in P.iterrows():
    lines.append(" & ".join([
        f.replace("&", "\\&"), "%d" % r["n"],
        bp(r["mean_d"]), pct(r["cis_raw"]),
        "%.2f (%.2f)" % (1 + r["J_bm1"], r["J_se_b"]),
        bp(r["J_beta_term"]), pct(stats.norm.cdf(r["J_t_b"])),
        bp(r["J_a"]), pct(r["cis_a_J"]),
    ]) + " \\\\")
(ROOT / "paper" / "table_attribution_rows.tex").write_text(
    "\\newcommand{\\AttributionRows}{%\n" + "\n".join(lines) + "\n}\n")
print("wrote table_attribution_rows.tex with %d rows" % len(lines))
