"""Table rows for the selection / beta / timing section (paper Section 4.9).

    python paper/make_timing_table.py [results/timing_models_extended.csv]

Writes paper/table_timing_rows.tex defining \\TimingRows: the programmes at or above 90% on the framework's
statistic plus the named comparison cases, with the Jensen beta and alpha, the beta term, and the
Treynor-Mazuy and Henriksson-Merton gammas with their t-statistics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "timing_models_extended.csv")
P = pd.read_csv(src, index_col=0)
EXTRA = ["CalSTRS", "TX TRS", "SBCERA", "CalPERS", "DAL F&P"]
rows = P[(P["cis_raw"] > 0.9) | P.index.isin(EXTRA)].sort_values("t_raw", ascending=False)


def num(x, d=0, sign=False):
    if not np.isfinite(x):
        return "--"
    s = ("%+." if sign else "%.") + "%df" % d
    return "$" + (s % x).replace("-", "-") + "$"


lines = []
for f, r in rows.iterrows():
    name = f.replace("&", "\\&")
    lines.append(" & ".join([
        name, "%d" % r["n"], "%d" % r["down_years"],
        num(1e4 * r["mean_d"], 0, True), num(r["t_raw"], 2),
        "%.2f (%.2f)" % (1 + r["J_bm1"], r["J_se_b"]),
        "%s (%s)" % (num(1e4 * r["J_a"], 0, True), num(r["J_t_a"], 1)),
        num(1e4 * r["J_beta_term"], 0, True),
        "%s (%s)" % (num(r["TM_g"], 2, True), num(r["TM_t_g"], 1)),
        "%s (%s)" % (num(r["HM_g"], 2, True), num(r["HM_t_g"], 1)),
    ]) + " \\\\")
body = "\n".join(lines)
(ROOT / "paper" / "table_timing_rows.tex").write_text("\\newcommand{\\TimingRows}{%\n" + body + "\n}\n")
print("wrote table_timing_rows.tex with %d rows" % len(rows))
