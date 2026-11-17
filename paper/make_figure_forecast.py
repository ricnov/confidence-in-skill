"""Figure 3 of the paper: confidence as a forecast, in matplotlib's default style.

    python paper/make_figure_forecast.py [data/panel_returns_extended.csv] [--clean]

Each programme's confidence after its first ten observed years (the framework's first look) against its
mean log excess return over all later years, with a one-standard-error bar on the latter. Programmes with
fewer than three later years are omitted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cis.panel import load_panel, clean_panel, log_excess_matrix  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = args[0] if args else str(ROOT / "data" / "panel_returns_extended.csv")
panel, _ = load_panel(src)
if "--clean" in sys.argv:
    panel = clean_panel(panel)
mat = log_excess_matrix(panel)

rows = []
for f in mat.columns:
    d = mat[f].dropna().values
    if len(d) < 13:
        continue
    a, b = d[:10], d[10:]
    t1 = np.sqrt(10) * a.mean() / a.std(ddof=1)
    rows.append(dict(fund=f, t1=t1, cis1=stats.norm.cdf(t1), m2=1e4 * b.mean(), se2=1e4 * b.std(ddof=1) / np.sqrt(len(b)), n2=len(b)))
df = pd.DataFrame(rows)
hi = df.cis1 >= 0.9
rho = stats.spearmanr(df.t1, df.m2)[0]

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(6.3, 3.4))
L = 260
inside = df.m2.abs() <= L
ax.axhline(0, color="k", linewidth=0.8)
ax.axvline(1.2816, color="k", linewidth=0.8, linestyle=":", label="90% confidence after ten years")
a = df[inside & hi]; b = df[inside & ~hi]
ax.errorbar(a.t1, a.m2, yerr=a.se2, fmt="o", color="C0", markersize=4, ecolor="0.6", elinewidth=0.8,
            label="at or above 90% after ten years")
ax.errorbar(b.t1, b.m2, yerr=b.se2, fmt="o", markerfacecolor="white", markeredgecolor="C0", color="C0", markersize=4,
            ecolor="0.6", elinewidth=0.8, label="below 90% after ten years")
labels = {"SWIB": (-14, 8), "NYSLRS": (6, -16), "WV ERS": (5, 2), "MASSPRIM": (5, 2), "M PERS": (5, -3), "AZSRS": (5, -4),
          "FL SBA": (4, -10), "IL TRS": (5, 2), "CalPERS": (-30, -10), "NYC TRS": (4, -9)}
for f, off in labels.items():
    if f in set(df.fund):
        r = df[df.fund == f].iloc[0]
        if abs(r.m2) <= L:
            ax.annotate(f, (r.t1, r.m2), xytext=off, textcoords="offset points", fontsize=7)
out = df[~inside]
for _, r in out.iterrows():
    ax.annotate("%s: %+.0f bp (s.e. %.0f)" % (r.fund, r.m2, r.se2), xy=(r.t1, -L + 8), xytext=(r.t1 + 0.15, -L + 40),
                fontsize=7, arrowprops=dict(arrowstyle="->", linewidth=0.8))
ax.set_xlim(-2.6, 7.0); ax.set_ylim(-L, L)
ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6])
ax.set_xlabel("$t$-statistic after the first ten years (confidence 90% at 1.28, 99% at 2.33)")
ax.set_ylabel("mean log excess return\nin all later years, bp a year")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_forecast.pdf"); fig.savefig(ROOT / "paper" / "fig_forecast.png", dpi=200)
print("fig forecast: %d programmes, spearman %.3f, %d at >=90%%, %d positive after; outside frame: %s" % (
    len(df), rho, int(hi.sum()), int((df.m2[hi] > 0).sum()), list(out.fund)))
