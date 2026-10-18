"""Figure 3 of the paper: confidence as a forecast.

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

BLUE, GRAY, INK, MUTED, LIGHT = "#2a78d6", "#c9ccd1", "#0b0b0b", "#52514e", "#9a9ea6"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

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

fig, ax = plt.subplots(figsize=(6.6, 2.9))
L = 260
inside = df.m2.abs() <= L
ax.axhline(0, color=GRAY, linewidth=1)
ax.axvline(1.2816, color=GRAY, linewidth=1, linestyle=":")
ax.text(1.2816 + 0.06, -L + 12, "90% after ten years", fontsize=7.5, color=MUTED, va="bottom")
for _, r in df[inside].iterrows():
    ax.plot([r.t1, r.t1], [r.m2 - r.se2, r.m2 + r.se2], color=LIGHT, linewidth=0.8, zorder=1)
ax.scatter(df.t1[inside & hi], df.m2[inside & hi], s=18, color=BLUE, zorder=3)
ax.scatter(df.t1[inside & ~hi], df.m2[inside & ~hi], s=18, facecolors="white", edgecolors=BLUE, linewidths=0.9, zorder=3)
labels = {"SWIB": (-14, 8), "NYSLRS": (-24, -13), "WV ERS": (5, 2), "MASSPRIM": (5, 2), "M PERS": (5, -3), "AZSRS": (5, -4),
          "FL SBA": (4, -10), "IL TRS": (5, 2), "CalPERS": (-30, -10), "NYC TRS": (4, -9)}
for f, off in labels.items():
    if f in set(df.fund):
        r = df[df.fund == f].iloc[0]
        if abs(r.m2) <= L:
            ax.annotate(f, (r.t1, r.m2), xytext=off, textcoords="offset points", fontsize=7, color=INK)
out = df[~inside]
for _, r in out.iterrows():
    ax.annotate("%s: %+.0f bp (s.e. %.0f)" % (r.fund, r.m2, r.se2), xy=(r.t1, -L + 8), xytext=(r.t1 + 0.15, -L + 40),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
ax.text(0.02, 0.97, "rank correlation %.2f\n%d of %d programmes at or above 90%% after ten years;\n%d of the %d positive afterwards"
        % (rho, int(hi.sum()), len(df), int((df.m2[hi] > 0).sum()), int(hi.sum())), transform=ax.transAxes, fontsize=7.5, va="top", color=INK)
ax.set_xlim(-2.6, 7.0); ax.set_ylim(-L, L)
ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6])
ax.set_xlabel("$t$-statistic after the first ten years (confidence 90% at 1.28, 99% at 2.33)")
ax.set_ylabel("mean log excess return\nin all later years, bp a year")
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_forecast.pdf"); fig.savefig(ROOT / "paper" / "fig_forecast.png", dpi=200)
print("fig forecast: %d programmes, spearman %.3f, %d at >=90%%, %d positive after; outside frame: %s" % (
    len(df), rho, int(hi.sum()), int((df.m2[hi] > 0).sum()), list(out.fund)))
