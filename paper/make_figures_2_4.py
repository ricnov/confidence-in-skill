"""Figures 2-4 of the paper.

    python paper/make_figures_2_4.py [data/panel_returns.csv]

Figure 2  Two benchmark columns: (a) the LASERS sheet's benchmark against the S&P 500 total return,
          fiscal years; (b) the New York fund's cumulative log excess return against each of its two
          benchmark columns, 2013-2025.
Figure 3  (no longer in the paper; kept for the record) Persistence: second-half against first-half mean
          log excess return, 42 programmes. The paper's Figure 3 is now paper/make_figure_forecast.py.
Figure 4  CalSTRS's expanding-window confidence path with the article's numerator, the model-consistent
          numerator, and the constant-gap reference curve.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cis.panel import load_panel, clean_panel, log_excess_matrix, SP500_FY_JUNE
from cis.dispersion import half_masks

ROOT = Path(__file__).resolve().parents[1]
src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "panel_returns.csv")
panel, _ = load_panel(src)

BLUE, GRAY, INK, MUTED, LIGHT = "#2a78d6", "#c9ccd1", "#0b0b0b", "#52514e", "#9a9ea6"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
ax = axes[0]
las = panel[panel.fund == "LASERS"].set_index("year")
sp = pd.Series({**{2007: .2059, 2008: -.1312, 2009: -.2621}, **SP500_FY_JUNE})   # June fiscal years
yrs = las.index
x = np.array([sp.get(y, np.nan) for y in yrs]); y = las["benchmark"].values
early = np.isin(yrs, [2007, 2008, 2009]); late = ~early
lim = (-0.30, 0.45)
ax.plot(lim, lim, color=GRAY, linewidth=1, zorder=1)
ax.scatter(x[late], y[late], s=14, color=BLUE, zorder=3)
ax.scatter(x[early], y[early], s=22, facecolors="white", edgecolors=INK, linewidths=0.9, zorder=3)
for yr in (2007, 2008, 2009):
    k = int(np.where(yrs == yr)[0][0])
    ax.annotate(str(yr), (x[k], y[k]), xytext=(5, -3), textcoords="offset points", fontsize=7, color=INK)
ax.text(0.03, 0.95, "2010–2025: sixteen years\non the diagonal", transform=ax.transAxes, fontsize=8, va="top", color=INK)
ax.set_xlabel("S&P 500 total return, fiscal year"); ax.set_ylabel("LASERS sheet, ‘benchmark’ column")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xticks([-0.2, 0, 0.2, 0.4]); ax.set_yticks([-0.2, 0, 0.2, 0.4])
ax.set_xticklabels(["−20%", "0", "20%", "40%"]); ax.set_yticklabels(["−20%", "0", "20%", "40%"])
ax = axes[1]
ny = panel[panel.fund == "NYSLRS"].set_index("year"); crf = panel[panel.fund == "NYSCRF"].set_index("year")
common = ny.index.intersection(crf.index)
d1 = np.log1p(ny.loc[common, "actual"]) - np.log1p(ny.loc[common, "benchmark"])
d2 = np.log1p(crf.loc[common, "actual"]) - np.log1p(crf.loc[common, "benchmark"])
c1 = np.concatenate([[0], np.cumsum(d1.values)]); c2 = np.concatenate([[0], np.cumsum(d2.values)])
t_axis = np.concatenate([[common[0] - 1], common])
ax.axhline(0, color=GRAY, linewidth=1)
ax.plot(t_axis, 100 * c1, color=BLUE, linewidth=2)
ax.plot(t_axis, 100 * c2, color=INK, linewidth=2, linestyle="--")
ax.text(t_axis[-1] + 0.3, 100 * c1[-1], "against the\nNYSLRS column", fontsize=8, va="center", color=BLUE)
ax.text(t_axis[-1] + 0.3, 100 * c2[-1] + 0.6, "against the\nNYSCRF column", fontsize=8, va="center", color=INK)
ax.set_xlabel("fiscal year"); ax.set_ylabel("cumulative log excess return, %")
ax.set_xlim(t_axis[0], t_axis[-1] + 5.5); ax.set_xticks([2013, 2016, 2019, 2022, 2025])
fig.tight_layout(w_pad=2.0)
fig.savefig(ROOT / "paper" / "fig_benchmarks.pdf"); fig.savefig(ROOT / "paper" / "fig_benchmarks.png", dpi=200)
print("fig 2: LASERS on-diagonal years %d; NY cumulative %.1f%% vs %.1f%%" % (int(late.sum()), 100 * c1[-1], 100 * c2[-1]))

# ---------------------------------------------------------------- Figure 3
cl = clean_panel(panel)
mat = log_excess_matrix(cl); V = mat.values
A_, B_ = half_masks(mat)
Za = np.where(np.isnan(V), 0, V) * A_; Zb = np.where(np.isnan(V), 0, V) * B_
ma = 100 * Za.sum(0) / A_.sum(0); mb = 100 * Zb.sum(0) / B_.sum(0)
names = np.array(mat.columns)
fig, ax = plt.subplots(figsize=(4.0, 3.3))
L = 2.6
ax.axhline(0, color=GRAY, linewidth=1); ax.axvline(0, color=GRAY, linewidth=1)
ax.plot([-L, L], [-L, L], color=GRAY, linewidth=1, linestyle=":")
inside = (np.abs(ma) <= L) & (np.abs(mb) <= L)
ax.scatter(ma[inside], mb[inside], s=16, color=BLUE, zorder=3)
offsets = {"NYSLRS": (5, -9), "WV ERS": (5, 2), "M PERS": (5, 0), "CalPERS": (-4, -11), "SO DAK RS": (5, -3), "LAFPP": (5, -3)}
for f, off in offsets.items():
    j = int(np.where(names == f)[0][0])
    if inside[j]:
        ax.annotate(f, (ma[j], mb[j]), xytext=off, textcoords="offset points", fontsize=7, color=INK)
j = int(np.where(names == "DAL F&P")[0][0])
ax.annotate("DAL F&P: (%.1f, %.1f)" % (ma[j], mb[j]), xy=(-L + 0.05, -L + 0.05), xytext=(-L + 0.35, -L + 0.45),
            fontsize=7, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
rho = stats.spearmanr(ma, mb)[0]
ax.text(0.03, 0.97, "rank correlation %.2f" % rho, transform=ax.transAxes, fontsize=8, va="top", color=INK)
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
ax.set_xlabel("first half of sample: mean log excess return, % p.a.")
ax.set_ylabel("second half of sample, % p.a.")
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_persistence.pdf"); fig.savefig(ROOT / "paper" / "fig_persistence.png", dpi=200)
print("fig 3: spearman %.3f; points outside the box: %s" % (rho, list(names[~inside])))

# ---------------------------------------------------------------- Figure 4
cs = panel[panel.fund == "CalSTRS"].sort_values("year")
rP, rL, yrs = cs["actual"].values, cs["benchmark"].values, cs["year"].values
d = np.log1p(rP) - np.log1p(rL)
looks = range(10, len(d) + 1)
path_star, path_paper = [], []
for k in looks:
    dd = d[:k]
    path_star.append(stats.norm.cdf(np.sqrt(k) * dd.mean() / dd.std(ddof=1)))
    # the article's numerator: annualised geometric returns minus sigma^2/2, tracking error from eq. (7) on simple returns
    gP = np.prod(1 + rP[:k]) ** (1 / k) - 1; gL = np.prod(1 + rL[:k]) ** (1 / k) - 1
    sP, sL = rP[:k].std(ddof=1), rL[:k].std(ddof=1); rho_ = np.corrcoef(rP[:k], rL[:k])[0, 1]
    te = np.sqrt(sP ** 2 + sL ** 2 - 2 * rho_ * sP * sL)
    gap = (gP - sP ** 2 / 2) - (gL - sL ** 2 / 2)
    path_paper.append(stats.norm.cdf(np.sqrt(k) * gap / te))
gap_T, sd_T = d.mean(), d.std(ddof=1)
ref = [stats.norm.cdf(np.sqrt(k) * gap_T / sd_T) for k in looks]
xs = yrs[9:]
fig, ax = plt.subplots(figsize=(6.6, 2.4))
ax.axhline(0.5, color=GRAY, linewidth=1)
ax.plot(xs, path_paper, color=LIGHT, linewidth=1.6, label="article's numerator (geometric returns, drag subtracted)")
ax.plot(xs, path_star, color=BLUE, linewidth=2, label="model-consistent $t$-statistic")
ax.plot(xs, ref, color=INK, linewidth=1.2, linestyle="--", label="constant-gap reference: terminal gap and tracking error held fixed")
ax.set_ylim(0.3, 1.0); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["40%", "50", "60", "70", "80", "90", "100"])
ax.set_xlabel("fiscal year at which the expanding window ends (first look after ten years)")
ax.set_ylabel("confidence in skill")
ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2025])
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_path.pdf"); fig.savefig(ROOT / "paper" / "fig_path.png", dpi=200)
print("fig 4: CalSTRS terminal: paper %.3f star %.3f ref %.3f; 2023: paper %.3f star %.3f" % (
    path_paper[-1], path_star[-1], ref[-1], path_paper[list(xs).index(2023)], path_star[list(xs).index(2023)]))
