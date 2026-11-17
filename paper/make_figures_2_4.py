"""Figures 2-4 of the paper, in matplotlib's default style.

    python paper/make_figures_2_4.py [data/panel_returns.csv]

Figure 2  Two benchmark columns: (a) the LASERS sheet's benchmark against the S&P 500 total return,
          fiscal years; (b) the New York fund's cumulative log excess return against the NYSLRS sheet's
          benchmark column, against its policy benchmark rebuilt from the Fund's reports (cis.panel.
          NY_POLICY_BENCHMARK) and against the NYSCRF sheet's column, which is the Fund's own return.
Figure 3  (no longer in the paper; kept for the record) Persistence: second-half against first-half mean
          log excess return, 42 programmes. The paper's Figure 3 is now paper/make_figure_forecast.py.
Figure 4  CalSTRS's expanding-window confidence path with the article's numerator, the model-consistent
          numerator, the constant-gap reference curve and the anytime-valid confidence.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from cis.panel import load_panel, clean_panel, log_excess_matrix, SP500_FY_JUNE, NY_POLICY_BENCHMARK
from cis.dispersion import half_masks
from cis.anytime import e_process, confidence_path

ROOT = Path(__file__).resolve().parents[1]
src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "panel_returns.csv")
panel, _ = load_panel(src)
plt.rcdefaults()

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
ax = axes[0]
las = panel[panel.fund == "LASERS"].set_index("year")
sp = pd.Series({**{2007: .2059, 2008: -.1312, 2009: -.2621}, **SP500_FY_JUNE})   # June fiscal years
yrs = las.index
x = np.array([sp.get(y, np.nan) for y in yrs]); y = las["benchmark"].values
early = np.isin(yrs, [2007, 2008, 2009]); late = ~early
lim = (-0.30, 0.45)
ax.plot(lim, lim, "k-", linewidth=0.8)
ax.plot(x[late], y[late], "o", color="C0", markersize=4, label="2010–2025")
ax.plot(x[early], y[early], "o", markerfacecolor="white", markeredgecolor="C0", markersize=5, label="2007–2009")
for yr in (2007, 2008, 2009):
    k = int(np.where(yrs == yr)[0][0])
    ax.annotate(str(yr), (x[k], y[k]), xytext=(5, -3), textcoords="offset points", fontsize=8)
ax.set_xlabel("S&P 500 total return, fiscal year"); ax.set_ylabel("LASERS sheet, benchmark column")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.legend(loc="upper left", fontsize=8)
ax = axes[1]
ny = panel[panel.fund == "NYSLRS"].set_index("year"); crf = panel[panel.fund == "NYSCRF"].set_index("year")
pol = pd.Series(NY_POLICY_BENCHMARK)
yrs_ny = ny.index
d_sheet = np.log1p(ny["actual"]) - np.log1p(ny["benchmark"])
d_pol = np.log1p(ny["actual"]) - np.log1p(pol.reindex(yrs_ny))
common = yrs_ny.intersection(crf.index)
d_crf = np.log1p(crf.loc[common, "actual"]) - np.log1p(crf.loc[common, "benchmark"])
cum = lambda d: np.concatenate([[0], np.cumsum(d.values)])
ax.axhline(0, color="k", linewidth=0.8)
ax.plot(np.concatenate([[yrs_ny[0] - 1], yrs_ny]), 100 * cum(d_sheet), "C0-", label="NYSLRS sheet's column")
ax.plot(np.concatenate([[yrs_ny[0] - 1], yrs_ny]), 100 * cum(d_pol), "C3-", label="policy benchmark rebuilt from the reports")
ax.plot(np.concatenate([[common[0] - 1], common]), 100 * cum(d_crf), "C1--", label="NYSCRF column (the Fund's own return)")
ax.set_xlabel("fiscal year"); ax.set_ylabel("cumulative log excess return, %")
ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
ax.set_ylim(-6, 28)
ax.legend(loc="upper left", fontsize=7)
c1, c2 = cum(d_sheet), cum(d_pol)
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_benchmarks.pdf"); fig.savefig(ROOT / "paper" / "fig_benchmarks.png", dpi=200)
print("fig 2: LASERS on-diagonal years %d; NY cumulative against the sheet's column %.1f%%, against the policy benchmark %.1f%%" % (int(late.sum()), 100 * c1[-1], 100 * c2[-1]))

# ---------------------------------------------------------------- Figure 3
cl = clean_panel(panel)
mat = log_excess_matrix(cl); V = mat.values
A_, B_ = half_masks(mat)
Za = np.where(np.isnan(V), 0, V) * A_; Zb = np.where(np.isnan(V), 0, V) * B_
ma = 100 * Za.sum(0) / A_.sum(0); mb = 100 * Zb.sum(0) / B_.sum(0)
names = np.array(mat.columns)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
L = 2.6
ax.axhline(0, color="k", linewidth=0.8); ax.axvline(0, color="k", linewidth=0.8)
ax.plot([-L, L], [-L, L], "k:", linewidth=0.8)
inside = (np.abs(ma) <= L) & (np.abs(mb) <= L)
ax.plot(ma[inside], mb[inside], "o", color="C0", markersize=4)
offsets = {"NYSLRS": (5, -9), "WV ERS": (5, 2), "M PERS": (5, 0), "CalPERS": (-4, -11), "SO DAK RS": (5, -3), "LAFPP": (5, -3)}
for f, off in offsets.items():
    j = int(np.where(names == f)[0][0])
    if inside[j]:
        ax.annotate(f, (ma[j], mb[j]), xytext=off, textcoords="offset points", fontsize=7)
j = int(np.where(names == "DAL F&P")[0][0])
ax.annotate("DAL F&P: (%.1f, %.1f)" % (ma[j], mb[j]), xy=(-L + 0.05, -L + 0.05), xytext=(-L + 0.35, -L + 0.45),
            fontsize=7, arrowprops=dict(arrowstyle="->", linewidth=0.8))
rho = stats.spearmanr(ma, mb)[0]
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
ax.set_xlabel("first half of sample: mean log excess return, % p.a.")
ax.set_ylabel("second half of sample, % p.a.")
ax.set_title("rank correlation %.2f" % rho, fontsize=10)
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
anytime = confidence_path(e_process(d)[0])[9:]
xs = yrs[9:]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.axhline(0.5, color="k", linewidth=0.8)
ax.plot(xs, path_paper, "C1-", label="article's numerator")
ax.plot(xs, path_star, "C0-", label="model-consistent $t$-statistic")
ax.plot(xs, ref, "k--", label="constant-gap reference")
ax.plot(xs, anytime, "k:", label="anytime-valid confidence")
ax.set_ylim(0.0, 1.0); ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("fiscal year at which the expanding window ends (first look after ten years)")
ax.set_ylabel("confidence in skill")
ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2025])
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_path.pdf"); fig.savefig(ROOT / "paper" / "fig_path.png", dpi=200)
print("fig 4: CalSTRS terminal: paper %.3f star %.3f ref %.3f; 2023: paper %.3f star %.3f" % (
    path_paper[-1], path_star[-1], ref[-1], path_paper[list(xs).index(2023)], path_star[list(xs).index(2023)]))
