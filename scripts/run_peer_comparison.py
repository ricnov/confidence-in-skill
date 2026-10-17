#!/usr/bin/env python3
"""Comparing plans with each other (Section 4.6 of the paper).

    python scripts/run_peer_comparison.py data/panel_returns.csv

(1) How much of the cross-plan dispersion of total returns within a fiscal year is the policy
    benchmark; how similar the benchmarks are to each other.
(2) Pairwise differences in mean log excess return, paired on common years: how many pairs are
    distinguishable, before and after a false-discovery-rate control, with and without the four
    extreme plans.
(3) Whether confidence values compare skill: the same confidence for very different gaps.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from cis.panel import load_panel
from cis.multiple import bh_reject

if len(sys.argv) < 2:
    sys.exit(__doc__)
panel, _ = load_panel(sys.argv[1])
p = panel.assign(lP=np.log1p(panel.actual), lL=np.log1p(panel.benchmark))
p["d"] = p.lP - p.lL
EXTREME = ["NYSLRS", "LASERS", "DAL F&P", "WV ERS"]

# (1) cross-plan dispersion within a year
rows = []
for y, g in p.groupby("year"):
    if len(g) < 10:
        continue
    rows.append(dict(year=y, n=len(g), sd_P=g.lP.std(ddof=1), sd_L=g.lL.std(ddof=1), sd_d=g.d.std(ddof=1)))
t = pd.DataFrame(rows).set_index("year")
print("== (1) within-year dispersion across plans, %d years with >= 10 plans ==" % len(t))
print("median cross-plan s.d.: total return %.2f%%, benchmark %.2f%%, log excess %.2f%%" %
      (100 * t.sd_P.median(), 100 * t.sd_L.median(), 100 * t.sd_d.median()))
g = p[p.groupby("year").year.transform("size") >= 10].copy()
for c in ["lP", "lL", "d"]:
    g[c + "_w"] = g[c] - g.groupby("year")[c].transform("mean")
print("pooled within-year variance shares of total return: benchmark %.2f, excess %.2f, 2cov %.2f" %
      (g.lL_w.var() / g.lP_w.var(), g.d_w.var() / g.lP_w.var(), 2 * np.cov(g.lL_w, g.d_w)[0, 1] / g.lP_w.var()))
C = p.pivot(index="year", columns="fund", values="lL").corr(min_periods=10).values
pc = C[np.triu_indices_from(C, 1)]
print("pairwise correlation of benchmark returns (>= 10 common years): mean %.2f, 10th pct %.2f, min %.2f" %
      (np.nanmean(pc), np.nanpercentile(pc, 10), np.nanmin(pc)))

# (2) pairwise differences in mean log excess return
E = p.pivot(index="year", columns="fund", values="d")
keys = list(E.columns)
res = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        x = (E[keys[i]] - E[keys[j]]).dropna()
        if len(x) < 10:
            continue
        tt = np.sqrt(len(x)) * x.mean() / x.std(ddof=1)
        res.append(dict(a=keys[i], b=keys[j], n=len(x), diff=x.mean(), se=x.std(ddof=1) / np.sqrt(len(x)),
                        t=tt, p=2 * stats.t.sf(abs(tt), len(x) - 1)))
R = pd.DataFrame(res)
inv = R[~R.a.isin(EXTREME) & ~R.b.isin(EXTREME)]
print("\n== (2) %d pairwise comparisons of mean log excess return, paired on common years ==" % len(R))
print("median s.e. of a difference %.0f bp, median |difference| %.0f bp" % (1e4 * R.se.median(), 1e4 * R["diff"].abs().median()))
print("pairs differing at 5%%: %d (%.1f%%); BH 10%% over all pairs: %d" % ((R.p < .05).sum(), 100 * (R.p < .05).mean(), bh_reject(R.p.values, 0.10)))
print("pairs not involving %s: %d; at 5%%: %.1f%%; BH 10%%: %d" % (EXTREME, len(inv), 100 * (inv.p < .05).mean(), bh_reject(inv.p.values, 0.10)))
ex = R[(R.a.isin(["CalPERS", "NVPERS"])) & (R.b.isin(["CalPERS", "NVPERS"]))]
print("CalPERS vs NVPERS: difference %.0f bp a year, t = %.2f, p = %.2f" % (1e4 * ex["diff"].iloc[0], ex.t.iloc[0], ex.p.iloc[0]))

# (3) confidence against gap
f = p.groupby("fund").agg(n=("d", "size"), gap=("d", "mean"), sd=("d", "std"))
f["t"] = np.sqrt(f.n) * f.gap / f.sd
f["cis"] = 100 * stats.norm.cdf(f.t)
print("\n== (3) confidence and skill ==")
print(f.loc[["SWIB", "MASSPRIM", "MONT COUNTY ERS", "SCRSIC"]].assign(gap_bp=lambda x: 1e4 * x.gap, sd_pct=lambda x: 100 * x.sd)[["n", "gap_bp", "sd_pct", "cis"]].round(1).to_string())
print("rank correlation of confidence with gap/sd %.2f, with gap %.2f, with n %.2f" %
      (stats.spearmanr(f.cis, f.gap / f.sd)[0], stats.spearmanr(f.cis, f.gap)[0], stats.spearmanr(f.cis, f.n)[0]))
