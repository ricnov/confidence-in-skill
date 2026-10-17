#!/usr/bin/env python3
"""Reproduce every cross-sectional number in the note from the published appendix alone.

    python scripts/check_published_numbers.py

No return series are needed: the inputs are the 47 rows of Appendix III as printed.
"""
import numpy as np
import pandas as pd
from scipy import stats

import cis
from cis.multiple import bh_reject, holm_reject, bh_thresholds

pd.set_option("display.width", 160)

df = cis.load_published()
s = cis.summary(df)
print("funds: %d | above 90%%: %d | expected under the null: %.1f (normal) %.1f (t)" %
      (s["n_funds"], s["n_above_90"], s["expected_above_90_normal"], s["expected_above_90_t"]))
print("mean CIS %.3f | KS vs U(0,1) p = %.3f | P(>=13 | Binomial(47, 0.1)) = %.4f" %
      (s["mean_cis"], s["ks_p"], s["binomial_p_13"]))
print("BH thresholds, ranks 1-3 at 10%%: %s" % [round(x, 5) for x in bh_thresholds(47, 0.10)])

mc = cis.model_consistent_from_appendix(df).sort_values("K_star", ascending=False)
print("\nmodel-consistent statistic from the appendix columns (top 8 / bottom 5):")
cols = ["years", "cis", "p_pub", "K_star", "CIS_star", "p_t", "q"]
print(pd.concat([mc.head(8), mc.tail(5)])[cols].round(4).to_string())

for a in (0.05, 0.10, 0.20):
    print("alpha=%.2f  published column: BH out %d / under %d | Holm out %d / under %d | model-consistent BH out %d" %
          (a, bh_reject(mc["p_pub"], a), bh_reject(mc["cis"], a), holm_reject(mc["p_pub"], a),
           holm_reject(mc["cis"], a), bh_reject(mc["p_t"], a)))

# rounding edges of the published whole-percentage values
pub = df["cis"].values
print("rounding: favourable edge BH10 out %d / under %d; unfavourable edge out %d / under %d" %
      (bh_reject(np.clip(1 - (pub + 0.005), 0, 1), 0.10), bh_reject(np.clip(pub - 0.005, 0, 1), 0.10),
       bh_reject(np.clip(1 - (pub - 0.005), 0, 1), 0.10), bh_reject(np.clip(pub + 0.005, 0, 1), 0.10)))

d = cis.implied_drag(df)
print("\ndrag recovered by inverting the published confidence (45 uncensored funds):")
print("median |drag|/|gap*| = %.2f ; IQR of sigma_P^2 - sigma_L^2: %s ; drag > |gap*| for %s" %
      (d["drag_share"].median(), d["variance_diff"].quantile([0.25, 0.75]).round(5).tolist(),
       list(d.index[d["drag"].abs() > d["gap_star"].abs()])))

# constant-gap reference path and the t-vs-normal gap
yrs = np.array([5, 10, 15, 20, 25, 30, 40])
print("\nconstant 25 bp gap / 150 bp TE path:", (100 * stats.norm.cdf(np.sqrt(yrs) * 0.0025 / 0.015)).round(1))
for n in (12, 34):
    print("t-distribution confidence at K = 1.28 with n = %d: %.1f%%" % (n, 100 * stats.t.cdf(stats.norm.ppf(0.9), df=n - 1)))
