#!/usr/bin/env python3
"""Cross-section on the author's annual returns (data/panel_returns.csv, or the workbook).

    python scripts/run_raw_analysis.py data/panel_returns.csv [B]      (or the author's workbook .xlsx)

Writes results/raw_results.csv (per-fund statistics, every variant) and prints the
dependence-aware bootstrap summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import cis
from cis.panel import load_panel, log_excess_matrix
from cis.bootstrap import wild_bootstrap, adjusted_p
from cis.multiple import bh_q, bh_reject, holm_reject

if len(sys.argv) < 2:
    sys.exit(__doc__)
path = Path(sys.argv[1])
B = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

panel, meta = load_panel(path)
rows = {}
for fund, d in panel.groupby("fund"):
    d = d.sort_values("year")
    rows[fund] = cis.skill_from_returns(d["actual"].values, d["benchmark"].values)
res = pd.DataFrame(rows).T
res = res.join(meta[["sheet_K", "sheet_CIS", "sheet_TE", "sheet_rho"]])
res["q_t"] = bh_q(res["p_t"].values)
res["q_t_under"] = bh_q(res["p_t_under"].values)
res = res.sort_values("t", ascending=False)

print("funds %d, fund-years %d" % (res.shape[0], len(panel)))
print("sheets reproduced exactly with the geometric numerator: %d; with the arithmetic numerator: %d" %
      ((res["k_paper"] - res["sheet_K"]).abs().lt(1e-3).sum(), (res["k_arith"] - res["sheet_K"]).abs().lt(1e-3).sum()))
print("above 90%%: sheet %d | arithmetic %d | model-consistent %d" %
      ((res["sheet_CIS"] > 0.9).sum(), (res["cis_arith"] > 0.9).sum(), (res["cis_star"] > 0.9).sum()))
print("median |drag|/|gap*| = %.2f; drag > |gap*|: %s" %
      ((res["drag"].abs() / res["gap_star"].abs()).median(), list(res.index[res["drag"].abs() > res["gap_star"].abs()])))
for a in (0.05, 0.10, 0.20):
    print("alpha=%.2f: BH out %d / under %d | Holm out %d / under %d" %
          (a, bh_reject(res["p_t"], a), bh_reject(res["p_t_under"], a), holm_reject(res["p_t"], a), holm_reject(res["p_t_under"], a)))

mat = log_excess_matrix(panel)
C = mat.corr(min_periods=8).values
pc = C[np.triu_indices_from(C, 1)]
print("mean pairwise correlation of annual log excess returns: %.2f" % np.nanmean(pc))

boot = wild_bootstrap(mat, B=B, seed=20261011)
obs = int((stats.norm.cdf(boot["t"]) > 0.9).sum())
print("year-cluster wild bootstrap, B=%d: count above 90%% observed %d, null mean %.1f, 95%% %d, P(>= obs) = %.4f" %
      (B, obs, boot["count_normal"].mean(), np.percentile(boot["count_normal"], 95), (boot["count_normal"] >= obs).mean()))
print("max t null 95%% = %.2f (independence benchmark would be about 3.6); observed %.2f" %
      (np.percentile(boot["max_t"], 95), boot["t"].max()))
adj = adjusted_p(boot)
print(adj.head(6).round(4).to_string())
print(adj.tail(3).round(4).to_string())

out = Path("results"); out.mkdir(exist_ok=True)
res.join(adj[["p_single_up", "p_stepdown_up", "p_single_down"]]).to_csv(out / "raw_results.csv")
print("wrote results/raw_results.csv")
