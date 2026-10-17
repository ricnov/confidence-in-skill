#!/usr/bin/env python3
"""Every number the paper quotes, from one command.

    python scripts/run_everything.py data/panel_returns.csv [--clean] [--ny NYSCRF] [--B 50000] [--out results/summary_X.json]

Without --clean: the article's 47 rows as they stand. With --clean: the 42 distinct, correctly
benchmarked programmes of cis.panel.clean_panel (and --ny NYSCRF measures the New York fund against the
NYSCRF sheet's benchmark instead of the NYSLRS sheet's). Seeds are the paper's (20261011 for the
bootstrap, 20261012 for the simulations).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import cis
from cis.panel import load_panel, clean_panel, log_excess_matrix
from cis.bootstrap import wild_bootstrap, adjusted_p
from cis.dispersion import null_functionals, summarise, half_masks, functionals
from cis.multiple import bh_q, bh_reject, holm_reject

ap = argparse.ArgumentParser()
ap.add_argument("panel")
ap.add_argument("--clean", action="store_true")
ap.add_argument("--ny", default="NYSLRS")
ap.add_argument("--B", type=int, default=50000)
ap.add_argument("--out", default=None)
a = ap.parse_args()

panel, meta = load_panel(a.panel)
label = "article's 47 rows"
if a.clean:
    panel = clean_panel(panel, ny_benchmark=a.ny)
    label = "cleaned panel, NY against the %s sheet's benchmark" % a.ny
S = {"panel": label, "B": a.B}
print("=== %s ===" % label)

# ---------------------------------------------------------------- per-plan statistics
rows = {}
for fund, d in panel.groupby("fund"):
    d = d.sort_values("year")
    rows[fund] = cis.skill_from_returns(d["actual"].values, d["benchmark"].values)
res = pd.DataFrame(rows).T
if "sheet_CIS" in meta.columns:
    res = res.join(meta[["sheet_CIS"]])
res["q_t"] = bh_q(res["p_t"].values)
res = res.sort_values("t", ascending=False)
m = len(res)
S["plans"] = m
S["plan_years"] = int(res["n"].sum())
S["above90"] = {"sheet": int((res.get("sheet_CIS", pd.Series(dtype=float)) > 0.9).sum()) if "sheet_CIS" in res else None,
                "arith": int((res["cis_arith"] > 0.9).sum()), "star": int((res["cis_star"] > 0.9).sum())}
S["below10_star"] = int((res["cis_star"] < 0.1).sum())
S["above50_star"] = int((res["cis_star"] > 0.5).sum())
S["mean_cis_star"] = round(float(res["cis_star"].mean()), 3)
S["ks_p_star"] = round(float(stats.kstest(res["cis_star"], "uniform").pvalue), 4)
S["binom_p_above90"] = round(float(stats.binom.sf(S["above90"]["star"] - 1, m, 0.10)), 4)
S["expected_above90_t"] = round(float(stats.t.sf(stats.norm.ppf(0.9), df=res["n"] - 1).sum()), 2)
S["drag_share_median"] = round(float((res["drag"].abs() / res["gap_star"].abs()).median()), 3)
S["drag_exceeds_gap"] = list(res.index[res["drag"].abs() > res["gap_star"].abs()])
S["sign_flip_paper_vs_star"] = list(res.index[np.sign(res["k_paper"]) != np.sign(res["t"])])
S["median_sd_log_excess_pct"] = round(float(100 * panel.assign(d=np.log1p(panel.actual) - np.log1p(panel.benchmark))
                                            .groupby("fund")["d"].std(ddof=1).median()), 2)
S["BH_Holm"] = {str(al): {"bh_out": int(bh_reject(res["p_t"].values, al)), "bh_under": int(bh_reject(res["p_t_under"].values, al)),
                          "holm_out": int(holm_reject(res["p_t"].values, al)), "holm_under": int(holm_reject(res["p_t_under"].values, al))}
                for al in (0.05, 0.10, 0.20)}
S["bh20_outperformers"] = list(res.index[res["q_t"] <= 0.20])
S["top5"] = res.head(5)[["n", "t", "cis_star", "p_t", "q_t"]].round(4).to_dict("index")
S["bottom3"] = res.tail(3)[["n", "t", "cis_star", "p_t"]].round(4).to_dict("index")
d_norm_t = 100 * (res["cis_star"] - stats.t.cdf(res["t"], df=res["n"] - 1))
S["normal_minus_t_pct"] = {"mean_abs": round(float(d_norm_t.abs().mean()), 2), "max_abs": round(float(d_norm_t.abs().max()), 2)}
print("plans %d, plan-years %d; above 90%%: %s; BH/Holm: %s" % (m, S["plan_years"], S["above90"], S["BH_Holm"]["0.1"]))
print(res.head(6)[["n", "t", "cis_star", "p_t", "q_t"]].round(4).to_string())
print(res.tail(3)[["n", "t", "cis_star", "p_t", "q_t"]].round(4).to_string())

# ---------------------------------------------------------------- dependence
mat = log_excess_matrix(panel)
C = mat.corr(min_periods=8).values
pc = C[np.triu_indices_from(C, 1)]
S["mean_pairwise_corr_excess"] = round(float(np.nanmean(pc)), 3)
S["median_pairwise_corr_excess"] = round(float(np.nanmedian(pc)), 3)
boot = wild_bootstrap(mat, B=a.B, seed=20261011)
obs_count = int((stats.norm.cdf(boot["t"]) > 0.9).sum())
S["bootstrap"] = {"count_obs": obs_count, "count_null_mean": round(float(boot["count_normal"].mean()), 2),
                  "count_null_95": int(np.percentile(boot["count_normal"], 95)), "count_null_99": int(np.percentile(boot["count_normal"], 99)),
                  "count_P": round(float((boot["count_normal"] >= obs_count).mean()), 4),
                  "count_P_tref": round(float((boot["count_t"] >= int((stats.t.cdf(boot["t"], df=boot["n"] - 1) > 0.9).sum())).mean()), 4),
                  "max_t_null_95": round(float(np.percentile(boot["max_t"], 95)), 2), "max_t_null_99": round(float(np.percentile(boot["max_t"], 99)), 2),
                  "min_t_null_5": round(float(np.percentile(boot["min_t"], 5)), 2)}
adj = adjusted_p(boot)
S["adjusted_p_top"] = adj.head(4)[["t", "n", "p_single_up", "p_stepdown_up"]].round(4).to_dict("index")
S["adjusted_p_bottom"] = adj.tail(3)[["t", "n", "p_single_down"]].round(4).to_dict("index")
# exact independence benchmarks
n = boot["n"]
F = lambda x: np.prod(stats.t.cdf(x, df=n - 1))
from scipy import optimize
S["independence"] = {"max_t_95": round(float(optimize.brentq(lambda x: F(x) - 0.95, 2, 7)), 2)}
p_i = stats.t.sf(stats.norm.ppf(0.9), df=n - 1)
poly = np.array([1.0])
for pi in p_i:
    poly = np.convolve(poly, [1 - pi, pi])
cdf = np.cumsum(poly)
S["independence"]["count_95"] = int(np.searchsorted(cdf, 0.95))
S["independence"]["P_count_ge_obs"] = round(float(1 - cdf[obs_count - 1]), 4)
print("bootstrap:", S["bootstrap"]); print(adj.head(4).round(4).to_string()); print(adj.tail(2).round(4).to_string())

# ---------------------------------------------------------------- dispersion / time variation / persistence
extremes = [f for f in ["NYSLRS", "LASERS", "DAL F&P", "WV ERS"] if f in mat.columns]
sets = {"all": [], "without_top_bottom": [f for f in [adj.index[0], adj.index[-1]] if f in mat.columns], "without_extremes": extremes}
S["dispersion"] = {}
for key, excl in sets.items():
    sub = mat.drop(columns=excl)
    null = null_functionals(sub, B=min(a.B, 50000), seed=20261011)
    sm = summarise(sub, null)
    S["dispersion"][key] = {"excluded": excl, "funds": int(sub.shape[1]),
                            "table": sm.round(4).to_dict("index"),
                            "tau_pct": round(float(100 * np.sqrt(max(sm.loc["sd_mean", "observed"] ** 2 - sm.loc["sd_mean", "null_rms"] ** 2, 0))), 2)}
    print("dispersion [%s] funds %d:" % (key, sub.shape[1])); print(sm.round(4).to_string())
# persistence: correlation across plans of first-half and second-half mean excess returns, with the same null
A_, B_ = half_masks(mat)
V = mat.values
def halves(X):
    Za = np.where(np.isnan(X), 0, X) * A_; Zb = np.where(np.isnan(X), 0, X) * B_
    return Za.sum(0) / A_.sum(0), Zb.sum(0) / B_.sum(0)
ma, mb = halves(V)
obs_r = float(np.corrcoef(ma, mb)[0, 1]); obs_rho = float(stats.spearmanr(ma, mb)[0])
rng = np.random.default_rng(20261012)
D0 = V - np.nanmean(V, axis=0)
rs = np.empty(20000); rhos = np.empty(20000)
for b in range(20000):
    s_ = rng.choice([-1.0, 1.0], size=(V.shape[0], 1)); xa, xb = halves(D0 * s_)
    rs[b] = np.corrcoef(xa, xb)[0, 1]; rhos[b] = stats.spearmanr(xa, xb)[0]
S["persistence"] = {"pearson": round(obs_r, 3), "pearson_null_95": round(float(np.percentile(rs, 95)), 3), "pearson_P": round(float((rs >= obs_r).mean()), 4),
                    "spearman": round(obs_rho, 3), "spearman_null_mean": round(float(rhos.mean()), 3), "spearman_null_95": round(float(np.percentile(rhos, 95)), 3),
                    "spearman_P": round(float((rhos >= obs_rho).mean()), 4)}
sub_idx = [i for i, f in enumerate(mat.columns) if f not in extremes]
S["persistence"]["spearman_without_extremes"] = round(float(stats.spearmanr(ma[sub_idx], mb[sub_idx])[0]), 3)
print("persistence:", S["persistence"])

# ---------------------------------------------------------------- sequential crossings on the panel
def crossings(first_look):
    cnt = 0
    for j in range(V.shape[1]):
        x = V[:, j]; x = x[~np.isnan(x)]; k = np.arange(1, x.size + 1)
        cs = np.cumsum(x); cs2 = np.cumsum(x ** 2); mean = cs / k; var = (cs2 - k * mean ** 2) / np.maximum(k - 1, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_ = np.sqrt(k) * mean / np.sqrt(var)
        c = stats.norm.cdf(t_[first_look - 1:])
        cnt += int(c.size and np.nanmax(c) > 0.9)
    return cnt
def crossings_null(first_look, Bn=3000):
    out = np.empty(Bn, int)
    for b in range(Bn):
        s_ = rng.choice([-1.0, 1.0], size=(V.shape[0], 1)); X = D0 * s_; cnt = 0
        for j in range(V.shape[1]):
            x = X[:, j]; x = x[~np.isnan(x)]; k = np.arange(1, x.size + 1)
            cs = np.cumsum(x); cs2 = np.cumsum(x ** 2); mean = cs / k; var = (cs2 - k * mean ** 2) / np.maximum(k - 1, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                t_ = np.sqrt(k) * mean / np.sqrt(var)
            c = stats.norm.cdf(t_[first_look - 1:]); cnt += int(c.size and np.nanmax(c) > 0.9)
        out[b] = cnt
    return out
S["crossings"] = {}
for fl in (1, 10):
    oc = crossings(fl); nc = crossings_null(fl)
    S["crossings"][str(fl)] = {"observed": oc, "null_mean": round(float(nc.mean()), 1), "null_95": int(np.percentile(nc, 95)), "P": round(float((nc >= oc).mean()), 3)}
print("crossings:", S["crossings"])

# ---------------------------------------------------------------- peer comparison
p2 = panel.assign(lP=np.log1p(panel.actual), lL=np.log1p(panel.benchmark)); p2["d"] = p2.lP - p2.lL
E = p2.pivot(index="year", columns="fund", values="d"); keys = list(E.columns); pr = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        x = (E[keys[i]] - E[keys[j]]).dropna()
        if len(x) < 10: continue
        tt = np.sqrt(len(x)) * x.mean() / x.std(ddof=1)
        pr.append(dict(a=keys[i], b=keys[j], diff=x.mean(), se=x.std(ddof=1) / np.sqrt(len(x)), p=2 * stats.t.sf(abs(tt), len(x) - 1)))
R = pd.DataFrame(pr); inv = R[~R.a.isin(extremes) & ~R.b.isin(extremes)]
BB = p2.pivot(index="year", columns="fund", values="lL").corr(min_periods=10).values; pcb = BB[np.triu_indices_from(BB, 1)]
g = p2[p2.groupby("year").year.transform("size") >= 10].copy()
for c_ in ["lP", "lL", "d"]:
    g[c_ + "_w"] = g[c_] - g.groupby("year")[c_].transform("mean")
S["peer"] = {"pairs": len(R), "median_se_bp": round(float(1e4 * R.se.median())), "median_absdiff_bp": round(float(1e4 * R["diff"].abs().median())),
             "pairs_5pct": int((R.p < .05).sum()), "bh10_pairs": int(bh_reject(R.p.values, 0.10)),
             "pairs_without_extremes": len(inv), "share_5pct_without": round(float((inv.p < .05).mean()), 3), "bh10_without": int(bh_reject(inv.p.values, 0.10)),
             "benchmark_corr_mean": round(float(np.nanmean(pcb)), 2), "benchmark_corr_p10": round(float(np.nanpercentile(pcb, 10)), 2),
             "var_share_benchmark": round(float(g.lL_w.var() / g.lP_w.var()), 2)}
ex = R[R.a.isin(["CalPERS", "NVPERS"]) & R.b.isin(["CalPERS", "NVPERS"])]
if len(ex): S["peer"]["calpers_vs_nvpers"] = {"diff_bp": round(float(1e4 * ex["diff"].iloc[0])), "p": round(float(ex.p.iloc[0]), 2)}
print("peer:", S["peer"])

out = Path(a.out) if a.out else Path("results") / ("summary_%s%s.json" % ("clean" if a.clean else "article", "_" + a.ny if a.clean and a.ny != "NYSLRS" else ""))
out.parent.mkdir(exist_ok=True)
json.dump(S, open(out, "w"), indent=1, default=str)
res.to_csv(out.with_suffix(".csv"))
print("wrote", out)
