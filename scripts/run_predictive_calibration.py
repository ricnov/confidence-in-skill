"""Confidence as a probability of outperformance: calibration, and the effective number of independent tests.

    python scripts/run_predictive_calibration.py [--B 20000] [--out results/predictive_calibration.json]

Two small results on top of scripts/run_framework_tests.py and the year-cluster null.

1. Predictive probability. With the normal prior N(0, tau^2) on a programme's true mean excess return
   and a record of n1 years with mean m1 and standard deviation s1, the posterior is normal with mean
   w*m1 and variance v, w = tau^2/(tau^2+se1^2), v = tau^2 se1^2/(tau^2+se1^2), se1 = s1/sqrt(n1), and the
   probability of being ahead of the benchmark over the next H years is
       Phi( w*m1 / sqrt(v + s1^2/H) ).
   Read as a probability, the framework's confidence Phi(t) is the limit tau -> infinity AND H -> infinity:
   it is the probability that the true mean is positive under a flat prior, not the probability of
   outperforming over any horizon a trustee will see. The script scores both readings on the panel: each
   programme's first ten years are the record, the later years the sequel (as in run_framework_tests.py),
   and the outcome is whether the sequel's mean excess return is positive. Brier and log scores against the
   base rate; calibration by tercile of the forecast.

2. Effective number of independent tests. The year-cluster null gives the variance of the count of
   programmes above 90% under no skill; the binomial variance m p (1-p) is what independence would give;
   the ratio says how many independent tests the m dependent ones are worth: m_eff = m / ratio.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cis.panel import load_panel, clean_panel, log_excess_matrix  # noqa: E402
from cis.dispersion import null_functionals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=20000)
ap.add_argument("--seed", type=int, default=20261011)
ap.add_argument("--out", default=str(ROOT / "results" / "predictive_calibration.json"))
args = ap.parse_args()
S = {}

# ------------------------------------------------------------------ 1. calibration on the record/sequel split
for lab, fname in (("extended", "framework_tests_extended.json"), ("clean", "framework_tests_clean.json")):
    path = ROOT / "results" / fname
    if not path.exists():
        continue
    F = json.load(open(path))
    T = pd.DataFrame(F["all"]["table"]).T.astype(float)
    tau = F["all"]["shrinkage"]["tau_record_zero_bp"]
    T["se1"] = T["mean_record_bp"] / T["t_record"]
    T["s1"] = T["se1"] * np.sqrt(T["n_record"])
    w = tau ** 2 / (tau ** 2 + T["se1"] ** 2)
    m = w * T["mean_record_bp"]
    v = tau ** 2 * T["se1"] ** 2 / (tau ** 2 + T["se1"] ** 2)
    T["p_pred"] = stats.norm.cdf(m / np.sqrt(v + T["s1"] ** 2 / T["n_sequel"]))
    T["p_naive"] = stats.norm.cdf(T["t_record"])
    T["y"] = (T["mean_sequel_bp"] > 0).astype(float)
    base = float(T["y"].mean())

    def brier(p):
        return float(np.mean((p - T["y"]) ** 2))

    def logscore(p):
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return float(np.mean(T["y"] * np.log(p) + (1 - T["y"]) * np.log(1 - p)))

    res = {"programmes": int(len(T)), "tau_record_bp": tau, "base_rate_positive_sequel": round(base, 3),
           "brier": {"naive_cis": round(brier(T["p_naive"]), 3), "predictive": round(brier(T["p_pred"]), 3),
                     "base_rate": round(brier(pd.Series(base, index=T.index)), 3), "coin": 0.25},
           "log_score": {"naive_cis": round(logscore(T["p_naive"]), 3), "predictive": round(logscore(T["p_pred"]), 3),
                         "base_rate": round(logscore(pd.Series(base, index=T.index)), 3)}}
    for col in ("p_naive", "p_pred"):
        b = pd.qcut(T[col], 3, labels=["low", "mid", "high"])
        g = T.groupby(b, observed=True).agg(n=("y", "size"), forecast=(col, "mean"), realised=("y", "mean"), sequel_bp=("mean_sequel_bp", "mean"))
        res["terciles_" + col] = g.round(3).to_dict(orient="index")
    hi = T["cis_record"] >= 90; lo = T["cis_record"] < 50
    res["group_at_or_above_90"] = {"n": int(hi.sum()), "naive": round(float(T["p_naive"][hi].mean()), 3), "predictive": round(float(T["p_pred"][hi].mean()), 3), "realised": round(float(T["y"][hi].mean()), 3)}
    res["group_below_50"] = {"n": int(lo.sum()), "naive": round(float(T["p_naive"][lo].mean()), 3), "predictive": round(float(T["p_pred"][lo].mean()), 3), "realised": round(float(T["y"][lo].mean()), 3)}
    res["per_plan"] = T[["n_record", "t_record", "cis_record", "n_sequel", "mean_sequel_bp", "p_naive", "p_pred", "y"]].round(4).to_dict(orient="index")
    S["calibration_" + lab] = res
    print(lab, json.dumps({k: v for k, v in res.items() if k != "per_plan"}, indent=1))

# ------------------------------------------------------------------ horizon table for a median plan
rows = []
for n1, te in ((16, 1.54), (10, 1.54)):
    se1 = te / np.sqrt(n1)
    for cis in (0.90, 0.99):
        t = stats.norm.ppf(cis); d = t * se1
        for tau_ in (0.56, 0.30):
            w = tau_ ** 2 / (tau_ ** 2 + se1 ** 2); m = w * d; v = tau_ ** 2 * se1 ** 2 / (tau_ ** 2 + se1 ** 2)
            rows.append(dict(n_record=n1, te_pct=te, cis=cis, tau_pct=tau_, posterior_skill=round(float(stats.norm.cdf(m / np.sqrt(v))), 3),
                             **{"ahead_%dy" % H: round(float(stats.norm.cdf(m / np.sqrt(v + te ** 2 / H))), 3) for H in (1, 5, 10, 20)}))
S["horizon_table_median_plan"] = rows
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ 2. effective number of independent tests
eff = {}
for lab, src, clean in (("42", "panel_returns.csv", True), ("50", "panel_returns_extended.csv", False)):
    path = ROOT / "data" / src
    if not path.exists():
        continue
    panel, _ = load_panel(path)
    if clean:
        panel = clean_panel(panel)
    mat = log_excess_matrix(panel)
    null = null_functionals(mat, B=args.B, seed=args.seed)
    m = mat.shape[1]; p = 0.10
    ratio = float(null["count"].var() / (m * p * (1 - p)))
    eff[lab] = {"programmes": m, "null_count_mean": round(float(null["count"].mean()), 2), "null_count_sd": round(float(null["count"].std()), 2),
                "binomial_sd": round(float(np.sqrt(m * p * (1 - p))), 2), "variance_ratio": round(ratio, 2), "m_eff": round(m / ratio, 1),
                "null_count_95": float(np.percentile(null["count"], 95))}
    print(lab, eff[lab])
S["effective_tests"] = eff
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(S, open(args.out, "w"), indent=1, default=float)
print("wrote", args.out)
