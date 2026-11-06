"""How fast can skill vary? A pooled local-level model of time-varying skill.

    python scripts/run_time_varying_skill.py [data/panel_returns_extended.csv] [--clean] [--B 2000]

The article's premise is that skill varies through time. Model each programme's year-demeaned log excess
return (the common component removed) as a random-walk skill plus noise,

    y_{it} = alpha_{it} + eps_{it},   alpha_{it} = alpha_{i,t-1} + eta_{it},   eps ~ N(0, s_i^2),  eta ~ N(0, q s_i^2),

with a diffuse initial level and ONE signal-to-noise ratio q for the panel. The Kalman filter gives each
programme's likelihood with s_i^2 concentrated out; the pooled profile likelihood over q is maximised on a
grid. q = 0 is constant skill. The likelihood-ratio statistic for q = 0 sits on the boundary of the parameter
space, so its null distribution is simulated by the year-cluster wild bootstrap under constant skill (sign
flips of year clusters of the plan-demeaned returns; the diffuse level makes the likelihood invariant to each
plan's constant mean). The 95% upper bound on q is the largest q whose profile likelihood lies within the
bootstrap-calibrated critical value of the maximum; it is translated into the standard deviation of a
programme's year-to-year change in skill at the panel's median tracking error, and into the change over ten
years. Outputs results/time_varying_skill_{extended,clean}.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cis.panel import load_panel, clean_panel, log_excess_matrix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("panel", nargs="?", default=str(ROOT / "data" / "panel_returns_extended.csv"))
ap.add_argument("--clean", action="store_true")
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20261011)
ap.add_argument("--no-demean-years", action="store_true", help="keep the common component in the data")
args = ap.parse_args()

QGRID = np.concatenate([[0.0], np.logspace(-4, 0.5, 120)])


def loglik_local_level(y, q):
    """Profile log-likelihood (s^2 concentrated out) of the local-level model with diffuse initial level.

    Kalman filter in units of s^2: P_t is the level variance / s^2. The first observation initialises the
    level (diffuse), contributing nothing to the likelihood; the remaining n-1 innovations v_t have variance
    s^2 F_t. Concentrated: s^2 = sum(v^2/F)/(n-1), loglik = -(n-1)/2 [log(2 pi s^2) + 1] - 1/2 sum log F."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return 0.0
    a = y[0]; P = 1.0 + q     # diffuse prior: after y_0 the level is N(y_0, s^2), i.e. P = 1 in s^2 units, plus q for the step
    sumlog = 0.0; sumv = 0.0
    for t in range(1, n):
        F = P + 1.0
        v = y[t] - a
        sumv += v * v / F; sumlog += np.log(F)
        K = P / F
        a = a + K * v
        P = P * (1 - K) + q
    s2 = sumv / (n - 1)
    return -(n - 1) / 2 * (np.log(2 * np.pi * s2) + 1) - 0.5 * sumlog


def pooled_profile(mat):
    """Pooled profile log-likelihood over the grid; mat is years x funds with NaN."""
    cols = [mat[f].dropna().values for f in mat.columns]
    return np.array([sum(loglik_local_level(y, q) for y in cols) for q in QGRID])


panel, _ = load_panel(args.panel)
if args.clean:
    panel = clean_panel(panel)
mat = log_excess_matrix(panel)
if not args.no_demean_years:
    mat = mat.sub(mat.mean(axis=1), axis=0)
te_median = float(np.nanmedian(mat.std(ddof=1)))
prof = pooled_profile(mat)
qhat = float(QGRID[prof.argmax()]); lr = float(prof.max() - prof[0])
print("q-hat %.4f  LR(q=0) %.2f  median TE %.4f" % (qhat, lr, te_median))

# ---------------------------------------------------------------- year-cluster null of the LR under constant skill
rng = np.random.default_rng(args.seed)
M = mat.values
dem = M - np.nanmean(M, axis=0)
lrs = []; qs = []
for b in range(args.B):
    Xb = dem * rng.choice([-1.0, 1.0], size=(M.shape[0], 1))
    pb = pooled_profile(pd.DataFrame(Xb, index=mat.index, columns=mat.columns))
    lrs.append(pb.max() - pb[0]); qs.append(QGRID[pb.argmax()])
lrs = np.array(lrs); qs = np.array(qs)
p_lr = float((lrs >= lr).mean())
crit95 = float(np.percentile(lrs, 95))
# 95% upper bound on q: largest grid q with prof[q] >= prof.max() - crit95
ok = prof >= prof.max() - crit95
q_upper = float(QGRID[ok].max())
sd_eta_bp = 1e4 * np.sqrt(q_upper) * te_median
res = {"programmes": int(mat.shape[1]), "plan_years": int(np.isfinite(M).sum()), "q_hat": qhat, "LR_q0": lr, "P_LR_year_cluster": p_lr,
       "LR_null_mean": float(lrs.mean()), "LR_null_95": crit95, "q_null_median": float(np.median(qs)), "q_null_95": float(np.percentile(qs, 95)),
       "q_upper_95": q_upper, "median_te": te_median,
       "sd_annual_skill_change_bp_upper": float(sd_eta_bp), "sd_ten_year_skill_change_bp_upper": float(sd_eta_bp * np.sqrt(10)),
       "sd_annual_skill_change_bp_at_qhat": float(1e4 * np.sqrt(qhat) * te_median),
       "profile": {"%g" % q: float(v) for q, v in zip(QGRID, prof)}, "B": args.B, "demeaned_years": not args.no_demean_years}
print(json.dumps({k: v for k, v in res.items() if k != "profile"}, indent=1))
out = str(ROOT / "results" / ("time_varying_skill_%s.json" % ("clean" if args.clean else "extended")))
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)
