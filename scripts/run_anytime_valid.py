"""An anytime-valid confidence in skill: the framework's path, read every year, with its error rate controlled.

    python scripts/run_anytime_valid.py [data/panel_returns_extended.csv] [--clean] [--a 0.5] [--B 2000]

The framework's statistic is scale-free: K is the t-statistic of the annual log excess returns d_1, ..., d_n,
assumed i.i.d. N(mu, sigma^2) with both parameters unknown. Reduce the data by scale to the direction vector
u_n = d_{1:n} / ||d_{1:n}||, whose law under H0: mu = 0 does not depend on sigma. The likelihood ratio of u_n
under delta = mu/sigma against delta = 0, mixed over a half-normal prior delta ~ |N(0, a^2)|, is

    M_n = 2 (1 + n a^2)^{-1/2}  E_R[ exp(kappa^2 R^2 / 2) Phi(kappa R) ],     kappa = a B / (sqrt(A) sqrt(1 + n a^2)),

with A = sum d_i^2, B = sum d_i and R a chi random variable with n degrees of freedom. The headline uses an
equal-weight mixture over the prior scales a = 0.25, 0.5 and 1 (an average of e-processes is an e-process): the
cross-section's spread of true information ratios is 0.2-0.4 at its centre with far members (SWIB's is 1.7), so a
single scale either lacks power for the far members or over-rewards short strong starts. Because the direction
vectors form a filtration (u_{n-1} is a function of u_n), (M_n) is a nonnegative martingale with M_0 = 1 under
H0 whatever sigma is, so by Ville's inequality  P(exists n: M_n >= 1/alpha) <= alpha. The anytime-valid
confidence in skill is C_n = 1 - 1 / max_{k <= n} M_k (0 when the running maximum is below 1): a trustee who
acts the first time C_n >= 90% has at most a 10% chance of ever acting on a plan with no skill, however
many years the plan is examined. The two-sided version (full normal prior) is closed form,
(1 + n a^2)^{-1/2} (1 - kappa^2)^{-n/2}, and is reported alongside.

Outputs: per plan and year the naive confidence Phi(t_n), the e-process and C_n; first crossing years at 90%
and 95%; a null simulation (i.i.d. normal, 25 and 50 years) of the ever-crossing probability and of E[M_n];
power for a constant information ratio; and the year-cluster null distribution of the number of programmes
whose anytime-valid confidence ever reaches 90%.
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

ap = argparse.ArgumentParser()
ap.add_argument("panel", nargs="?", default=str(ROOT / "data" / "panel_returns_extended.csv"))
ap.add_argument("--clean", action="store_true", help="apply the exclusions of the paper's Section 2")
ap.add_argument("--a", default="0.25,0.5,1", help="prior scale(s) of the information ratio delta = mu/sigma; several values give the equal-weight mixture")
ap.add_argument("--B", type=int, default=1500, help="year-cluster null replicates")
ap.add_argument("--nsim", type=int, default=6000, help="null simulation size")
ap.add_argument("--seed", type=int, default=20261011)
ap.add_argument("--first-look", type=int, default=10, help="first year at which the naive path is read")
ap.add_argument("--out", default=None)
ap.add_argument("--skip-sim", action="store_true", help="skip the i.i.d. null simulation and the power study (panel-independent; see the extended run)")
args = ap.parse_args()
A_PRIOR = tuple(float(v) for v in str(args.a).split(","))
LEVELS = (0.90, 0.95)

from cis.anytime import e_process, confidence_path, naive_path, first_cross  # noqa: E402

# ------------------------------------------------------------------ 1. the panel
panel, _ = load_panel(args.panel)
if args.clean:
    panel = clean_panel(panel)
mat = log_excess_matrix(panel)        # years x funds
years = mat.index.values
rows = []
paths = {}
for f in mat.columns:
    s = mat[f].dropna()
    x = s.values; yrs = s.index.values
    one, two = e_process(x, a=A_PRIOR)
    C1 = confidence_path(one); C2 = confidence_path(two); naive = naive_path(x)
    paths[f] = pd.DataFrame({"year": yrs, "n": np.arange(1, len(x) + 1), "naive": naive, "M_one": one, "C_one": C1, "M_two": two, "C_two": C2})
    fl = args.first_look - 1
    r = {"fund": f, "n": len(x), "mean_bp": 1e4 * x.mean(), "t": np.sqrt(len(x)) * x.mean() / x.std(ddof=1),
         "naive_final": naive[-1], "C_one_final": C1[-1], "C_two_final": C2[-1], "M_one_final": one[-1], "M_one_max": np.nanmax(one)}
    for lev in LEVELS:
        k = first_cross(naive, lev, fl); r["naive_first_%d" % (100 * lev)] = int(yrs[k]) if k is not None else None
        k = first_cross(C1, lev, 0); r["anytime_first_%d" % (100 * lev)] = int(yrs[k]) if k is not None else None
        k = first_cross(C2, lev, 0); r["anytime_two_first_%d" % (100 * lev)] = int(yrs[k]) if k is not None else None
    rows.append(r)
P = pd.DataFrame(rows).set_index("fund").sort_values("t", ascending=False)
summary = {"programmes": int(len(P)), "a_prior": list(A_PRIOR), "first_look_naive": args.first_look}
for lev in LEVELS:
    L = 100 * lev
    summary["naive_ever_%d_from_first_look" % L] = int(P["naive_first_%d" % L].notna().sum())
    summary["naive_final_%d" % L] = int((P["naive_final"] >= lev).sum())
    summary["anytime_ever_%d" % L] = int(P["anytime_first_%d" % L].notna().sum())
    summary["anytime_two_ever_%d" % L] = int(P["anytime_two_first_%d" % L].notna().sum())
    summary["anytime_final_%d" % L] = int((P["C_one_final"] >= lev).sum())
print(json.dumps(summary, indent=1))
print(P[["n", "mean_bp", "t", "naive_final", "C_one_final", "naive_first_90", "anytime_first_90", "anytime_first_95"]].round(3).to_string())

# sensitivity to the prior scale, and the equal-weight mixture over scales (an average of e-processes is an e-process)
sens = {}
for lab, scales in (("a=0.25", (0.25,)), ("a=0.3", (0.3,)), ("a=0.5", (0.5,)), ("a=1", (1.0,)), ("mixture 0.25/0.5/1", (0.25, 0.5, 1.0))):
    ev90 = []; ev95 = []
    for f in mat.columns:
        x = mat[f].dropna().values
        C = confidence_path(e_process(x, a=scales)[0])
        if np.nanmax(C) >= 0.90: ev90.append(f)
        if np.nanmax(C) >= 0.95: ev95.append(f)
    sens[lab] = {"ever_90": ev90, "n_ever_90": len(ev90), "n_ever_95": len(ev95)}
    print("sensitivity", lab, len(ev90), ev90, "| 95%:", len(ev95))
summary["sensitivity"] = sens

# ------------------------------------------------------------------ 2. null simulation: validity and the martingale property
rng = np.random.default_rng(args.seed)
sim = {}
for T in ((25, 50) if not args.skip_sim else ()):
    nsim = args.nsim
    X = rng.standard_normal((nsim, T))
    ever = {lev: 0 for lev in LEVELS}; ever_naive = {lev: 0 for lev in LEVELS}
    EM = np.zeros(T)
    for i in range(nsim):
        one, _ = e_process(X[i], a=A_PRIOR)
        C = confidence_path(one); nv = naive_path(X[i])
        EM += np.nan_to_num(one, nan=1.0)
        for lev in LEVELS:
            ever[lev] += np.nanmax(C) >= lev
            ever_naive[lev] += np.nanmax(nv[args.first_look - 1:]) >= lev
    sim["T%d" % T] = {"P_anytime_ever_90": ever[0.90] / nsim, "P_anytime_ever_95": ever[0.95] / nsim,
                      "P_naive_ever_90_from_first_look": ever_naive[0.90] / nsim, "P_naive_ever_95_from_first_look": ever_naive[0.95] / nsim,
                      "E_M_n_mean_over_n": float(np.mean(EM[1:] / nsim)), "E_M_n_at_T": float(EM[-1] / nsim), "nsim": nsim}
    print("null", T, sim["T%d" % T])

# ------------------------------------------------------------------ 3. power for a constant information ratio
power = {}
for ir in ((0.167, 0.333, 0.5) if not args.skip_sim else ()):
    nsim = args.nsim // 3; T = 50
    X = rng.standard_normal((nsim, T)) + ir
    first = []; by = {10: 0, 16: 0, 25: 0, 50: 0}
    for i in range(nsim):
        one, _ = e_process(X[i], a=A_PRIOR); C = confidence_path(one)
        k = first_cross(C, 0.90)
        first.append(k + 1 if k is not None else np.nan)
        for h in by:
            by[h] += (k is not None) and (k + 1 <= h)
    first = np.array(first)
    power["IR_%.3f" % ir] = {"P_reach_90_by_year": {str(h): v / nsim for h, v in by.items()},
                             "median_year_to_90": float(np.nanmedian(first)) if np.isfinite(first).any() else None,
                             "naive_years_to_90_at_constant_gap": float((stats.norm.ppf(0.9) / ir) ** 2)}
    print("power", ir, power["IR_%.3f" % ir])

# ------------------------------------------------------------------ 4. year-cluster null of the count of anytime-valid crossings
M = mat.values
dem = M - np.nanmean(M, axis=0)
counts = []
for b in range(args.B):
    flips = rng.choice([-1.0, 1.0], size=(M.shape[0], 1))
    Xb = dem * flips
    c = 0
    for j in range(M.shape[1]):
        x = Xb[:, j]; x = x[~np.isnan(x)]
        if len(x) < 2:
            continue
        one, _ = e_process(x, a=A_PRIOR)
        c += np.nanmax(confidence_path(one)) >= 0.90
    counts.append(c)
counts = np.array(counts)
obs = summary["anytime_ever_90"]
null_count = {"mean": float(counts.mean()), "sd": float(counts.std()), "p95": float(np.percentile(counts, 95)),
              "observed": int(obs), "P_ge_observed": float((counts >= obs).mean()), "B": args.B}
print("year-cluster null of the count of anytime crossings:", null_count)

out = args.out or str(ROOT / "results" / ("anytime_valid_%s.json" % ("clean" if args.clean else "extended")))
json.dump({"summary": summary, "null_simulation": sim, "power": power, "year_cluster_null_count": null_count,
           "per_plan": P.round(4).replace({np.nan: None}).to_dict(orient="index")}, open(out, "w"), indent=1, default=float)
P.to_csv(out.replace(".json", ".csv"))
pd.concat(paths, names=["fund"]).reset_index(level=1, drop=True).to_csv(out.replace(".json", "_paths.csv"))
print("wrote", out)
