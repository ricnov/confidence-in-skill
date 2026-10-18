"""Selection, beta and timing: Treynor-Mazuy and Henriksson-Merton regressions inside the framework.

    python scripts/run_timing_models.py data/panel_returns_extended.csv [--B 20000] [--out results/timing_models_extended.json]
    python scripts/run_timing_models.py data/panel_returns.csv --clean --out results/timing_models_clean.json

The framework's statistic is the t-statistic of d_t = log(1+r_P) - log(1+r_L), which treats the plan as the
benchmark plus skill (unit beta, no timing). Three nested regressions say what the mean of d_t contains:

  J   (Jensen, benchmark-relative)   d_t = a + (b-1) l_t + e_t                     l_t = log(1+r_L)
  TM  (Treynor-Mazuy 1966)           d_t = a + (b-1) l_t + g l_t^2 + e_t
  HM  (Henriksson-Merton 1981)       d_t = a + (b-1) l_t + g max(0, -l_t) + e_t

so that, exactly by least squares, mean(d) = a + (b-1) mean(l) + g mean(l^2)  [TM]  or  + g mean(max(0,-l)) [HM].
The first term is selection, the second a static beta mismatch times the benchmark's average return (a
spanned return, not skill), the third timing (a convex payoff in the benchmark, which an option position
would also produce; Glosten and Jagannathan 1994). Confidence in each is Phi of its t-statistic.

No risk-free rate enters: with annual data and b close to one, the term (1-b) r_f is a few basis points and
is absorbed in a. Per-plan standard errors are classical OLS. Panel-level inference uses the restricted
year-cluster wild bootstrap of the paper: fit the model under the null, flip the sign of each calendar
year's residual cross-section with one Rademacher sign, re-estimate, and compare the panel functionals
(sum of squared t, largest |t|, count above a threshold) with their null distributions.

Pre-specified (18 October 2026, before the numbers were seen):
  H1 beta: dispersion of b_i across plans against the null b_i = 1 (restricted model d = a_i + e).
  H2 timing: pooled and per-plan g (TM and HM) against the null g_i = 0 (restricted model J); the number of
     benchmark-down years per plan, which is what identifies HM.
  H3 decomposition: what share of the cross-plan variance of mean(d) is the beta term; the rank correlation of
     the raw t with t_a; the count above 90% on Phi(t_a) against the null a_i = 0 (restricted model with b, g free
     and no intercept); the decomposition for the programmes that survive the paper's corrections.
  H4 the common component: year-mean d regressed on year-mean l and its square (and the HM form).
  H5 power: the median standard errors of b-1, g_TM, g_HM at the panel's sample sizes.
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
from cis.panel import load_panel, clean_panel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("panel", nargs="?", default=str(ROOT / "data" / "panel_returns_extended.csv"))
ap.add_argument("--clean", action="store_true")
ap.add_argument("--B", type=int, default=10000, help="draws for the pooled wild-cluster p-values")
ap.add_argument("--Bpanel", type=int, default=5000, help="draws for the panel-level restricted bootstraps")
ap.add_argument("--rf", type=float, default=0.01, help="cash rate used only in the sensitivity of the beta term")
ap.add_argument("--seed", type=int, default=20261011)
ap.add_argument("--out", default=str(ROOT / "results" / "timing_models.json"))
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

panel, _ = load_panel(args.panel)
if args.clean:
    panel = clean_panel(panel)
panel = panel.sort_values(["fund", "year"]).reset_index(drop=True)
panel["p"] = np.log1p(panel["actual"]); panel["l"] = np.log1p(panel["benchmark"]); panel["d"] = panel["p"] - panel["l"]
funds = sorted(panel["fund"].unique())
years = sorted(panel["year"].unique())
yidx = {y: i for i, y in enumerate(years)}


def ols(y, X):
    """Coefficients, classical standard errors, residuals, fitted."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = float(e @ e) / dof
    se = np.sqrt(np.diag(XtX_inv) * s2)
    return b, se, e, X @ b, dof


def design(l, model):
    one = np.ones_like(l)
    if model == "J":
        return np.column_stack([one, l])
    if model == "TM":
        return np.column_stack([one, l, l ** 2])
    if model == "HM":
        return np.column_stack([one, l, np.maximum(0.0, -l)])
    raise ValueError(model)


def fit_fund(d, l, model):
    X = design(l, model)
    if np.linalg.matrix_rank(X) < X.shape[1]:        # e.g. HM with no benchmark-down year
        nan = float("nan")
        out = dict(a=nan, se_a=nan, t_a=nan, bm1=nan, se_b=nan, t_b=nan, dof=len(d) - X.shape[1], resid_sd=nan, beta_term=nan,
                   g=nan, se_g=nan, t_g=nan, timing_term=nan, mean_d=d.mean(), t_raw=np.sqrt(len(d)) * d.mean() / d.std(ddof=1), p_a=nan)
        return out, np.full_like(d, nan), np.full_like(d, nan)
    b, se, e, fit, dof = ols(d, X)
    out = dict(a=b[0], se_a=se[0], t_a=b[0] / se[0], bm1=b[1], se_b=se[1], t_b=b[1] / se[1], dof=dof, resid_sd=np.sqrt(e @ e / dof))
    out["beta_term"] = b[1] * l.mean()
    if model in ("TM", "HM"):
        z = X[:, 2]
        out.update(g=b[2], se_g=se[2], t_g=b[2] / se[2], timing_term=b[2] * z.mean())
    out["mean_d"] = d.mean()
    out["t_raw"] = np.sqrt(len(d)) * d.mean() / d.std(ddof=1)
    out["p_a"] = stats.t.sf(out["t_a"], dof)
    return out, e, fit


# ------------------------------------------------------------------ per-plan estimates
per = {}
series = {}
for f in funds:
    g = panel[panel["fund"] == f]
    d, l = g["d"].values, g["l"].values
    series[f] = (d, l, g["year"].values)
    row = dict(n=len(d), down_years=int((l < 0).sum()), mean_l=l.mean(), sd_l=l.std(ddof=1), mean_d=d.mean(), sd_d=d.std(ddof=1),
               t_raw=np.sqrt(len(d)) * d.mean() / d.std(ddof=1))
    for m in ("J", "TM", "HM"):
        o, e, fit = fit_fund(d, l, m)
        for k, v in o.items():
            row["%s_%s" % (m, k)] = float(v)
    per[f] = row
P = pd.DataFrame(per).T.astype(float)
P["cis_raw"] = stats.norm.cdf(P["t_raw"])
P["cis_a_J"] = stats.norm.cdf(P["J_t_a"]); P["cis_a_TM"] = stats.norm.cdf(P["TM_t_a"]); P["cis_a_HM"] = stats.norm.cdf(P["HM_t_a"])
P = P.sort_values("t_raw", ascending=False)

# ------------------------------------------------------------------ pooled regressions with plan fixed effects, year-clustered s.e.
def pooled(model):
    rows = []
    for f in funds:
        d, l, yrs = series[f]
        X = design(l, model)[:, 1:]            # slopes only; the intercept is the plan effect (within transformation)
        Xw = X - X.mean(0); dw = d - d.mean()
        for i in range(len(d)):
            rows.append((f, yrs[i], dw[i], *Xw[i]))
    cols = ["fund", "year", "dw"] + ["x%d" % k for k in range(1, 1 + (1 if model == "J" else 2))]
    D = pd.DataFrame(rows, columns=cols)
    X = D[cols[3:]].values; y = D["dw"].values
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    # Liang-Zeger cluster by calendar year
    meat = np.zeros((X.shape[1], X.shape[1]))
    for yv, idx in D.groupby("year").indices.items():
        s = X[idx].T @ e[idx]; meat += np.outer(s, s)
    G = D["year"].nunique(); N = len(y); K = X.shape[1] + len(funds)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    # wild cluster bootstrap p-values (Rademacher by year) under the null that each slope is zero, one at a time
    pv = []
    for j in range(X.shape[1]):
        keep = [k for k in range(X.shape[1]) if k != j]
        if keep:
            Xr = X[:, keep]; br = np.linalg.lstsq(Xr, y, rcond=None)[0]; er = y - Xr @ br; fitr = Xr @ br
        else:
            er = y.copy(); fitr = np.zeros_like(y)
        yrs_arr = D["year"].map(yidx).values
        tb = np.empty(args.B)
        for bb in range(args.B):
            s = rng.choice([-1.0, 1.0], size=len(years))[yrs_arr]
            yb = fitr + s * er
            bbeta = XtX_inv @ X.T @ yb; eb = yb - X @ bbeta
            meat = np.zeros_like(V)
            for yv, idx in D.groupby("year").indices.items():
                sc = X[idx].T @ eb[idx]; meat += np.outer(sc, sc)
            Vb = XtX_inv @ meat @ XtX_inv
            tb[bb] = bbeta[j] / np.sqrt(Vb[j, j])
        t_obs = b[j] / np.sqrt(V[j, j] / ((G / (G - 1)) * ((N - 1) / (N - K))))
        pv.append(float((np.abs(tb) >= abs(t_obs)).mean()))
    names = {"J": ["beta_minus_1"], "TM": ["beta_minus_1", "gamma"], "HM": ["beta_minus_1", "gamma"]}[model]
    return {nm: dict(estimate=float(b[k]), se_cluster=float(se[k]), t_cluster=float(b[k] / se[k]), p_wild=pv[k]) for k, nm in enumerate(names)} | {"clusters": int(G), "obs": int(N)}


S = {"panel": args.panel, "programmes": len(funds), "B": args.B, "seed": args.seed}
S["pooled"] = {m: pooled(m) for m in ("J", "TM", "HM")}
print("pooled:", json.dumps(S["pooled"], indent=1))

# ------------------------------------------------------------------ panel-level nulls by restricted wild bootstrap
def panel_functionals(tvals, thr=1.2816):
    tvals = np.asarray(tvals, dtype=float); tvals = tvals[np.isfinite(tvals)]
    return dict(sum_t2=float(np.sum(tvals ** 2)), max_abs_t=float(np.max(np.abs(tvals))), max_t=float(np.max(tvals)),
                count_above=int(np.sum(tvals > thr)), count_abs_above_196=int(np.sum(np.abs(tvals) > 1.96)))


def restricted_bootstrap(model, param, restricted_design, B):
    """Null distribution of the panel functionals of the t-statistic of `param` in `model`, generating data from
    the restricted model (the model without that parameter) with one sign per calendar year on the residuals."""
    fits = {}
    for f in funds:
        d, l, yrs = series[f]
        Xr = restricted_design(l)
        if Xr.shape[1] == 0:
            fit, e = np.zeros_like(d), d.copy()
        else:
            br = np.linalg.lstsq(Xr, d, rcond=None)[0]; fit = Xr @ br; e = d - fit
        fits[f] = (fit, e, l, np.array([yidx[y] for y in yrs]))
    keys = ["sum_t2", "max_abs_t", "max_t", "count_above", "count_abs_above_196"]
    out = np.empty((B, len(keys)))
    T = np.empty((B, len(funds)))
    for bb in range(B):
        s = rng.choice([-1.0, 1.0], size=len(years))
        tv = []
        for f in funds:
            fit, e, l, yi = fits[f]
            db = fit + s[yi] * e
            o, _, _ = fit_fund(db, l, model)
            tv.append(o[param])
        T[bb] = tv
        fn = panel_functionals(tv)
        out[bb] = [fn[k] for k in keys]
    return pd.DataFrame(out, columns=keys), T


def minp(obs_t, T, upper=True):
    """Westfall-Young min-p: each plan's t is converted to its own null p-value (upper tail, or two-sided), so
    plans with heavy-tailed nulls do not dominate; the adjusted P of the best plan is P(min null p <= observed min p)."""
    T = np.asarray(T, dtype=float); obs = np.asarray(obs_t, dtype=float)
    ok = np.isfinite(obs) & np.all(np.isfinite(T), axis=0)
    T, obs = T[:, ok], obs[ok]
    B = T.shape[0]
    if not upper:
        T, obs = np.abs(T), np.abs(obs)
    p_obs = (T >= obs[None, :]).mean(0)
    ranks = (-T).argsort(0).argsort(0)                    # 0 = largest
    p_null = (ranks + 1) / B
    min_null = p_null.min(1)
    best = int(np.argmin(p_obs))
    return dict(best_plan=str(np.array(funds)[ok][best]), p_best_unadjusted=float(p_obs[best]),
                P_minp=float((min_null <= p_obs[best]).mean()), n_plans=int(ok.sum()))


def summarise(obs_t, null, label):
    fn = panel_functionals(obs_t)
    res = {}
    for k in null.columns:
        o = fn[k]
        res[k] = dict(observed=o, null_mean=float(null[k].mean()), null_95=float(np.percentile(null[k], 95)),
                      P=float((null[k] >= o).mean()))
    print(label, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}, indent=0))
    return res


Bn = args.Bpanel
# H1: beta = 1 everywhere (restricted: d = a_i + e) -> t of (b-1) in J
null_b, Tb = restricted_bootstrap("J", "t_b", lambda l: np.ones((len(l), 1)), Bn)
S["beta_null"] = summarise(P["J_t_b"].values, null_b, "beta: ")
S["beta_null"]["minp_two_sided"] = minp(P.loc[funds, "J_t_b"].values, Tb, upper=False)
# H2: no timing anywhere (restricted: J) -> t of g in TM and HM
null_gTM, TgTM = restricted_bootstrap("TM", "t_g", lambda l: design(l, "J"), Bn)
S["timing_TM_null"] = summarise(P["TM_t_g"].values, null_gTM, "TM gamma: ")
S["timing_TM_null"]["minp_two_sided"] = minp(P.loc[funds, "TM_t_g"].values, TgTM, upper=False)
null_gHM, TgHM = restricted_bootstrap("HM", "t_g", lambda l: design(l, "J"), Bn)
S["timing_HM_null"] = summarise(P["HM_t_g"].values, null_gHM, "HM gamma: ")
S["timing_HM_null"]["minp_two_sided"] = minp(P.loc[funds, "HM_t_g"].values, TgHM, upper=False)
# H3: no selection anywhere given beta free (restricted: d = (b-1) l + e, no intercept) -> t of a in J
null_aJ, TaJ = restricted_bootstrap("J", "t_a", lambda l: l.reshape(-1, 1), Bn)
S["alpha_J_null"] = summarise(P["J_t_a"].values, null_aJ, "alpha (J): ")
S["alpha_J_null"]["minp_upper"] = minp(P.loc[funds, "J_t_a"].values, TaJ, upper=True)
null_aTM, TaTM = restricted_bootstrap("TM", "t_a", lambda l: design(l, "TM")[:, 1:], Bn)
S["alpha_TM_null"] = summarise(P["TM_t_a"].values, null_aTM, "alpha (TM): ")
S["alpha_TM_null"]["minp_upper"] = minp(P.loc[funds, "TM_t_a"].values, TaTM, upper=True)
# and, for comparison, the framework's own statistic under its own null (d = e, unit beta)
null_raw, Traw = restricted_bootstrap("J", "t_raw", lambda l: np.zeros((len(l), 0)), Bn)
S["raw_null"] = summarise(P["t_raw"].values, null_raw, "raw t (unit beta): ")
S["raw_null"]["minp_upper"] = minp(P.loc[funds, "t_raw"].values, Traw, upper=True)
for k in ("beta_null", "timing_TM_null", "timing_HM_null", "alpha_J_null", "alpha_TM_null", "raw_null"):
    print(k, "min-p:", {kk: v for kk, v in S[k].items() if kk.startswith("minp")})

# ------------------------------------------------------------------ H3 decomposition summaries
md = P["mean_d"]; bt = P["J_beta_term"]; aJ = P["J_a"]
S["decomposition"] = {
    "mean_d_bp": round(1e4 * float(md.mean()), 1),
    "mean_alpha_J_bp": round(1e4 * float(aJ.mean()), 1), "mean_beta_term_bp": round(1e4 * float(bt.mean()), 1),
    "mean_alpha_TM_bp": round(1e4 * float(P["TM_a"].mean()), 1), "mean_timing_TM_bp": round(1e4 * float(P["TM_timing_term"].mean()), 1),
    "mean_beta_term_TM_bp": round(1e4 * float(P["TM_beta_term"].mean()), 1),
    "beta_mean": round(float(1 + P["J_bm1"].mean()), 3), "beta_sd": round(float(P["J_bm1"].std(ddof=1)), 3),
    "beta_q10_q90": [round(float(1 + P["J_bm1"].quantile(q)), 3) for q in (0.1, 0.9)],
    "median_se_beta": round(float(P["J_se_b"].median()), 3),
    "gamma_TM_median": round(float(P["TM_g"].median()), 2), "gamma_TM_iqr": [round(float(P["TM_g"].quantile(q)), 2) for q in (0.25, 0.75)],
    "median_se_gamma_TM": round(float(P["TM_se_g"].median()), 2),
    "gamma_HM_median": round(float(P["HM_g"].median()), 2), "median_se_gamma_HM": round(float(P["HM_se_g"].median()), 2),
    "down_years_median": float(P["down_years"].median()), "down_years_min": int(P["down_years"].min()), "down_years_max": int(P["down_years"].max()),
    "var_share_beta_term": round(float(np.cov(bt, md)[0, 1] / md.var(ddof=1)), 3),
    "var_share_alpha_J": round(float(np.cov(aJ, md)[0, 1] / md.var(ddof=1)), 3),
    "spearman_t_raw_t_alpha_J": round(float(stats.spearmanr(P["t_raw"], P["J_t_a"])[0]), 3),
    "spearman_t_raw_t_alpha_TM": round(float(stats.spearmanr(P["t_raw"], P["TM_t_a"])[0]), 3),
    "spearman_beta_t_raw": round(float(stats.spearmanr(P["J_bm1"], P["t_raw"])[0]), 3),
    "spearman_beta_te": round(float(stats.spearmanr(P["J_bm1"].abs(), P["sd_d"])[0]), 3),
    "above_90_raw": int((P["cis_raw"] > .9).sum()), "above_90_alpha_J": int((P["cis_a_J"] > .9).sum()), "above_90_alpha_TM": int((P["cis_a_TM"] > .9).sum()),
    "above_90_alpha_HM": int((P["cis_a_HM"] > .9).sum()),
    "n_t_gamma_TM_above_196": int((P["TM_t_g"].abs() > 1.96).sum()), "n_t_gamma_HM_above_196": int((P["HM_t_g"].abs() > 1.96).sum()),
    "n_t_beta_above_196": int((P["J_t_b"].abs() > 1.96).sum()),
    "se_inflation_alpha_J_median": round(float(np.median(np.sqrt(1 + P["mean_l"] ** 2 / P["sd_l"] ** 2))), 3),
    "se_inflation_alpha_TM_median": round(float(np.median(P["TM_se_a"] / (P["sd_d"] / np.sqrt(P["n"])))), 3),
    "se_inflation_alpha_J_median_actual": round(float(np.median(P["J_se_a"] / (P["sd_d"] / np.sqrt(P["n"])))), 3),
    "corr_l_l2_median": round(float(np.median([np.corrcoef(series[f][1], series[f][1] ** 2)[0, 1] for f in funds])), 3),
    "beta_term_rf_adjusted_mean_bp": round(1e4 * float((P["J_bm1"] * (P["mean_l"] - np.log1p(args.rf))).mean()), 1),
    "var_share_beta_term_rf_adjusted": round(float(np.cov(P["J_bm1"] * (P["mean_l"] - np.log1p(args.rf)), md)[0, 1] / md.var(ddof=1)), 3),
    "beta_true_sd_estimate": round(float(np.sqrt(max(P["J_bm1"].var(ddof=1) - (P["J_se_b"] ** 2).mean(), 0))), 3),
    "sd_mean_d_bp": round(1e4 * float(md.std(ddof=1)), 1),
    "down_year_calendar": {int(y): int(v) for y, v in panel[panel["l"] < 0].groupby("year").size().items()},
}
print("decomposition:", json.dumps(S["decomposition"], indent=1))

# ------------------------------------------------------------------ H4 the common component's shape in the benchmark
ym = panel.groupby("year").agg(d=("d", "mean"), l=("l", "mean"), n=("d", "size"))
ym = ym[ym["n"] >= 10]
y = ym["d"].values; l = ym["l"].values
for m in ("J", "TM", "HM"):
    X = design(l, m); b, se, e, fit, dof = ols(y, X)
    # HC1 standard errors
    XtX_inv = np.linalg.inv(X.T @ X); meat = (X * (e ** 2)[:, None]).T @ X; V = XtX_inv @ meat @ XtX_inv * len(y) / dof
    se_hc = np.sqrt(np.diag(V))
    S["common_component_%s" % m] = {"years": int(len(y)), "coef": [float(v) for v in b], "se_hc1": [float(v) for v in se_hc],
                                     "r2": float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))}
print("common component vs benchmark:", json.dumps({k: v for k, v in S.items() if k.startswith("common_component")}, indent=1))

# ------------------------------------------------------------------ per-plan table
keep = ["n", "down_years", "mean_l", "sd_l", "mean_d", "sd_d", "t_raw", "cis_raw", "J_a", "J_se_a", "J_t_a", "cis_a_J", "J_bm1", "J_se_b", "J_t_b", "J_beta_term",
        "TM_a", "TM_t_a", "cis_a_TM", "TM_bm1", "TM_g", "TM_se_g", "TM_t_g", "TM_timing_term", "TM_beta_term", "HM_a", "HM_t_a", "cis_a_HM", "HM_g", "HM_se_g", "HM_t_g", "HM_timing_term"]
S["per_plan"] = P[keep].round(5).to_dict(orient="index")
pd.set_option("display.width", 250)
show = P[["n", "down_years", "mean_d", "t_raw", "cis_raw", "J_a", "J_t_a", "cis_a_J", "J_bm1", "J_t_b", "J_beta_term", "TM_a", "TM_g", "TM_t_g", "TM_timing_term", "HM_g", "HM_t_g"]].copy()
for c in ["mean_d", "J_a", "J_beta_term", "TM_a", "TM_timing_term"]:
    show[c] = 1e4 * show[c]
print(show.round(2).to_string())
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(S, fh, indent=1, default=float)
P[keep].to_csv(Path(args.out).with_suffix(".csv"))
print("wrote", args.out)
