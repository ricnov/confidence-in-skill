"""Confidence as a forecast: pre-specified tests of the framework's decision rule on the panel.

    python scripts/run_framework_tests.py data/panel_returns_extended.csv [--B 50000] [--out results/framework_tests.json]
    python scripts/run_framework_tests.py data/panel_returns.csv --clean        # the public 42-programme panel

Pre-specification (written before the numbers were seen, 18 October 2026).

Split rule.  Each programme's first ten observed years -- the framework's own minimum horizon, the
first look in the article's expanding windows -- form the "record"; every later year forms the
"sequel". Programmes with fewer than three sequel years are dropped.

A. Forecast.  Rank correlation across programmes between the ten-year confidence (equivalently the
   ten-year t) and the sequel mean log excess return, against the year-cluster sign-flip null
   (one sign per calendar year, common to all programmes, after demeaning each programme).
B. Decision rule.  The trustee's rule "retain at 90%": sequel mean excess return of the programmes
   at or above 90% after ten years minus that of the others (equal-weighted across programmes),
   same null with the groups re-formed in every draw. Also the plans below 50%.
C. Shrinkage.  James-Stein / empirical-Bayes forecasts of the sequel mean from the record, with the
   prior variance tau^2 estimated from the record cross-section alone (method of moments). Mean
   squared forecast error, raw vs shrunk vs zero, in basis points and standardised by the sequel
   standard error. Under the null the raw forecast must lose to zero (it adds the record's noise);
   the sign-flip null gives the reference distribution of the raw/zero ratio.
D. Posterior probability of skill.  With a normal prior N(0, tau^2) on the true mean excess return,
   P(skill > 0 | record) = Phi( t * tau / sqrt(tau^2 + se^2) ): the framework's confidence Phi(t) is
   the flat-prior limit tau -> infinity. Reported for every programme with tau from the full sample.
E. Holdout.  A and B repeated on the nine programmes of the second workbook alone (received after
   the persistence result was found on the first), with the exact sign-flip null on their years.
F. Replication of the lag-2 autocorrelation of excess returns found on the first workbook, on the
   nine alone, sign-flip null.
G. Confidence against tracking error across programmes (rank correlation; null as above).
H. Peer-relative confidence: each programme's excess return minus the leave-one-out cross-plan mean
   of the same calendar label (fiscal and calendar years are not aligned across plans, so this
   understates the common component).
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
ap.add_argument("--clean", action="store_true")
ap.add_argument("--B", type=int, default=50000)
ap.add_argument("--seed", type=int, default=20261011)
ap.add_argument("--record", type=int, default=10)
ap.add_argument("--min-sequel", type=int, default=3)
ap.add_argument("--out", default=str(ROOT / "results" / "framework_tests.json"))
args = ap.parse_args()

panel, _ = load_panel(args.panel)
if args.clean:
    panel = clean_panel(panel)
mat = log_excess_matrix(panel)
NINE = ["IL TRS", "Houston POPS", "INPRS", "Colorado PERA", "Ohio PERS", "SWIB", "Iowa PERS", "CT TRS", "Maine PERS"]
S = {"panel": args.panel, "programmes": int(mat.shape[1]), "record_years": args.record, "B": args.B, "seed": args.seed}
rng = np.random.default_rng(args.seed)


def split_masks(m, record, min_sequel):
    V = m.values
    A = np.zeros_like(V)
    R = np.zeros_like(V)
    keep = []
    for j in range(V.shape[1]):
        ii = np.where(~np.isnan(V[:, j]))[0]
        if len(ii) >= record + min_sequel:
            A[ii[:record], j] = 1.0
            R[ii[record:], j] = 1.0
            keep.append(j)
    return A[:, keep], R[:, keep], keep


def moments(X, W):
    Z = np.where(np.isnan(X), 0.0, X) * W
    n = W.sum(0)
    mean = Z.sum(0) / n
    var = ((Z - mean * W) ** 2).sum(0) / (n - 1)
    return n, mean, var


def split_stats(X, A, R):
    n1, m1, v1 = moments(X, A)
    n2, m2, v2 = moments(X, R)
    t1 = np.sqrt(n1) * m1 / np.sqrt(v1)
    return dict(n1=n1, m1=m1, se1=np.sqrt(v1 / n1), t1=t1, cis1=stats.norm.cdf(t1), n2=n2, m2=m2, se2=np.sqrt(v2 / n2))


def rule_stats(st, hi=0.90, lo=0.50):
    g = st["cis1"] >= hi
    b = st["cis1"] < lo
    out = dict(n_hi=int(g.sum()), n_lo=int(b.sum()))
    out["diff_hi"] = float(st["m2"][g].mean() - st["m2"][~g].mean()) if 0 < g.sum() < len(g) else np.nan
    if 0 < g.sum() < len(g):
        w = 1 / st["se2"] ** 2
        out["pooled_hi"] = float(np.sum(st["m2"][g] * st["n2"][g]) / np.sum(st["n2"][g]))
        out["pooled_rest"] = float(np.sum(st["m2"][~g] * st["n2"][~g]) / np.sum(st["n2"][~g]))
        out["ivw_hi"] = float(np.sum(st["m2"][g] * w[g]) / np.sum(w[g]))
        out["ivw_rest"] = float(np.sum(st["m2"][~g] * w[~g]) / np.sum(w[~g]))
    out["mean_hi"] = float(st["m2"][g].mean()) if g.sum() else np.nan
    out["mean_rest"] = float(st["m2"][~g].mean()) if (~g).sum() else np.nan
    out["mean_lo"] = float(st["m2"][b].mean()) if b.sum() else np.nan
    out["pos_hi"] = int((st["m2"][g] > 0).sum())
    out["pos_lo"] = int((st["m2"][b] > 0).sum())
    return out


def js_forecasts(st):
    m1, se1 = st["m1"], st["se1"]
    tau2_0 = max(float(np.mean(m1 ** 2) - np.mean(se1 ** 2)), 0.0)       # second moment about zero
    tau2_g = max(float(np.var(m1, ddof=1) - np.mean(se1 ** 2)), 0.0)     # about the grand mean
    g = m1.mean()
    w0 = tau2_0 / (tau2_0 + se1 ** 2) if tau2_0 > 0 else np.zeros_like(m1)
    wg = tau2_g / (tau2_g + se1 ** 2) if tau2_g > 0 else np.zeros_like(m1)
    return dict(raw=m1, js0=w0 * m1, jsg=g + wg * (m1 - g), zero=np.zeros_like(m1), grand=np.full_like(m1, g),
                tau0=np.sqrt(tau2_0), taug=np.sqrt(tau2_g), w0=w0, wg=wg)


def losses(F, st):
    m2, se2 = st["m2"], st["se2"]
    return {k: (float(np.mean((F[k] - m2) ** 2)), float(np.mean(((F[k] - m2) / se2) ** 2)))
            for k in ("raw", "js0", "jsg", "zero", "grand")}


def run_block(m, label, B, exact=False):
    A, R, keep = split_masks(m, args.record, args.min_sequel)
    X = m.values[:, keep]
    names = np.array(m.columns)[keep]
    st = split_stats(X, A, R)
    out = {"programmes": int(len(keep)), "names": names.tolist()}
    rho_t = float(stats.spearmanr(st["t1"], st["m2"])[0])
    rho_m = float(stats.spearmanr(st["m1"], st["m2"])[0])
    pear = float(np.corrcoef(st["m1"], st["m2"])[0, 1])
    rule = rule_stats(st)
    F = js_forecasts(st)
    L = losses(F, st)
    # null: one sign per calendar year on the demeaned matrix
    D0 = X - np.nanmean(X, 0)
    ny = X.shape[0]
    if exact and ny <= 16:
        signs = np.array([[1.0 if (k >> i) & 1 else -1.0 for i in range(ny)] for k in range(2 ** ny)])
    else:
        signs = rng.choice([-1.0, 1.0], size=(B, ny))
    nb = signs.shape[0]
    n_rho_t = np.empty(nb); n_rho_m = np.empty(nb); n_diff = np.full(nb, np.nan); n_ratio = np.empty(nb)
    n_ratio_s = np.empty(nb); n_pos_hi = np.full(nb, np.nan); n_nhi = np.empty(nb)
    for b in range(nb):
        Xb = D0 * signs[b][:, None]
        sb = split_stats(Xb, A, R)
        n_rho_t[b] = stats.spearmanr(sb["t1"], sb["m2"])[0]
        n_rho_m[b] = stats.spearmanr(sb["m1"], sb["m2"])[0]
        rb = rule_stats(sb)
        n_diff[b] = rb["diff_hi"]; n_pos_hi[b] = rb["pos_hi"] if rb["n_hi"] else np.nan; n_nhi[b] = rb["n_hi"]
        n_ratio[b] = np.mean((sb["m1"] - sb["m2"]) ** 2) / np.mean(sb["m2"] ** 2)
        n_ratio_s[b] = np.mean(((sb["m1"] - sb["m2"]) / sb["se2"]) ** 2) / np.mean((sb["m2"] / sb["se2"]) ** 2)
    ok = ~np.isnan(n_diff)
    out["forecast"] = {"spearman_t_sequel": round(rho_t, 3), "spearman_P": round(float((n_rho_t >= rho_t).mean()), 4),
                       "spearman_null_95": round(float(np.percentile(n_rho_t, 95)), 3),
                       "spearman_mean_sequel": round(rho_m, 3), "spearman_mean_P": round(float((n_rho_m >= rho_m).mean()), 4),
                       "pearson": round(pear, 3)}
    out["rule"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rule.items()}
    out["rule"]["diff_P"] = round(float((n_diff[ok] >= rule["diff_hi"]).mean()), 4) if not np.isnan(rule["diff_hi"]) else None
    out["rule"]["diff_null_95"] = round(float(np.percentile(n_diff[ok], 95)), 5)
    out["rule"]["n_hi_null_mean"] = round(float(n_nhi.mean()), 2)
    out["rule"]["pos_hi_P"] = round(float(np.nanmean(n_pos_hi >= rule["pos_hi"])), 4) if rule["n_hi"] else None
    out["shrinkage"] = {"tau_record_zero_bp": round(1e4 * F["tau0"], 1), "tau_record_grand_bp": round(1e4 * F["taug"], 1),
                        "grand_mean_record_bp": round(1e4 * float(st["m1"].mean()), 1),
                        "grand_mean_sequel_bp": round(1e4 * float(st["m2"].mean()), 1),
                        "median_shrink_factor_zero": round(float(np.median(F["w0"])), 2),
                        "mse_bp2": {k: round(1e8 * v[0], 1) for k, v in L.items()},
                        "mse_standardised": {k: round(v[1], 3) for k, v in L.items()},
                        "ratio_raw_zero": round(L["raw"][0] / L["zero"][0], 3),
                        "ratio_raw_zero_P": round(float((n_ratio <= L["raw"][0] / L["zero"][0]).mean()), 4),
                        "ratio_raw_zero_null_5": round(float(np.percentile(n_ratio, 5)), 3),
                        "ratio_js0_raw": round(L["js0"][0] / L["raw"][0], 3),
                        "ratio_jsg_raw": round(L["jsg"][0] / L["raw"][0], 3),
                        "ratio_grand_zero": round(L["grand"][0] / L["zero"][0], 3),
                        "ratio_raw_zero_std": round(L["raw"][1] / L["zero"][1], 3),
                        "ratio_raw_zero_std_P": round(float((n_ratio_s <= L["raw"][1] / L["zero"][1]).mean()), 4),
                        "ratio_js0_raw_std": round(L["js0"][1] / L["raw"][1], 3),
                        "share_js0_beats_raw": round(float(np.mean(np.abs(F["js0"] - st["m2"]) < np.abs(F["raw"] - st["m2"]))), 2)}
    # plan-resampling CI for the js/raw ratio
    idx = rng.integers(0, len(keep), size=(10000, len(keep)))
    e_js = (F["js0"] - st["m2"]) ** 2; e_raw = (F["raw"] - st["m2"]) ** 2; e_zero = st["m2"] ** 2
    rr = e_js[idx].mean(1) / e_raw[idx].mean(1); rz = e_raw[idx].mean(1) / e_zero[idx].mean(1)
    out["shrinkage"]["ratio_js0_raw_ci90"] = [round(float(np.percentile(rr, 5)), 3), round(float(np.percentile(rr, 95)), 3)]
    out["shrinkage"]["ratio_raw_zero_ci90"] = [round(float(np.percentile(rz, 5)), 3), round(float(np.percentile(rz, 95)), 3)]
    tab = pd.DataFrame({"n_record": st["n1"], "mean_record_bp": 1e4 * st["m1"], "t_record": st["t1"], "cis_record": 100 * st["cis1"],
                        "shrink_factor": F["w0"], "forecast_js_bp": 1e4 * F["js0"], "n_sequel": st["n2"], "mean_sequel_bp": 1e4 * st["m2"],
                        "se_sequel_bp": 1e4 * st["se2"]}, index=names).sort_values("t_record", ascending=False)
    out["table"] = tab.round(2).to_dict(orient="index")
    print("\n== %s: %d programmes with a %d-year record and >= %d sequel years" % (label, len(keep), args.record, args.min_sequel))
    print(tab.round(2).to_string())
    print("forecast:", out["forecast"])
    print("rule:", out["rule"])
    print("shrinkage:", {k: v for k, v in out["shrinkage"].items() if k != "table"})
    return out


S["all"] = run_block(mat, "all programmes", args.B)
# robustness: the same tests on peer-relative returns (each year's leave-one-out cross-plan mean removed)
_V = mat.values
_cnt = np.sum(~np.isnan(_V), 1, keepdims=True); _tot = np.nansum(_V, 1, keepdims=True)
_loo = (_tot - np.where(np.isnan(_V), 0, _V)) / np.maximum(_cnt - 1, 1); _loo[_cnt[:, 0] < 4] = np.nan
peer = pd.DataFrame(_V - _loo, index=mat.index, columns=mat.columns)
S["all_peer_relative"] = run_block(peer, "all programmes, peer-relative", args.B)
nine = [f for f in NINE if f in mat.columns]
if len(nine) >= 5:
    sub = mat[nine].dropna(how="all")
    S["holdout_nine"] = run_block(sub, "holdout: the nine programmes of the second workbook", args.B)
    old = [f for f in mat.columns if f not in nine]
    S["first_workbook"] = run_block(mat[old].dropna(how="all"), "first-workbook programmes only", args.B)

# ---------------------------------------------------------------- C'. the common component of excess returns
V = mat.values
_names = np.array(mat.columns)
_n = np.sum(~np.isnan(V), 0); _sd = np.nanstd(V, 0, ddof=1); _mu = np.nanmean(V, 0); _t = _mu / (_sd / np.sqrt(_n))
cnt = np.sum(~np.isnan(V), 1, keepdims=True); tot = np.nansum(V, 1, keepdims=True)
loo = (tot - np.where(np.isnan(V), 0, V)) / np.maximum(cnt - 1, 1)
loo[cnt[:, 0] < 4] = np.nan
lam = np.full(len(_names), np.nan); r2 = lam.copy(); alpha = lam.copy()
for j in range(len(_names)):
    ok = ~np.isnan(V[:, j]) & ~np.isnan(loo[:, j])
    sl, ic, r, _, _ = stats.linregress(loo[ok, j], V[ok, j])
    lam[j], r2[j], alpha[j] = sl, r * r, ic
ym = pd.Series(np.nanmean(V, 1), index=mat.index)[cnt[:, 0] >= 10]
D0 = V - np.nanmean(V, 0)
C = pd.DataFrame(D0, columns=_names).corr(min_periods=8).values
pair = C[np.triu_indices(len(_names), 1)]
q = np.percentile(_sd, [25, 75]); lo_, hi_ = _sd <= q[0], _sd >= q[1]
S["common_component"] = {
    "year_means_bp": {int(y): round(1e4 * float(v), 0) for y, v in ym.items()},
    "sd_year_means_bp": round(1e4 * float(ym.std()), 1), "mean_year_means_bp": round(1e4 * float(ym.mean()), 1),
    "mean_2010_2025_bp": round(1e4 * float(ym.loc[2010:2025].mean()), 1), "mean_2020_2025_bp": round(1e4 * float(ym.loc[2020:2025].mean()), 1),
    "autocorr_year_means": {("lag%d" % L): round(float(ym.autocorr(L)), 2) for L in (1, 2, 3)},
    "share_variance_year_means": round(float(np.nanvar(np.repeat(np.nanmean(V, 1)[:, None], V.shape[1], 1)[~np.isnan(V)]) / np.nanvar(D0)), 3),
    "mean_pairwise_correlation": round(float(np.nanmean(pair)), 3),
    "spearman_r2_te": round(float(stats.spearmanr(r2, _sd)[0]), 2), "spearman_r2_t": round(float(stats.spearmanr(r2, _t)[0]), 2),
    "spearman_r2_abs_t": round(float(stats.spearmanr(r2, np.abs(_t))[0]), 2),
    "spearman_loading_te": round(float(stats.spearmanr(lam, _sd)[0]), 2), "spearman_loading_t": round(float(stats.spearmanr(lam, _t)[0]), 2),
    "median_r2_low_te_quartile": round(float(np.median(r2[lo_])), 2), "median_r2_high_te_quartile": round(float(np.median(r2[hi_])), 2),
    "median_loading": round(float(np.median(lam)), 2), "loading_iqr": [round(float(x), 2) for x in np.percentile(lam, [25, 75])],
    "per_plan": pd.DataFrame({"n": _n, "te_pct": 100 * _sd, "t": _t, "loading": lam, "r2": r2, "alpha_bp": 1e4 * alpha, "mean_bp": 1e4 * _mu},
                             index=_names).round(2).sort_values("t", ascending=False).to_dict(orient="index")}
print("\n== common component:", {k: v for k, v in S["common_component"].items() if k not in ("per_plan", "year_means_bp")})

# ---------------------------------------------------------------- D. posterior probability of skill, full sample
n = np.sum(~np.isnan(V), 0); mu = np.nanmean(V, 0); sd = np.nanstd(V, 0, ddof=1); se = sd / np.sqrt(n); t = mu / se
cis = stats.norm.cdf(t)
tau2_mom = max(float(np.mean(mu ** 2) - np.mean(se ** 2)), 0.0)
names = np.array(mat.columns)
ext = [f for f in ("SWIB", "DAL F&P") if f in names]
mask = ~np.isin(names, ext)
tau2_mom_x = max(float(np.mean(mu[mask] ** 2) - np.mean(se[mask] ** 2)), 0.0)


def posterior(tau):
    return stats.norm.cdf(t * tau / np.sqrt(tau ** 2 + se ** 2))


post = {}
for lab, tau in (("tau_mom_all", np.sqrt(tau2_mom)), ("tau_mom_without_extremes", np.sqrt(tau2_mom_x)), ("tau_30bp", 0.0030), ("tau_60bp", 0.0060)):
    p = posterior(tau)
    post[lab] = {"tau_bp": round(1e4 * tau, 1), "above_90": int((p > 0.9).sum()), "above_95": int((p > 0.95).sum()),
                 "median_prob_of_cis_90_plan": None,
                 "median_posterior": round(float(np.median(p)), 3),
                 "swib": round(float(p[names == "SWIB"][0]), 4) if "SWIB" in names else None,
                 "nyslrs": round(float(p[names == "NYSLRS"][0]), 4) if "NYSLRS" in names else None}
    # what a 90% confidence is worth at the median tracking error and sample size, and at the 90th percentile of TE
    for q, lab2 in ((50, "median_te"), (90, "te_p90"), (10, "te_p10")):
        se_q = np.percentile(sd, q) / np.sqrt(np.median(n))
        post[lab][lab2 + "_posterior_at_cis90"] = round(float(stats.norm.cdf(1.2816 * tau / np.sqrt(tau ** 2 + se_q ** 2))), 3)
S["posterior"] = {"cis_above_90": int((cis > 0.9).sum()), "cis_above_95": int((cis > 0.95).sum()), **post}
S["posterior"]["per_plan"] = pd.DataFrame({"n": n, "mean_bp": 1e4 * mu, "te_pct": 100 * sd, "t": t, "cis": 100 * cis,
                                           "posterior_mom_all": 100 * posterior(np.sqrt(tau2_mom)),
                                           "posterior_mom_x": 100 * posterior(np.sqrt(tau2_mom_x))}, index=names).round(2).sort_values("t", ascending=False).to_dict(orient="index")
print("\n== posterior probability of skill:", {k: v for k, v in S["posterior"].items() if k != "per_plan"})

# ---------------------------------------------------------------- F. lag-2 autocorrelation on the nine
def pooled_autocorr(m, lag):
    V = m.values; D = V - np.nanmean(V, 0)
    num = np.nansum(D[lag:] * D[:-lag]); den = np.nansum(D ** 2)
    return num / den


if len(nine) >= 5:
    sub = mat[nine].dropna(how="all")
    D0 = sub.values - np.nanmean(sub.values, 0)
    res = {}
    for lag in (1, 2):
        obs = pooled_autocorr(sub, lag)
        nulls = np.empty(args.B)
        for b in range(args.B):
            s = rng.choice([-1.0, 1.0], size=(sub.shape[0], 1))
            Db = D0 * s
            nulls[b] = np.nansum(Db[lag:] * Db[:-lag]) / np.nansum(Db ** 2)
        Tbar = float(np.mean(np.sum(~np.isnan(sub.values), 0)))
        res["lag%d" % lag] = {"observed": round(float(obs), 3), "bias_corrected": round(float(obs + 1 / (Tbar - 1)), 3),
                              "P_two_sided": round(float((np.abs(nulls) >= abs(obs)).mean()), 4), "null_sd": round(float(nulls.std()), 3)}
    S["autocorrelation_nine"] = res
    print("\n== autocorrelation on the nine:", res)

# ---------------------------------------------------------------- G. confidence against tracking error
rho_te = float(stats.spearmanr(cis, sd)[0]); rho_gap = float(stats.spearmanr(mu, sd)[0])
D0 = V - np.nanmean(V, 0)
nulls = np.empty(args.B)
for b in range(args.B):
    s = rng.choice([-1.0, 1.0], size=(V.shape[0], 1))
    Xb = D0 * s
    mb = np.nanmean(Xb, 0); sb = np.nanstd(Xb, 0, ddof=1)
    nulls[b] = stats.spearmanr(mb / sb, sb)[0]
q = np.percentile(sd, [25, 75])
lo, hi = sd <= q[0], sd >= q[1]
S["confidence_vs_te"] = {"spearman_cis_te": round(rho_te, 3), "P_two_sided": round(float((np.abs(nulls) >= abs(rho_te)).mean()), 4),
                         "spearman_gap_te": round(rho_gap, 3),
                         "low_te_quartile": {"n": int(lo.sum()), "mean_gap_bp": round(1e4 * float(mu[lo].mean()), 1), "above_90": int((cis[lo] > .9).sum()), "mean_te_pct": round(100 * float(sd[lo].mean()), 2)},
                         "high_te_quartile": {"n": int(hi.sum()), "mean_gap_bp": round(1e4 * float(mu[hi].mean()), 1), "above_90": int((cis[hi] > .9).sum()), "mean_te_pct": round(100 * float(sd[hi].mean()), 2)}}
print("\n== confidence vs tracking error:", S["confidence_vs_te"])

# ---------------------------------------------------------------- H. peer-relative confidence (leave-one-out year mean)
cnt = np.sum(~np.isnan(V), 1, keepdims=True); tot = np.nansum(V, 1, keepdims=True)
loo = (tot - np.where(np.isnan(V), 0, V)) / np.maximum(cnt - 1, 1)
loo[cnt[:, 0] < 4] = np.nan
P = V - loo
np_ = np.sum(~np.isnan(P), 0); mup = np.nanmean(P, 0); sdp = np.nanstd(P, 0, ddof=1); tp = np.sqrt(np_) * mup / sdp
cisp = stats.norm.cdf(tp)
year_mean = np.nanmean(V, 1)
S["peer_relative"] = {"above_90_raw": int((cis > .9).sum()), "above_90_peer": int((cisp > .9).sum()),
                      "spearman_raw_peer_t": round(float(stats.spearmanr(t, tp)[0]), 3),
                      "year_means_bp": {int(y): round(1e4 * float(v), 0) for y, v in zip(mat.index, year_mean) if cnt[list(mat.index).index(y), 0] >= 10},
                      "share_variance_common": round(float(np.nanvar(np.repeat(year_mean[:, None], V.shape[1], 1)[~np.isnan(V)]) / np.nanvar(D0)), 3),
                      "per_plan": pd.DataFrame({"t": t, "t_peer": tp, "cis": 100 * cis, "cis_peer": 100 * cisp}, index=names).round(2).sort_values("t", ascending=False).to_dict(orient="index")}
print("\n== peer-relative:", {k: v for k, v in S["peer_relative"].items() if k != "per_plan"})

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump(S, f, indent=1, default=float)
print("\nwrote", args.out)
