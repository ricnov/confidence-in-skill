"""Corrected versions of the article's exhibits, for the journal's online update.

    python scripts/make_corrected_exhibits.py [data/panel_returns.csv] --out <folder>
           [--lasers <LASERS_data.xlsx>] [--extra <second workbook .xlsx>] [--published data/appendix_III_published.csv]

What the correction is (errata item B1). Equation (9b)'s numerator (mu_P - sigma_P^2/2) - (mu_L - sigma_L^2/2)
takes arithmetic drifts; the calculation workbook put annualised geometric returns in it, so the Ito
adjustment was applied twice. The corrected statistic keeps the article's method and replaces the
geometric returns by arithmetic mean returns ("arithmetic numerator"). The model-consistent statistic
(the one-sample t of the log excess return, "t") is also reported in every table, as the alternative.

Exhibits produced (each sheet of the workbook, plus PNG/PDF figures):
  Table1        CalSTRS 1999-2023 inputs (one row), columns as printed, with the corrected K and CIS
  AppendixI     CalSTRS expanding-window table 1999-2025, columns as printed, corrected K and CIS from year 10
  AppendixII    CalSTRS ten-year rolling-window CIS, as published and corrected
  AppendixIII   the 47 funds (+ Maine PERS when the second workbook is given), columns as printed, corrected
                CIS and category, with the published and model-consistent values alongside and a note column
  Fig1_CalSTRS  the series of Figure 1 (IR, tracking error, correlation, CIS), published method and corrected
  Fig2_CalSTRS  CIS by number of years of data (Figure 2's CalSTRS line), published method and corrected
  Fig4_paths    corrected CIS paths of the eight plans of Figure 4 (from the tenth year)
  Fig5_paths    corrected CIS paths of the seven California plans of Figure 5 (from the tenth year) + inset
  LASERS        the author's reconstructed LASERS series (policy benchmark from target weights), if given
Categories: High >= 80%, Low < 50%, Mod otherwise (the article's Appendix III classification; its Figure 5 draws the 80% and 50% lines).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cis.panel import load_panel, DATA_PROBLEMS  # noqa: E402
from cis.core import skill_from_returns, tracking_error  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("panel", nargs="?", default=str(ROOT / "data" / "panel_returns.csv"))
ap.add_argument("--published", default=str(ROOT / "data" / "appendix_III_published.csv"))
ap.add_argument("--lasers", default=None)
ap.add_argument("--extra", default=None)
ap.add_argument("--out", default=str(ROOT / "results" / "corrected_exhibits"))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

panel, _ = load_panel(args.panel)
pub = pd.read_csv(args.published)


def category(c):
    """The article's Appendix III categories: High at 80% and above, Low below 50%, Mod between (the 80% and 50%
    thresholds are the lines drawn in its Figure 5)."""
    return "High" if round(100 * c) >= 80 else ("Low" if round(100 * c) < 50 else "Mod")


def window_table(d, min_k=1):
    """The article's Appendix I columns on expanding windows of one fund's annual returns, plus corrected ones."""
    rP, rL, yrs = d["actual"].values, d["benchmark"].values, d["year"].values
    rows = []
    for k in range(1, len(rP) + 1):
        p, l = rP[:k], rL[:k]
        cumP, cumL = np.prod(1 + p), np.prod(1 + l)
        gP, gL = cumP ** (1 / k) - 1, cumL ** (1 / k) - 1
        row = dict(fy_year=int(yrs[k - 1]), time=k, actual=p[-1], cumulative_actual=cumP, annualised_actual=gP,
                   benchmark=l[-1], cumulative_benchmark=cumL, annualised_benchmark=gL, excess=p[-1] - l[-1],
                   annualised_excess=gP - gL, mean_actual=p.mean(), mean_benchmark=l.mean(), mean_excess=(p - l).mean())
        if k >= 3:
            s = skill_from_returns(p, l)
            row.update(ir_published=(gP - gL) / s["tracking_error"], vol_portfolio=s["sigma_p"], vol_benchmark=s["sigma_l"],
                       tracking_error=s["tracking_error"], correlation=s["rho"], k_published=s["k_paper"], cis_published=s["cis_paper"],
                       ir_corrected=(p - l).mean() / s["tracking_error"], k_corrected=s["k_arith"], cis_corrected=s["cis_arith"],
                       t_model=s["t"], cis_model=s["cis_star"])
        rows.append(row)
    return pd.DataFrame(rows)


def fund(name):
    return panel[panel["fund"] == name].sort_values("year")


# ------------------------------------------------------------------ CalSTRS: Table 1, Appendix I, II, Figures 1-2
cs = fund("CalSTRS")
W = window_table(cs)
appI_cols = ["fy_year", "time", "actual", "cumulative_actual", "annualised_actual", "benchmark", "cumulative_benchmark",
             "annualised_benchmark", "excess", "annualised_excess", "mean_actual", "mean_benchmark", "mean_excess",
             "ir_published", "ir_corrected", "vol_portfolio", "vol_benchmark", "tracking_error", "correlation",
             "k_published", "cis_published", "k_corrected", "cis_corrected", "t_model", "cis_model"]
appI = W[appI_cols].copy()
for c in ["ir_published", "ir_corrected", "vol_portfolio", "vol_benchmark", "tracking_error", "correlation", "k_published",
          "cis_published", "k_corrected", "cis_corrected", "t_model", "cis_model"]:
    appI.loc[appI["time"] < 10, c] = np.nan          # the article reports K and CIS from the tenth year
table1 = appI[appI["fy_year"] == 2023].copy()
assert abs(float(table1["k_published"].iloc[0]) - 0.3745) < 5e-4, table1["k_published"]
# Appendix II: ten-year rolling window
roll = []
rP, rL, yrs = cs["actual"].values, cs["benchmark"].values, cs["year"].values
for i in range(9, len(rP)):
    s = skill_from_returns(rP[i - 9:i + 1], rL[i - 9:i + 1])
    roll.append(dict(fy_year=int(yrs[i]), window="%d-%d" % (yrs[i - 9], yrs[i]), tracking_error=s["tracking_error"],
                     k_published=s["k_paper"], cis_published=s["cis_paper"], k_corrected=s["k_arith"], cis_corrected=s["cis_arith"],
                     t_model=s["t"], cis_model=s["cis_star"]))
appII = pd.DataFrame(roll)
fig1 = appI[appI["time"] >= 10][["fy_year", "ir_published", "ir_corrected", "tracking_error", "correlation", "cis_published", "cis_corrected", "cis_model"]]
fig2 = W[W["time"] >= 3][["time", "fy_year", "cis_published", "cis_corrected", "cis_model"]]

# ------------------------------------------------------------------ Appendix III
rows = []
for _, r in pub.iterrows():
    d = fund(r["fund"])
    s = skill_from_returns(d["actual"].values, d["benchmark"].values)
    note = DATA_PROBLEMS.get(r["fund"], "")
    rows.append(dict(fund=r["fund"], start_published=int(r["start"]), years_published=int(r["years"]),
                     start_workbook=int(d["year"].min()), end_workbook=int(d["year"].max()), years_workbook=len(d),
                     ann_return_published=r["ann_return"], ann_return=s["g_p"], excess_published=r["excess"], ann_excess=s["g_p"] - s["g_l"],
                     mean_excess=float((d["actual"] - d["benchmark"]).mean()),
                     tracking_error_published=r["tracking_error"], tracking_error=s["tracking_error"],
                     cis_published=r["cis"], cis_geometric_numerator=s["cis_paper"], k_corrected=s["k_arith"], cis_corrected=s["cis_arith"],
                     category_published=category(r["cis"]), category_corrected=category(s["cis_arith"]),
                     t_model=s["t"], cis_model=s["cis_star"], category_model=category(s["cis_star"]), note=note))
appIII = pd.DataFrame(rows)
if args.extra:
    from cis.panel_extra import load_extra_workbook
    ex = load_extra_workbook(args.extra)
    m = ex[(ex["fund"] == "Maine PERS") & (ex["year"] != 2020)].sort_values("year")
    s = skill_from_returns(m["actual"].values, m["benchmark"].values)
    appIII.loc[len(appIII)] = dict(fund="Maine PERS", start_published=np.nan, years_published=np.nan, start_workbook=int(m["year"].min()),
                                   end_workbook=int(m["year"].max()), years_workbook=len(m), ann_return_published=np.nan, ann_return=s["g_p"],
                                   excess_published=np.nan, ann_excess=s["g_p"] - s["g_l"], mean_excess=float((m["actual"] - m["benchmark"]).mean()),
                                   tracking_error_published=np.nan, tracking_error=s["tracking_error"], cis_published=np.nan,
                                   cis_geometric_numerator=s["cis_paper"], k_corrected=s["k_arith"], cis_corrected=s["cis_arith"],
                                   category_published="", category_corrected=category(s["cis_arith"]), t_model=s["t"], cis_model=s["cis_star"],
                                   category_model=category(s["cis_star"]),
                                   note="48th plan named in the article; from the second workbook (fiscal 2004-2023; fiscal 2020 omitted, its benchmark is recorded as 0.0)")
EXTRA_NOTES = {
    "COAERS": "last observation (2026) is a part-year return, per the sheet name (errata C9)",
    "IL IMRF": "last observation (2026) is a part-year return, per the sheet name (errata C9)",
    "DCRB": "last observation (2026) is a part-year return, per the sheet name (errata C9)",
    "SWIB": "sheet already used arithmetic means (terminal 98.6%); the printed 93% matches neither the sheet nor the "
            "Wisconsin Core Fund series of the second workbook, which is the correct one and gives 100% (errata C1)",
    "AZSRS": "sheet already used arithmetic means; corrected value equals the sheet's",
    "NC TERS": "sheet already used arithmetic means; corrected value equals the sheet's",
    "MONT COUNTY ERS": "sheet's terminal value reproduces neither numerator exactly (different window or cell reference)",
    "TNCRS": "sheet's terminal value reproduces neither numerator exactly (different window or cell reference)",
    "VCERA": "sheet's terminal value reproduces neither numerator exactly (different window or cell reference)",
}
for f, txt in EXTRA_NOTES.items():
    i = appIII.index[appIII["fund"] == f]
    appIII.loc[i, "note"] = (appIII.loc[i, "note"] + "; " if appIII.loc[i, "note"].iloc[0] else "") + txt
lasers = None
if args.lasers:
    from openpyxl import load_workbook
    wb = load_workbook(args.lasers, read_only=True, data_only=True)
    recs = [(int(r[0]), float(r[1]), float(r[2])) for r in wb["Sheet1"].iter_rows(values_only=True)
            if isinstance(r[0], (int, float)) and isinstance(r[1], (int, float)) and isinstance(r[2], (int, float))]
    lasers = pd.DataFrame(recs, columns=["fy_year", "actual", "benchmark"]).sort_values("fy_year")
    s = skill_from_returns(lasers["actual"].values, lasers["benchmark"].values)
    i = appIII.index[appIII["fund"] == "LASERS"]
    appIII.loc[i, ["start_workbook", "end_workbook", "years_workbook", "ann_return", "ann_excess", "mean_excess", "tracking_error",
                   "cis_geometric_numerator", "k_corrected", "cis_corrected", "category_corrected", "t_model", "cis_model", "category_model"]] = \
        [int(lasers.fy_year.min()), int(lasers.fy_year.max()), len(lasers), s["g_p"], s["g_p"] - s["g_l"],
         float((lasers["actual"] - lasers["benchmark"]).mean()), s["tracking_error"], s["cis_paper"], s["k_arith"], s["cis_arith"],
         category(s["cis_arith"]), s["t"], s["cis_star"], category(s["cis_star"])]
    appIII.loc[i, "note"] = ("published row used the S&P 500 as benchmark; this row uses the author's reconstruction of October 2026 "
                             "(total-plan return against a policy benchmark built from target weights and asset-class indices, fiscal 2013-2025)")
    lasers_stats = s
appIII = appIII.sort_values("cis_corrected", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------------ Figures 4 and 5: paths
FIG4 = ["MDSRA", "LACERA", "KPERS", "SWIB", "TNCRS", "UT RS", "CalPERS", "TX TRS"]
FIG5 = ["CalPERS", "CalSTRS", "LACERA", "LAFPP", "OCERS", "SBCERA", "VCERA"]


def paths(names, min_k=10):
    out = {}
    for f in names:
        W_ = window_table(fund(f))
        W_ = W_[W_["time"] >= min_k]
        out[f] = W_.set_index("fy_year")[["cis_published", "cis_corrected", "cis_model"]]
    return out


p4, p5 = paths(FIG4), paths(FIG5)


def wide(p, col):
    return pd.concat({f: v[col] for f, v in p.items()}, axis=1).sort_index()


fig4 = wide(p4, "cis_corrected"); fig4_pub = wide(p4, "cis_published")
fig5 = wide(p5, "cis_corrected"); fig5_pub = wide(p5, "cis_published")
inset = appIII.set_index("fund").loc[FIG5, ["ann_return", "ann_excess", "tracking_error", "cis_published", "cis_corrected", "cis_model"]].sort_values("cis_corrected", ascending=False)

# ------------------------------------------------------------------ draw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRAY = "#0b0b0b", "#52514e", "#c9ccd1"
PAL = ["#2a78d6", "#e3703a", "#2f9e5b", "#8e44ad", "#c9a227", "#1b998b", "#d6336c", "#6c757d"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def pct_axis(ax):
    ax.set_ylim(0, 1.02); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])


# Figure 1: IR and TE (left axis), correlation and CIS (right axis)
f1 = fig1.set_index("fy_year")
fig, ax = plt.subplots(figsize=(7.0, 3.4)); ax2 = ax.twinx()
ax.plot(f1.index, f1["ir_corrected"], color=PAL[2], linewidth=1.6, label="IR (mean excess / tracking error), LHS")
ax.plot(f1.index, 100 * f1["tracking_error"], color=PAL[1], linewidth=1.6, label="Annualized tracking error, % (LHS)")
ax2.plot(f1.index, f1["correlation"], color=PAL[3], linewidth=1.4, label="Correlation (RHS)")
ax2.plot(f1.index, f1["cis_corrected"], color=PAL[0], linewidth=2.2, label="CalSTRS confidence in skill, corrected (RHS)")
ax2.plot(f1.index, f1["cis_published"], color=GRAY, linewidth=1.4, linestyle="--", label="as published (RHS)")
ax.set_ylim(0, 2.5); ax2.set_ylim(0, 1.05)
ax2.set_yticks([0, .2, .4, .6, .8, 1.0]); ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax2.spines["right"].set_visible(True)
ax.set_xticks(list(range(2008, 2026, 2))); ax.set_xlabel("fiscal year (expanding window from 1999)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
ax.set_title("Figure 1 (corrected): CalSTRS time-varying skill analysis, 1999-2025", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(OUT / "Figure1_CalSTRS_corrected.png", dpi=200); fig.savefig(OUT / "Figure1_CalSTRS_corrected.pdf"); plt.close(fig)

# Figure 2: CalSTRS by number of years
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.plot(fig2["time"], fig2["cis_corrected"], color=PAL[0], linewidth=2.2, label="CalSTRS, corrected (arithmetic numerator)")
ax.plot(fig2["time"], fig2["cis_published"], color=GRAY, linewidth=1.4, linestyle="--", label="CalSTRS, as published (geometric numerator)")
ax.plot(fig2["time"], fig2["cis_model"], color=PAL[2], linewidth=1.2, linestyle=":", label="model-consistent t-statistic")
ax.axvline(10, color=GRAY, linewidth=1); pct_axis(ax)
ax.set_xlabel("number of years of data used (fiscal 1999 onwards)"); ax.set_ylabel("confidence in skill")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title("Figure 2 (corrected, CalSTRS line only): confidence in skill by years of data", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(OUT / "Figure2_CalSTRS_corrected.png", dpi=200); fig.savefig(OUT / "Figure2_CalSTRS_corrected.pdf"); plt.close(fig)


def path_figure(wide_df, title, fname, labels):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, f in enumerate(wide_df.columns):
        s = wide_df[f].dropna()
        ax.plot(s.index, s.values, color=PAL[i % len(PAL)], linewidth=1.8, label=labels.get(f, f))
    ax.axhline(0.9, color=GRAY, linewidth=1, linestyle=":"); ax.axhline(0.5, color=GRAY, linewidth=1, linestyle=":")
    pct_axis(ax); ax.set_xlabel("fiscal year (expanding window from each plan's first year; lines start at the tenth year)")
    ax.set_ylabel("confidence in skill, corrected")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / (fname + ".png"), dpi=200); fig.savefig(OUT / (fname + ".pdf")); plt.close(fig)


lab4 = {f: "%s %02d-%02d" % (f, fund(f)["year"].min() % 100, fund(f)["year"].max() % 100) for f in FIG4}
path_figure(fig4, "Figure 4 (corrected): TVS of eight US public plans", "Figure4_corrected", lab4)
path_figure(fig5, "Figure 5 (corrected): TVS of seven California public plans", "Figure5_corrected", {})

# ------------------------------------------------------------------ write the workbook
readme = pd.DataFrame({"sheet": ["Table1", "AppendixI", "AppendixII", "AppendixIII", "Fig1_CalSTRS", "Fig2_CalSTRS", "Fig4_paths", "Fig4_published",
                                 "Fig5_paths", "Fig5_published", "Fig5_inset", "LASERS_reconstruction"],
                       "replaces": ["Table 1 (p. 14)", "Appendix I (p. 21)", "Appendix II (p. 21)", "Appendix III (pp. 22-23)", "Figure 1 (p. 14)",
                                    "Figure 2 (p. 16), CalSTRS line", "Figure 4 (p. 19)", "(check) Figure 4 as published", "Figure 5 (p. 20)",
                                    "(check) Figure 5 as published", "Figure 5 inset table", "basis of the LASERS row"],
                       "what changed": [
                           "K and CIS recomputed with arithmetic mean returns in the numerator (errata B1); correlation is the workbook-consistent value (errata A7); mean_* columns added",
                           "same, for every year; K and CIS from the tenth year as in the article; cis_model = model-consistent alternative",
                           "ten-year rolling windows, published method and corrected",
                           "columns as printed plus corrected K, CIS and category; cis_geometric_numerator is the article's implemented method (reproduces 41 of the 47 sheets); note column flags the data items of errata C1, C5-C9",
                           "IR, tracking error and correlation unchanged except correlation (A7); CIS corrected; published CIS kept for comparison",
                           "CalSTRS line only: the SBCERA and VIMCO lines need the monthly data (and errata A9)",
                           "corrected CIS paths; SWIB is the first workbook's sheet (errata C1: wrong before 2014; the Core Fund series is the correct one)",
                           "", "corrected CIS paths from the tenth year (the published figure starts earlier)", "", "annualised return, excess and tracking error as printed; CIS corrected",
                           "the author's reconstructed series (October 2026) and its statistics"]})
with pd.ExcelWriter(OUT / "JPM_TVS_corrected_exhibits.xlsx", engine="openpyxl") as xw:
    readme.to_excel(xw, sheet_name="README", index=False)
    table1.to_excel(xw, sheet_name="Table1", index=False)
    appI.to_excel(xw, sheet_name="AppendixI", index=False)
    appII.to_excel(xw, sheet_name="AppendixII", index=False)
    appIII.to_excel(xw, sheet_name="AppendixIII", index=False)
    fig1.to_excel(xw, sheet_name="Fig1_CalSTRS", index=False)
    fig2.to_excel(xw, sheet_name="Fig2_CalSTRS", index=False)
    fig4.to_excel(xw, sheet_name="Fig4_paths"); fig4_pub.to_excel(xw, sheet_name="Fig4_published")
    fig5.to_excel(xw, sheet_name="Fig5_paths"); fig5_pub.to_excel(xw, sheet_name="Fig5_published")
    inset.to_excel(xw, sheet_name="Fig5_inset")
    if lasers is not None:
        lasers.to_excel(xw, sheet_name="LASERS_reconstruction", index=False)
appIII.to_csv(OUT / "AppendixIII_corrected.csv", index=False)
appI.to_csv(OUT / "AppendixI_corrected.csv", index=False)

# ------------------------------------------------------------------ console summary
pd.set_option("display.width", 220)
t1 = table1.iloc[0]
print("Table 1 (2023): K published %.4f -> corrected %.4f; CIS %.2f%% -> %.2f%% (model-consistent %.2f%%); TE %.2f%%, rho %.4f" % (
    t1.k_published, t1.k_corrected, 100 * t1.cis_published, 100 * t1.cis_corrected, 100 * t1.cis_model, 100 * t1.tracking_error, t1.correlation))
a25 = appI[appI.fy_year == 2025].iloc[0]
print("Appendix I (2025): CIS %.2f%% -> %.2f%% (model %.2f%%)" % (100 * a25.cis_published, 100 * a25.cis_corrected, 100 * a25.cis_model))
print("Appendix III: above 90%% published %d, corrected %d, model %d; category changes:" % (
    (appIII.cis_published >= .9).sum(), (appIII.cis_corrected >= .9).sum(), (appIII.cis_model >= .9).sum()))
ch = appIII[(appIII.category_published != appIII.category_corrected) & appIII.category_published.ne("")]
print(ch[["fund", "cis_published", "cis_corrected", "category_published", "category_corrected"]].round(3).to_string(index=False))
print("\nFigure 5 inset:\n", inset.round(4).to_string())
if lasers is not None:
    s = lasers_stats
    print("\nLASERS reconstruction: %d years %d-%d; mean excess %.2f%%, TE %.2f%%, K(published method) %.2f CIS %.1f%%; K corrected %.2f CIS %.1f%%; t %.2f CIS* %.1f%%" % (
        len(lasers), lasers.fy_year.min(), lasers.fy_year.max(), 100 * (lasers.actual - lasers.benchmark).mean(), 100 * s["tracking_error"],
        s["k_paper"], 100 * s["cis_paper"], s["k_arith"], 100 * s["cis_arith"], s["t"], 100 * s["cis_star"]))
print("\nwrote", OUT)
