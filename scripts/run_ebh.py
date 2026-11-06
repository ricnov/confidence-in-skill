"""From one plan to the panel: e-BH on the plans' e-processes, year by year.

    python scripts/run_ebh.py [data/panel_returns_extended.csv] [--clean] [--alpha 0.05 0.1 0.2]

Each programme's e-process M_{i,t} (src/cis/anytime.py, mixture of prior scales) is an e-value at any
stopping time. Wang and Ramdas (2022) show that the e-BH procedure -- sort the e-values, reject the k* largest
where k* = max{k : E_(k) >= m / (k alpha)} -- controls the false discovery rate at alpha under ARBITRARY
dependence between the e-values, which the common market years of a panel create; Wang, Dandapanthula and
Ramdas (2025) show that applying it to e-processes at any stopping time (the "stopped e-BH") keeps that
control when, given the current year, the plans' returns are independent of the past (dependence within a
year is unrestricted) -- the exchangeable-years assumption of the paper's year-cluster bootstrap. Without
that condition the running maximum lifted through an adjuster, A(m) = sqrt(m) - 1 (Shafer, Shen, Vereshchagin
and Vovk, 2011), is an e-value under any filtration (Wang, Dandapanthula and Ramdas, 2025, Corollary 4.3);
it is reported alongside. The average of the e-values is an e-value under arbitrary dependence (Vovk and
Wang, 2021) and tests the global null that no plan has skill.

Outputs results/ebh_{extended,clean}.json: the discovery set at the end of the record at each alpha, the year
each discovered plan entered the set, the discovery path year by year, and the average e-value over time.
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
from cis.anytime import e_process  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("panel", nargs="?", default=str(ROOT / "data" / "panel_returns_extended.csv"))
ap.add_argument("--clean", action="store_true")
ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10, 0.20])
ap.add_argument("--out", default=None)
args = ap.parse_args()


def ebh(e, alpha):
    """Indices rejected by e-BH at level alpha; e is a 1-d array of e-values, m = len(e)."""
    e = np.asarray(e, float)
    m = len(e)
    order = np.argsort(-e)
    ks = np.arange(1, m + 1)
    ok = e[order] >= m / (ks * alpha)
    if not ok.any():
        return []
    kstar = ks[ok].max()
    return sorted(order[:kstar].tolist())


panel, _ = load_panel(args.panel)
if args.clean:
    panel = clean_panel(panel)
mat = log_excess_matrix(panel)                        # years x funds
funds = list(mat.columns)
years = list(mat.index)
m = len(funds)

# e-processes on the calendar: E[t, i] = M_{i,n(t)} where n(t) is plan i's count of observations up to year t;
# 1 before the plan has two observations (an e-value of 1 rejects nothing).
E = pd.DataFrame(np.nan, index=years, columns=funds)
for f in funds:
    s = mat[f].dropna()
    one, _ = e_process(s.values)
    E.loc[s.index, f] = np.where(np.isnan(one), 1.0, one)
E = E.ffill().fillna(1.0)                             # a plan whose record ended keeps its last e-value
Emax = E.cummax()                                     # running maximum, for the adjuster
Eadj = np.sqrt(Emax) - 1                              # A(m) = sqrt(m) - 1 is an adjuster

res = {"programmes": m, "years": [int(y) for y in years], "alpha": args.alpha, "discoveries": {}, "discoveries_adjusted": {},
       "entry_year": {}, "entry_year_adjusted": {}, "path": {}, "average_e_value": {int(y): float(E.loc[y].mean()) for y in years}}
for a in args.alpha:
    path = {}; entry = {}; path_adj = {}; entry_adj = {}
    for y in years:
        rej = [funds[i] for i in ebh(E.loc[y].values, a)]
        path[int(y)] = rej
        for f in rej:
            entry.setdefault(f, int(y))
        rej_adj = [funds[i] for i in ebh(Eadj.loc[y].values, a)]
        path_adj[int(y)] = rej_adj
        for f in rej_adj:
            entry_adj.setdefault(f, int(y))
    key = "%g" % a
    res["discoveries"][key] = path[int(years[-1])]
    res["discoveries_adjusted"][key] = path_adj[int(years[-1])]
    res["entry_year"][key] = entry
    res["entry_year_adjusted"][key] = entry_adj
    res["path"][key] = path
    print("alpha", a, "| e-BH at the end:", path[int(years[-1])], "| entry years:", entry,
          "| adjusted running-max version at the end:", path_adj[int(years[-1])])
# the e-values themselves at the end, sorted
final = E.iloc[-1].sort_values(ascending=False)
res["e_values_end"] = {f: float(v) for f, v in final.items()}
res["average_e_value_end"] = float(final.mean())
res["global_anytime_p_value_end"] = float(min(1.0, 1 / final.mean()))
print("average e-value at the end: %.1f (anytime-valid global p-value %.4f); top five:" % (final.mean(), min(1, 1 / final.mean())))
print(final.head(5).round(1).to_string())
# the years in which the average e-value first exceeded 10 and 20
avg = E.mean(axis=1)
for thr in (10, 20, 100):
    hit = avg[avg >= thr]
    res["average_first_above_%d" % thr] = int(hit.index[0]) if len(hit) else None
    print("average e-value first >= %d in" % thr, res["average_first_above_%d" % thr])
# how much of the average is the largest plan
res["share_of_top_plan_in_average_end"] = float(final.iloc[0] / final.sum())
print("share of the top plan in the average at the end: %.2f" % res["share_of_top_plan_in_average_end"])

out = args.out or str(ROOT / "results" / ("ebh_%s.json" % ("clean" if args.clean else "extended")))
json.dump(res, open(out, "w"), indent=1, default=float)
E.round(3).to_csv(out.replace(".json", "_e_values.csv"))
print("wrote", out)
