"""The framework's statistic against the actuarial assumed rate instead of the policy benchmark.

    python scripts/run_assumed_rate.py <second workbook .xlsx> [--out results/assumed_rate_nine.csv]

The second workbook's sheets record each plan's assumed actuarial rate by year. Replacing the policy
benchmark by that rate in the same statistic -- sqrt(n) x mean log(1+r_P) - log(1+a) over its standard
deviation -- gives the confidence that the plan earns its assumed rate, and equation (9a) the years of
data needed to reach 90% at the realised gap. The denominator is then the plan's total-return volatility
rather than its tracking error, which is the point of the exercise (Section 6 of the paper).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cis.panel_extra import load_extra_workbook, _rows, _num  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("workbook")
ap.add_argument("--out", default=str(ROOT / "results" / "assumed_rate_nine.csv"))
args = ap.parse_args()

from openpyxl import load_workbook  # noqa: E402

wb = load_workbook(args.workbook, read_only=True, data_only=True)
panel = load_extra_workbook(args.workbook)


def by_year(rows, year_col, rate_col, scale=1.0):
    out = {}
    for r in rows:
        try:
            y = int(float(r[year_col]))
        except (TypeError, ValueError):
            continue
        if 1990 <= y <= 2030 and np.isfinite(_num(r[rate_col])):
            out[y] = _num(r[rate_col]) * scale
    return out


rates = {}
R = _rows(wb, "IL TRS")
hdr = [r for r in R if r[0] == "Year"][0]
yrs = [(i, int(c)) for i, c in enumerate(hdr) if isinstance(c, (int, float)) and 1990 <= c <= 2030]
ar = [r for r in R if isinstance(r[0], str) and r[0].startswith("Assumed")][0]
rates["IL TRS"] = {y: _num(ar[i]) for i, y in yrs}
rates["Houston POPS"] = by_year(_rows(wb, "Houston POPS"), 0, 4)
rates["INPRS"] = by_year(_rows(wb, "INPRS PERF 2010-2025"), 0, 4, scale=0.01)
R = _rows(wb, "Colorado PERA")
for i, r in enumerate(R):
    if isinstance(r[0], str) and r[0].startswith("Assumed"):
        ys = [int(c) for c in R[i + 1] if isinstance(c, (int, float)) and 1990 <= c <= 2030]
        vs = [_num(c) for c in R[i + 2][: len(ys)]]
        rates["Colorado PERA"] = dict(zip(ys, vs))
rates["Ohio PERS"] = by_year(_rows(wb, "Ohio PERS"), 1, 4)
rates["SWIB Core"] = by_year(_rows(wb, "Wisconsin SWIB"), 0, 4)
rates["Iowa PERS"] = by_year(_rows(wb, "Iowa PERS"), 1, 7)
rates["CT TRS"] = by_year(_rows(wb, "CT TRS"), 0, 3)
rates["Maine PERS"] = by_year(_rows(wb, "Maine Returns"), 1, 7)

rows = []
for fund, rt in rates.items():
    rt = {y: v for y, v in rt.items() if np.isfinite(v) and 0.03 < v < 0.12}
    d = panel[(panel.fund == fund) & (panel.year.isin(rt))].sort_values("year")
    if fund == "Maine PERS":
        d = d[d.year != 2020]
    if len(d) < 5:
        continue
    lp = np.log1p(d.actual.values); la = np.log1p(np.array([rt[y] for y in d.year]))
    lb = np.log1p(d.benchmark.values)
    x = lp - la; n = len(x)
    t_a = np.sqrt(n) * x.mean() / x.std(ddof=1)
    e = lp - lb
    t_b = np.sqrt(n) * e.mean() / e.std(ddof=1)
    years90 = (1.2816 * x.std(ddof=1) / x.mean()) ** 2 if x.mean() > 0 else np.inf
    rows.append(dict(fund=fund, years="%d-%d" % (d.year.min(), d.year.max()), n=n,
                     mean_log_return_pct=round(100 * lp.mean(), 2), mean_assumed_pct=round(100 * la.mean(), 2),
                     gap_vs_assumed_bp=round(1e4 * x.mean(), 0), sd_vs_assumed_pct=round(100 * x.std(ddof=1), 2),
                     t_vs_assumed=round(t_a, 2), cis_vs_assumed_pct=round(100 * stats.norm.cdf(t_a), 1),
                     years_to_90_vs_assumed=round(years90, 0),
                     gap_vs_benchmark_bp=round(1e4 * e.mean(), 0), sd_vs_benchmark_pct=round(100 * e.std(ddof=1), 2),
                     t_vs_benchmark=round(t_b, 2), cis_vs_benchmark_pct=round(100 * stats.norm.cdf(t_b), 1)))
out = pd.DataFrame(rows).sort_values("t_vs_assumed", ascending=False)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
print("\nmedian years to 90%% against the assumed rate: %.0f; median total-return s.d. %.1f%%; median tracking error %.2f%%"
      % (out.years_to_90_vs_assumed.replace(np.inf, np.nan).median(), out.sd_vs_assumed_pct.median(), out.sd_vs_benchmark_pct.median()))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print("wrote", args.out)
