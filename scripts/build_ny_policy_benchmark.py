"""The New York fund's policy benchmark, fiscal 2010-2025, rebuilt from its Annual Comprehensive Financial Reports.

    python scripts/build_ny_policy_benchmark.py [data/panel_returns.csv] [--write]

The NYSLRS sheet's benchmark column was built by the article's collectors as the sum over asset classes of the
Fund's long-term policy weight times the one-year return of the class's benchmark, both read from the Annual
Comprehensive Financial Report (ACFR; Comprehensive Annual Financial Report before 2022) of the New York State
and Local Retirement System for the fiscal year ending 31 March. This script rebuilds that sum from the reports
for every year of the sheet. The inputs are the ACFR's long-term policy allocation and the one-year benchmark
returns in the Investment Results table of the Investment Section; where the report lists no benchmark for a
class, opportunistic (through fiscal 2020) and real assets (through fiscal 2016) take the absolute-return
benchmark, as in the collectors' workbook, and cash (one or two percent of the policy) takes zero. From fiscal
2021 the report lists the actuarial rate of return plus two points for opportunistic and absolute-return
strategies and the consumer price index plus five for real assets, and those are used.

With --write the inputs go to data/nyslrs_policy_benchmark.csv (one row per year and class) and the annual sums
are printed in the form of the NY_POLICY_BENCHMARK constant in cis/panel.py. Without it the script prints the
series next to the sheet's column and the fund's return, with NYSLRS's statistics against each.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cis.panel import load_panel  # noqa: E402

# One-year benchmark returns and long-term policy weights, by fiscal year (ended 31 March), from the ACFR of
# that year: Investment Section, Investment Results (returns) and asset allocation (policy). Returns in percent.
HFRX = {2010: 12.90, 2011: 6.29, 2012: -7.18, 2013: 2.74, 2014: 5.62, 2015: -0.21, 2016: -8.19, 2017: 7.48,
        2018: 4.26, 2019: -4.10, 2020: -7.51}          # HFRX; HFRI from fiscal 2020; 2010 from the report's narrative
rows = []


def add(year, cls, weight, benchmark, ret):
    rows.append(dict(year=year, asset_class=cls, policy_weight=weight, benchmark=benchmark, benchmark_return=ret))


# fiscal 2010: domestic and international equity measured separately; bonds, cash and mortgages one class
add(2010, "Domestic equity", .35, "Russell 3000", 52.44)
add(2010, "International equity", .16, "MSCI ACWI ex US", 60.93)
add(2010, "Private equity", .08, "Cambridge Associates US Private Equity", 13.46)
add(2010, "Real estate", .06, "NCREIF Property", -16.86)
add(2010, "Absolute return", .05, "HFRX Global Hedge Fund", HFRX[2010])
add(2010, "Bonds, cash and mortgages", .225, "Barclays Capital Aggregate", 7.69)
add(2010, "TIPS", .075, "Salomon Smith Barney Inflation Linked", 5.95)

# fiscal 2011-2014: global equity 43 (domestic 30, international 13), PE 10, RE 6, AR 4, opportunistic 4,
# real assets 3, bonds and mortgages 22, TIPS 8
R = {2011: dict(eq=14.08, pe=19.97, re=13.11, agg=5.12, tips=8.24),
     2012: dict(eq=-0.73, pe=21.07, re=14.26, agg=7.71, tips=12.58),
     2013: dict(eq=10.55, pe=13.81, re=10.54, agg=3.77, tips=5.71),
     2014: dict(eq=16.55, pe=20.77, re=10.99, agg=-0.10, tips=-6.78)}
for y, r in R.items():
    add(y, "Global equity", .43, "MSCI ACWI", r["eq"])
    add(y, "Private equity", .10, "Cambridge Associates US Private Equity", r["pe"])
    add(y, "Real estate", .06, "NCREIF Property", r["re"])
    add(y, "Absolute return", .04, "HFRX Global Hedge Fund", HFRX[y])
    add(y, "Opportunistic", .04, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Real assets", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Bonds and mortgages", .22, "Barclays Capital Aggregate", r["agg"])
    add(y, "TIPS", .08, "Citigroup Inflation Linked Securities", r["tips"])

# fiscal 2015-2016: long-term policy equity 51 (38 + 13), PE 10, RE 8, AR 3, opportunistic 3, real assets 3,
# bonds and mortgages 18, cash 2, TIPS 2
R = {2015: dict(eq=5.42, pe=11.09, re=11.80, agg=5.72, tips=3.82),
     2016: dict(eq=-4.34, pe=5.26, re=13.35, agg=1.96, tips=1.51)}
for y, r in R.items():
    add(y, "Global equity", .51, "MSCI ACWI", r["eq"])
    add(y, "Private equity", .10, "Cambridge Associates US Private Equity", r["pe"])
    add(y, "Real estate", .08, "NCREIF Property", r["re"])
    add(y, "Absolute return", .03, "HFRX Global Hedge Fund", HFRX[y])
    add(y, "Opportunistic", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Real assets", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Bonds and mortgages", .18, "Barclays Capital Aggregate", r["agg"])
    add(y, "Cash", .02, "none listed; zero", 0.0)
    add(y, "TIPS", .02, "Citigroup Inflation Linked Securities", r["tips"])

# fiscal 2017-2020: long-term policy equity 50 (36 + 14), PE 10, RE 10, AR 2, opportunistic 3, real assets 3,
# bonds and mortgages 17, cash 1, TIPS 4; real assets measured against CPI + 5% from fiscal 2017
R = {2017: dict(eq=15.04, pe=13.25, re=7.97, ra=2.40, agg=0.44, tips=1.48),
     2018: dict(eq=14.85, pe=17.49, re=6.98, ra=7.57, agg=1.20, tips=0.92),
     2019: dict(eq=2.60, pe=10.69, re=6.70, ra=6.90, agg=4.48, tips=2.70),
     2020: dict(eq=-11.26, pe=14.04, re=5.19, ra=6.59, agg=8.93, tips=6.85)}
for y, r in R.items():
    add(y, "Global equity", .50, "MSCI ACWI", r["eq"])
    add(y, "Private equity", .10, "Cambridge Associates US Private Equity", r["pe"])
    add(y, "Real estate", .10, "NCREIF Property" if y < 2020 else "real estate benchmark as listed", r["re"])
    add(y, "Absolute return", .02, "HFRX Global Hedge Fund" if y < 2020 else "HFRI Fund of Funds Composite", HFRX[y])
    add(y, "Opportunistic", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Real assets", .03, "CPI + 5%", r["ra"])
    add(y, "Bonds and mortgages", .17, "Bloomberg Barclays Aggregate", r["agg"])
    add(y, "Cash", .01, "none listed; zero", 0.0)
    add(y, "TIPS", .04, "Bloomberg Barclays US TIPS", r["tips"])

# fiscal 2021-2025: public equity 47 (32 + 15), PE 10, RE 9, opportunistic/absolute-return 3, credit 4,
# real assets 3, fixed income 23, cash 1; fiscal 2025 policy: 39 (25 + 14), 15, 12, 3, 4, 4, 22, 1
R = {2021: dict(eq=54.60, pe=22.93, re=1.11, oars=8.80, cr=23.74, ra=7.86, agg=0.71),
     2022: dict(eq=7.28, pe=39.92, re=21.66, oars=8.80, cr=5.79, ra=14.08, agg=-4.15),
     2023: dict(eq=-7.44, pe=-2.05, re=7.26, oars=8.80, cr=4.67, ra=9.80, agg=-4.78),
     2024: dict(eq=23.22, pe=9.07, re=-12.14, oars=7.90, cr=15.19, ra=8.61, agg=1.70),
     2025: dict(eq=7.15, pe=8.36, re=-1.64, oars=7.90, cr=9.67, ra=7.47, agg=4.88)}
for y, r in R.items():
    w = dict(eq=.47, pe=.10, re=.09, oars=.03, cr=.04, ra=.03, fi=.23, cash=.01) if y < 2025 else \
        dict(eq=.39, pe=.15, re=.12, oars=.03, cr=.04, ra=.04, fi=.22, cash=.01)
    add(y, "Public equity", w["eq"], "MSCI ACWI", r["eq"])
    add(y, "Private equity", w["pe"], "Cambridge Associates US Private Equity", r["pe"])
    add(y, "Real estate", w["re"], "real estate benchmark as listed", r["re"])
    add(y, "Opportunistic and absolute return", w["oars"], "actuarial rate of return + 2%", r["oars"])
    add(y, "Credit", w["cr"], "Credit Suisse Leveraged Loan + 2.5%", r["cr"])
    add(y, "Real assets", w["ra"], "CPI + 5%", r["ra"])
    add(y, "Fixed income", w["fi"], "Bloomberg US Aggregate", r["agg"])
    add(y, "Cash", w["cash"], "none listed; zero", 0.0)

inputs = pd.DataFrame(rows)
wsum = inputs.groupby("year")["policy_weight"].sum()
assert np.allclose(wsum, 1.0), wsum[~np.isclose(wsum, 1.0)]
series = (inputs["policy_weight"] * inputs["benchmark_return"] / 100).groupby(inputs["year"]).sum().round(6)

if "--write" in sys.argv:
    out = ROOT / "data" / "nyslrs_policy_benchmark.csv"
    inputs.to_csv(out, index=False)
    print("wrote", out, "(%d rows)" % len(inputs))
    print("NY_POLICY_BENCHMARK = {" + ", ".join("%d: %.6f" % (y, v) for y, v in series.items()) + "}")
    sys.exit()

args = [a for a in sys.argv[1:] if not a.startswith("--")]
panel, _ = load_panel(args[0] if args else ROOT / "data" / "panel_returns.csv")
ny = panel[panel["fund"] == "NYSLRS"].set_index("year")
cmp = pd.DataFrame({"fund": ny["actual"], "sheet column": ny["benchmark"], "policy benchmark": series})
cmp["excess, sheet"] = cmp["fund"] - cmp["sheet column"]
cmp["excess, policy"] = cmp["fund"] - cmp["policy benchmark"]
pd.set_option("display.width", 140)
print((100 * cmp).round(2).to_string())


def stat(rP, rL):
    d = np.log1p(rP) - np.log1p(rL)
    t = np.sqrt(len(d)) * d.mean() / d.std(ddof=1)
    return "mean log excess %.0f bp, tracking error %.0f bp, t = %.2f, Phi(t) = %.3f, one-sided p = %.4f" % (
        1e4 * d.mean(), 1e4 * d.std(ddof=1), t, stats.norm.cdf(t), stats.t.sf(t, len(d) - 1))


print("\nNYSLRS against the sheet's column:   ", stat(cmp["fund"].values, cmp["sheet column"].values))
print("NYSLRS against the policy benchmark: ", stat(cmp["fund"].values, cmp["policy benchmark"].values))
print("  the same without fiscal 2010:      ", stat(cmp["fund"].values[1:], cmp["policy benchmark"].values[1:]))
