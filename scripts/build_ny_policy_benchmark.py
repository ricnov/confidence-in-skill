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

With --write the inputs go to data/nyslrs_policy_benchmark.csv (one row per year and class, with the report, the
pages of its two tables and its URL at osc.ny.gov) and the annual sums are printed in the form of the
NY_POLICY_BENCHMARK constant in cis/panel.py. Index names are as each report prints them. Without it the script prints the
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
# where each year's two tables are: (Investment Results page, Asset Allocation page), as the PDFs number them
PAGES = {2010: (69, 71), 2011: (71, 73), 2012: (70, 72), 2013: (74, 76), 2014: (78, 80), 2015: (86, 88), 2016: (90, 92),
         2017: (93, 95), 2018: (94, 96), 2019: (95, 97), 2020: (98, 100), 2021: (104, 106), 2022: (108, 110),
         2023: (108, 110), 2024: (113, 115), 2025: (113, 115)}
URL = ("https://www.osc.ny.gov/files/retirement/resources/pdf/comprehensive-annual-financial-report-%d.pdf",
       "https://www.osc.ny.gov/files/retirement/resources/pdf/annual-comprehensive-financial-report-%d.pdf")
rows = []


NOTES = {"none listed; absolute-return benchmark": "no benchmark listed for this class; the absolute-return benchmark is used, as in the collectors' sheet",
         "none listed; zero": "no benchmark listed; zero is used (the Fund's own short-term return would add at most 6 bp a year to the benchmark)"}


def add(year, cls, weight, benchmark, ret):
    report = ("CAFR" if year < 2022 else "ACFR") + " FY%d" % year
    pages = "Investment Results p. %d; Asset Allocation p. %d" % PAGES[year]
    note = NOTES.get(benchmark, "")
    if year == 2017 and cls == "Real assets":
        note = "the report prints 2.40 for 'U.S. CPI +5', the inflation rate alone; taken as printed (CPI + 5 = 7.4 would give 10.37 for the year and t = 1.62)"
    if year == 2010 and cls == "Absolute return":
        note = "not in the table; the narrative (p. 65) says the program benchmark, the HFRX, returned 12.9 percent"
    rows.append(dict(year=year, asset_class=cls, policy_weight=weight, benchmark=benchmark, benchmark_return=ret,
                     report=report, pages=pages, url=URL[0 if year < 2022 else 1] % year, note=note))


# fiscal 2010: domestic and international equity measured separately; bonds, cash and mortgages one class
add(2010, "Domestic equity", .35, "Russell 3000 Index", 52.44)
add(2010, "International equity", .16, "MSCI All Country World Index Free ex US", 60.93)
add(2010, "Private equity", .08, "Cambridge US Private Equity Index", 13.46)
add(2010, "Real estate", .06, "NCREIF", -16.86)
add(2010, "Absolute return", .05, "HFRX", HFRX[2010])
add(2010, "Bonds, cash and mortgages", .225, "Barclays Capital Aggregate Bond Index", 7.69)
add(2010, "TIPS", .075, "Salomon Smith Barney Inflation Linked Index", 5.95)

# fiscal 2011-2014: global equity 43 (domestic 30, international 13), PE 10, RE 6, AR 4, opportunistic 4,
# real assets 3, bonds and mortgages 22, TIPS 8
R = {2011: dict(eq=14.08, pe=19.97, re=13.11, agg=5.12, tips=8.24),
     2012: dict(eq=-0.73, pe=21.07, re=14.26, agg=7.71, tips=12.58),
     2013: dict(eq=10.55, pe=13.81, re=10.54, agg=3.77, tips=5.71),
     2014: dict(eq=16.55, pe=20.77, re=10.99, agg=-0.10, tips=-6.78)}
for y, r in R.items():
    add(y, "Global equity", .43, "MSCI All Country World Index Free", r["eq"])
    add(y, "Private equity", .10, "Cambridge U.S. Private Equity Index", r["pe"])
    add(y, "Real estate", .06, "NCREIF", r["re"])
    add(y, "Absolute return", .04, "HFRX Global Hedge Fund Index", HFRX[y])
    add(y, "Opportunistic", .04, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Real assets", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Bonds and mortgages", .22, "Barclays Capital Aggregate Bond Index", r["agg"])
    add(y, "TIPS", .08, "Salomon Smith Barney Inflation Linked Index", r["tips"])

# fiscal 2015-2016: long-term policy equity 51 (38 + 13), PE 10, RE 8, AR 3, opportunistic 3, real assets 3,
# bonds and mortgages 18, cash 2, TIPS 2
R = {2015: dict(eq=5.42, pe=11.09, re=11.80, agg=5.72, tips=3.82),
     2016: dict(eq=-4.34, pe=5.26, re=13.35, agg=1.96, tips=1.51)}
for y, r in R.items():
    add(y, "Global equity", .51, "MSCI All Country World Index Free", r["eq"])
    add(y, "Private equity", .10, "Cambridge U.S. Private Equity Index", r["pe"])
    add(y, "Real estate", .08, "NCREIF", r["re"])
    add(y, "Absolute return", .03, "HFRX Global Hedge Fund Index", HFRX[y])
    add(y, "Opportunistic", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Real assets", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Bonds and mortgages", .18, "Barclays Capital Aggregate Bond Index", r["agg"])
    add(y, "Cash", .02, "none listed; zero", 0.0)
    add(y, "TIPS", .02, "Salomon Smith Barney Inflation Linked Index" if y == 2015 else "Barclays Capital U.S. TIPS Index", r["tips"])

# fiscal 2017-2020: long-term policy equity 50 (36 + 14), PE 10, RE 10, AR 2, opportunistic 3, real assets 3,
# bonds and mortgages 17, cash 1, TIPS 4; real assets measured against CPI + 5% from fiscal 2017
R = {2017: dict(eq=15.04, pe=13.25, re=7.97, ra=2.40, agg=0.44, tips=1.48),
     2018: dict(eq=14.85, pe=17.49, re=6.98, ra=7.57, agg=1.20, tips=0.92),
     2019: dict(eq=2.60, pe=10.69, re=6.70, ra=6.90, agg=4.48, tips=2.70),
     2020: dict(eq=-11.26, pe=14.04, re=5.19, ra=6.59, agg=8.93, tips=6.85)}
AGG = {2017: "Barclays Capital Aggregate Bond Index", 2018: "Bloomberg Barclays Capital Aggregate Bond Index",
       2019: "Bloomberg Barclays Aggregate Bond Index", 2020: "Bloomberg Barclays Aggregate Bond Index"}
TIPS = {2017: "Barclays Capital U.S. TIPS Index", 2018: "Bloomberg Barclays Capital U.S. TIPS Index",
        2019: "Bloomberg Barclays U.S. TIPS Index", 2020: "Bloomberg Barclays U.S. TIPS Index"}
for y, r in R.items():
    add(y, "Global equity", .50, "MSCI All Country World Index Free" if y < 2019 else "MSCI All Country World Index", r["eq"])
    add(y, "Private equity", .10, "Cambridge U.S. Private Equity Index" + (" (1Q Lag)" if y == 2020 else ""), r["pe"])
    add(y, "Real estate", .10, "NCREIF" if y < 2020 else "Real Estate Benchmark", r["re"])
    add(y, "Absolute return", .02, "HFRX Global Hedge Fund Index" if y < 2020 else "HFRI Asset-Weighted Composite Index", HFRX[y])
    add(y, "Opportunistic", .03, "none listed; absolute-return benchmark", HFRX[y])
    add(y, "Real assets", .03, "U.S. CPI + 5%", r["ra"])
    add(y, "Bonds and mortgages", .17, AGG[y], r["agg"])
    add(y, "Cash", .01, "none listed; zero", 0.0)
    add(y, "TIPS", .04, TIPS[y], r["tips"])

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
    add(y, "Public equity", w["eq"], "MSCI All Country World Index", r["eq"])
    add(y, "Private equity", w["pe"], "Cambridge U.S. Private Equity Index (1Q Lag)", r["pe"])
    add(y, "Real estate", w["re"], "Real Estate Benchmark", r["re"])
    add(y, "Opportunistic and absolute return", w["oars"], "CRF Actuarial Return + 2%", r["oars"])
    add(y, "Credit", w["cr"], "Credit Suisse Leveraged Loan Index + 2.5%", r["cr"])
    add(y, "Real assets", w["ra"], "U.S. CPI + 5%", r["ra"])
    add(y, "Fixed income", w["fi"], "Bloomberg Barclays Aggregate Bond Index" if y < 2024 else "Bloomberg U.S. Aggregate Bond Index", r["agg"])
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
