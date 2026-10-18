# Confidence in skill is one minus a *p*-value

Replication and multiple-testing analysis of the Time-Varying Skill (TVS) framework in
Muralidhar (2026), *"Time-Varying Skill: Analyzing Internal/External Programs and Pension Plans"*,
**The Journal of Performance Measurement**, Summer 2026, pp. 10–24.

The paper measures a fund's skill as "confidence in skill", CIS = Φ(K), with
K = √T · gap / σ_R the fund's drift advantage over its benchmark in units of tracking error,
and applies it to CalSTRS, two internal overlay programmes, a Dutch pension fund and a
cross-section of 48 US public pension plans. This repository makes three points and
reproduces every number behind them.

1. **K is a t-statistic.** With the numerator and denominator taken as their sample
   counterparts on the log excess return d_t = log(1+r_P,t) − log(1+r_L,t), K equals
   √n · mean(d)/sd(d) exactly, so CIS is one minus a one-sided *p*-value evaluated with the
   normal rather than the Student reference distribution (`tests/test_core.py` pins the
   identity to 1e-10). That connects the framework to the track-record-length and
   multiple-testing literatures.
2. **The numerator as implemented double-counts the variance drag.** The paper's tables
   populate the drift μ with annualised *geometric* returns and then subtract σ²/2; a
   geometric return is already a log drift. The extra term is ½(σ²_P − σ²_L) per year. The
   author has confirmed the geometric returns were unintended.
3. **The cross-section is 47 simultaneous tests.** Under the null of no skill anywhere about
   five of 47 funds are expected above 90% confidence; the paper finds 13. Controlling the
   false discovery rate at 10% leaves one significant outperformer (NYSLRS) and one
   significant underperformer (LASERS). With the author's annual returns, a year-cluster
   bootstrap that respects the dependence between funds shows that **the count of
   high-confidence funds is consistent with luck (P ≈ 0.07) while the two extremes are not
   (adjusted p ≈ 0.006 and 0.002).**

## What is in the repository

```
src/cis/core.py        the statistic in every version: paper's, arithmetic-mean, model-consistent t
src/cis/multiple.py    Benjamini–Hochberg, Holm, q-values
src/cis/appendix.py    the published 47-row appendix and everything computable from it alone
src/cis/panel.py       loader for the panel: data/panel_returns.csv, or the author's workbook
src/cis/bootstrap.py   year-cluster wild bootstrap under the global null; max-t adjusted p-values
src/cis/dispersion.py  panel-level statistics under the same null: is there skill anywhere, does it change
data/appendix_III_published.csv   Appendix III as printed, transcribed and checked cell by cell
data/panel_returns.csv            the 47 plans' annual actual and benchmark returns (data/DATA.md)
data/sheet_terminal_values.csv    what each workbook sheet printed (K, confidence, tracking error)
scripts/check_published_numbers.py  reproduces the cross-sectional results from the printed appendix
scripts/run_raw_analysis.py         the same on the annual returns, plus the bootstrap
scripts/run_dispersion_tests.py     the panel-level tests (results/dispersion_tests.csv, paper Table 2)
scripts/run_peer_comparison.py      what comparisons between plans are worth (paper Section 4.6)
scripts/data_checks.py              benchmark columns that are the S&P 500; sheets that duplicate each other
scripts/run_everything.py           every number the paper quotes, from one command (results/summary_*.json)
scripts/build_extended_panel.py     the 50-programme panel with the nine further plans (second workbook)
scripts/run_framework_tests.py      confidence as a forecast: record-and-sequel tests, shrinkage, posterior
                                    probabilities, the common component of excess returns (paper Sections 4.3,
                                    4.8; pre-specification in the docstring; results/framework_tests_*.json)
scripts/run_assumed_rate.py         the statistic against the actuarial assumed rate (paper Section 6;
                                    results/assumed_rate_nine.csv)
scripts/run_timing_models.py        Jensen, Treynor-Mazuy and Henriksson-Merton regressions inside the framework:
                                    per plan, pooled with year-clustered s.e., and panel-level restricted
                                    bootstraps with min-p adjustment (paper Section 4.9; results/timing_models_*.json/.csv)
scripts/make_corrected_exhibits.py  corrected versions of the article's Table 1, Appendices I-III and Figures 1, 2, 4, 5
                                    for the journal's online update (arithmetic numerator; model-consistent alongside)
src/cis/panel_extra.py              loader for the second workbook's nine sheets
results/fund_statistics.csv         per-fund statistics on the annual returns (every column of the paper's Appendix A)
paper/                 the paper's figures and generated tables: paper/build_paper_inputs.py (tables, Figure 2),
                       paper/make_figures_2_4.py (Figures 1, 4 and the half-sample persistence scatter) and
                       paper/make_figure_forecast.py (Figure 3) from the panel; the manuscript is added on
                       submission
tests/test_core.py     identity, BH/Holm, and reproduction of the published-column results
```

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                   # 9 tests
python scripts/check_published_numbers.py   # needs nothing but the printed appendix
```

Output of the second command, from the published column alone: 13 funds above 90% against 4.7
expected under the null; mean confidence 0.603; Kolmogorov–Smirnov *p* = 0.014; BH at 10%
FDR: one outperformer (NYSLRS, published 100%) and one underperformer (LASERS, 0%); the
drag term recovered by inverting the published confidences exceeds the model-consistent gap
for five funds and determines its sign for two.

## What the data checks found

`scripts/data_checks.py` compares every sheet's columns with each other and with the S&P 500. Five of the
47 rows are not independent, correctly benchmarked plans: the **LASERS** benchmark column is the S&P 500
total return for fiscal 2010–2025 (16 years to the decimal); **NYSCRF** is the same fund as **NYSLRS**
(identical returns in all 13 common years) under a different benchmark column, against which the fund's
*t* is 0.15 instead of 4.24; the three **New Jersey** rows are one investment programme; the **PA SERS**
sheet copies PA PSERS in 20 of 22 years. `cis.panel.clean_panel` reduces the panel to 42 programmes, and
`scripts/run_everything.py --clean` reproduces every number in the paper on it (`--ny NYSCRF` measures
New York against the other column). The article's two extreme results — a plan at 0% and a plan at
100% — are these two benchmark columns.

## Nine further plans (second workbook, October 2026)

`cis.panel_extra.load_extra_workbook` parses the second workbook's nine sheets (each laid out
differently) into the same tidy panel; `scripts/build_extended_panel.py` appends them to the 42
programmes, with the Wisconsin Core Fund series replacing the first workbook's SWIB sheet (wrong before
2014) and Maine PERS's fiscal 2020 dropped (benchmark recorded as 0.0): 50 programmes, 927 plan-years.
`results/summary_extended.json` holds every statistic on it. The returns of the nine are not yet
distributed here; the loader and results are.

`scripts/run_framework_tests.py` asks what a confidence value is worth as a forecast. Each programme is
split at its tenth observed year (the framework's first look) into a record and a sequel: on the 50
programmes, fifteen were at or above 90% after ten years and went on to beat their benchmarks by 66 bp
a year against −4 bp for the other 35 (P = 0.02 under the year-cluster null; 14 of 15 positive); the
record forecasts the sequel better than zero (MSE ratio 0.69, P = 0.006) but is no closer than zero for
the median programme, and an empirical-Bayes shrinkage of the record beats it for 28 of 50 while
over-shrinking DAL F&P. The posterior probability of skill under the cross-section's normal prior is
Φ(t·τ/√(τ²+se²)); with τ = 56 bp a 90% confidence is an 86% probability at the median tracking error.
The same script measures the common component of excess returns (a quarter of the variance, a two-year
reversal, exposure proportional to tracking error and near zero for the surviving records). The 42-programme
version runs on the public data: `python scripts/run_framework_tests.py data/panel_returns.csv --clean`.

`scripts/run_timing_models.py` relaxes the framework's two implicit assumptions, unit beta and no timing,
with Jensen, Treynor-Mazuy and Henriksson-Merton regressions of the log excess return on the benchmark's log
return. Pooled with plan fixed effects and year-clustered standard errors the panel is unit-beta (β − 1 =
0.02 ± 0.04) with no convexity (γ ≈ 0 on either model); plan by plan the coefficients are fitted through
fiscal 2009, 2021 and 2022, and the Henriksson-Merton regressor is non-zero in three years for the median
plan. Freeing the beta inflates the standard error of alpha by √(1 + ℓ̄²/s²) ≈ 1.3 and leaves the panel
indistinguishable from no selection skill anywhere; with it fixed at one (the framework's statistic) the
panel rejects. Several high-confidence records are beta rather than selection (PA PSERS 1.25, MIZZOU 1.21,
Indiana 1.07, CalSTRS 1.10); SWIB's and NYSLRS's are selection. The 42-programme version runs on the public
data: `python scripts/run_timing_models.py data/panel_returns.csv --clean`.

## Results on the annual returns

The author provided his calculation workbook (one sheet per fund, annual actual and benchmark
returns, 47 funds, 895 fund-years, 1991–2026) and agreed to its redistribution: the transcribed
panel is `data/panel_returns.csv` (provenance and credit in `data/DATA.md`), and
`scripts/run_raw_analysis.py data/panel_returns.csv` runs the full analysis on it. The scripts
accept the workbook itself in place of the CSV.

| | published column | annual returns |
|---|---|---|
| Table 1 (CalSTRS, 1999–2023) | K = 0.374528, 64.60% | reproduces exactly, with return correlation 0.9937 (the printed 0.996516 is not what the table uses) |
| Sheets reproduced with geometric returns in the gap | — | 41 of 47; 3 (AZSRS, NC TERS, SWIB) use arithmetic means |
| Median drag share, ½|σ²_P − σ²_L| / |gap*| | 22% (inverted) | 14% (measured) |
| CalSTRS, model-consistent numerator | 74.8% (FY2023) | 75.0% (FY2023), 79.0% (FY2025) |
| Funds above 90% | 13 | 13 published, 14 model-consistent |
| NYSLRS | K* = 3.68, p_t = 0.0012, q = 0.058 | t = 4.24, p_t = 0.0004, q = 0.017 |
| Outperformers at 10% / 20% FDR | 1 / 5 | 1 / 9 |
| Mean pairwise correlation of annual excess returns | — | 0.24 |
| Count above 90%, year-cluster bootstrap under the null | — | observed 14; null 95th percentile 15; **P = 0.07** (0.0003 under independence) |
| Max t, same bootstrap | — | null 95% = 3.29; NYSLRS 4.24, **adjusted p = 0.006**; LASERS −4.73, **0.002** |
| S.d. of mean excess return across plans, same null | — | observed 1.02% p.a. against a null of 0.50% (95th percentile 0.67%): **P < 0.0001**; without NYSLRS, LASERS, DAL F&P and WV ERS, 0.52% against 0.44%, P = 0.20 |
| Half-sample change in mean excess return | — | sum of squared z 31 against a null mean of 54 (P = 0.93); largest |z| 2.83 (NYSLRS, a fall), null 95th percentile 3.73 |

So the panel does contain skill, but it is concentrated: four plans carry the whole rejection of the
global null, the other 43 are statistically one population, and no plan's skill can be shown to have
changed over its sample.

The bootstrap multiplies each calendar year's cross-section of demeaned log excess returns
by a common random sign (50,000 draws), which imposes the null, keeps every fund's sample
size and the cross-sectional dependence, and avoids the duplicated-year artefacts a
resampling bootstrap produces on short annual series.

## Reference

Muralidhar, A. (2026). Time-varying skill: Analyzing internal/external programs and pension
plans. *The Journal of Performance Measurement*, Summer 2026, 10–24.

The paper, *Confidence in Skill Is One Minus a p-Value: Multiple testing and dependence in
time-varying skill measures for pension funds* (draft, October 2026), will be added to `paper/` when it
is submitted; every number in it is produced by the scripts above. Views are the author's own. MIT
licence; the returns are credited as described in `data/DATA.md`.
