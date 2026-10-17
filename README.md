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
results/fund_statistics.csv         per-fund statistics on the annual returns (every column of the paper's Appendix A)
paper/                 the paper's figure and generated tables: paper/make_rows.py and
                       paper/make_tests_table.py build the tables from results/, paper/make_figure.py the
                       figure from the panel (same seed as the paper); the manuscript is added on submission
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
