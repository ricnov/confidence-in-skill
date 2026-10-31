# Data

## `panel_returns.csv` — annual returns of 47 US public pension plans

One row per plan and fiscal year: `fund` (short name as in the article's Appendix III), `year`
(fiscal year), `actual` (total-fund return), `benchmark` (policy-benchmark return), as decimals.
895 plan-years, 13 to 35 years per plan, fiscal 1991–2026.

**Source and credit.** The returns were collected from the plans' annual financial reports by
Arun Muralidhar and his students at Georgetown University for Muralidhar (2026), *"Time-Varying Skill:
Analyzing Internal/External Programs and Pension Plans"*, The Journal of Performance Measurement,
Summer 2026, pp. 10–24, and provided to the author of this repository in the calculation workbook
behind the article. They are redistributed here with his permission (correspondence, October 2026).
Anyone using them should cite the article and credit the collectors as above.

**Transcription.** `cis.panel.load_workbook_panel` reads each plan's sheet of the workbook (the
actual and benchmark columns of rows whose first cell is a fiscal year between 1980 and 2030 and
whose returns are below 150% in absolute value); the file is that function's output, unchanged.
Maine PERS, named in the article, has no return sheet in the workbook and is absent.

## `panel_returns_extended.csv` — the 50-programme panel (first workbook plus the nine plans of October 2026)

Same columns. The 42 programmes of `panel_returns.csv` after the exclusions of the paper's Section 2
(`cis.panel.clean_panel`: LASERS, NYSCRF, NJ TPAF, NJDRPB, PA SERS dropped as duplicates or
mis-benchmarked rows), with the first workbook's SWIB sheet replaced by the Wisconsin Core Fund series
of the second workbook (calendar 2010–2025; the first sheet is wrong before 2014), plus eight further
plans from the second workbook: CT TRS, Colorado PERA, Houston POPS, IL TRS, INPRS, Iowa PERS and Ohio
PERS (fiscal 2010/2011–2025) and Maine PERS (fiscal 2004–2023, fiscal 2020 dropped because its
benchmark is recorded as 0.0). 50 programmes, 927 plan-years. Built by `scripts/build_extended_panel.py`
from the second workbook (`cis.panel_extra.load_extra_workbook`, one parser per sheet layout).

**Source and credit.** The nine series were collected by Arun Muralidhar and his students at Georgetown
University after the article and provided in a second workbook in October 2026; they are redistributed
here with his permission (correspondence, 31 October 2026), on the same terms as the first panel: cite
the article and credit the collectors.

## `sheet_terminal_values.csv` — what each workbook sheet printed

Per plan: the sheet name, and the terminal K, confidence, tracking error and correlation printed
by the sheet's own formulas, with the last fiscal year. Used to establish how the published
numbers were computed (41 sheets with geometric returns in the drift gap, 3 with arithmetic means,
3 reproducing neither exactly).

## `appendix_III_published.csv` — the article's Appendix III as printed

Transcribed from the published table and checked cell by cell; the only input the published-column
analysis (`scripts/check_published_numbers.py`) needs.
