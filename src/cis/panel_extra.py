"""Loader for the second workbook (nine further plans and Maine PERS, October 2026).

Each student team laid its sheet out differently; this module parses each layout explicitly rather
than guessing, and returns the same tidy panel as `cis.panel.load_workbook_panel`: fund, year,
actual, benchmark (decimals). Notes recorded in the sheets that matter for interpretation are kept
in NOTES.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_extra_workbook", "NOTES"]

NOTES = {
    "IL TRS": "fiscal 2011-2025, total fund net of fees, benchmark as reported",
    "Houston POPS": "fiscal 2010-2025; 2010-2012 benchmark from the FY2012 CAFR ten-year table; FY2018 benchmark from the FY2019 CAFR",
    "INPRS": "Indiana consolidated DB assets, fiscal 2010-2025; 2010 gross of fees; benchmark = reported target/dynamic index",
    "Colorado PERA": "calendar 2010-2025",
    "Ohio PERS": "calendar 2010-2025",
    "SWIB Core": "Wisconsin Core Fund, calendar 2010-2025 (a second SWIB series; the first workbook's SWIB sheet covers 2007-2024)",
    "Iowa PERS": "fiscal 2010-2025",
    "CT TRS": "fiscal 2010-2025",
    "Maine PERS": "fiscal 2004-2023; benchmark 0.0 recorded for 2020",
}


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan


def _rows(wb, name):
    return [r for r in wb[name].iter_rows(values_only=True) if any(c is not None for c in r)]


def _year_rows(rows, year_col, act_col, bmk_col, scale=1.0, lo=1990, hi=2030):
    out = []
    for r in rows:
        y = r[year_col] if len(r) > year_col else None
        try:
            y = int(float(y))
        except (TypeError, ValueError):
            continue
        if not lo <= y <= hi:
            continue
        a, b = _num(r[act_col]) * scale, _num(r[bmk_col]) * scale
        if np.isfinite(a) and np.isfinite(b):
            out.append((y, a, b))
    return out


def _wide(rows, label_act, label_bmk):
    """Years across a header row; actual and benchmark in rows whose first cell starts with a label."""
    years = None
    for r in rows:
        if r[0] == "Year" or (isinstance(r[0], (int, float)) and 1990 <= float(r[0]) <= 2030):
            years = [(i, int(c)) for i, c in enumerate(r) if isinstance(c, (int, float)) and 1990 <= c <= 2030]
            break
    act = bmk = None
    for r in rows:
        if isinstance(r[0], str) and r[0].startswith(label_act):
            act = r
        if isinstance(r[0], str) and r[0].startswith(label_bmk):
            bmk = r
    return [(y, _num(act[i]), _num(bmk[i])) for i, y in years]


def load_extra_workbook(path: str | Path) -> pd.DataFrame:
    from openpyxl import load_workbook  # optional dependency

    wb = load_workbook(path, read_only=True, data_only=True)
    panel = []

    def add(fund, triples):
        for y, a, b in triples:
            if np.isfinite(a) and np.isfinite(b) and abs(a) < 1.5 and abs(b) < 1.5:
                panel.append(dict(fund=fund, year=y, actual=a, benchmark=b))

    # IL TRS: years across columns; rows labelled
    add("IL TRS", _wide(_rows(wb, "IL TRS"), "Actual Return", "Benchmark Return"))
    # Houston POPS: Fiscal Year as text in col 0, actual col 1, benchmark col 2
    add("Houston POPS", _year_rows(_rows(wb, "Houston POPS"), 0, 1, 2))
    # INPRS: percent units
    add("INPRS", _year_rows(_rows(wb, "INPRS PERF 2010-2025"), 0, 1, 2, scale=0.01))
    # Colorado PERA: two blocks, each a year row followed by a value row
    rows = _rows(wb, "Colorado PERA")
    blocks = {}
    for i, r in enumerate(rows):
        if r[0] in ("Actual Returns", "Benchmark Returns ", "Benchmark Returns"):
            yrs = [int(c) for c in rows[i + 1] if isinstance(c, (int, float)) and 1990 <= c <= 2030]
            vals = [_num(c) for c in rows[i + 2][: len(yrs)]]
            blocks["act" if r[0].startswith("Actual") else "bmk"] = dict(zip(yrs, vals))
    add("Colorado PERA", [(y, blocks["act"][y], blocks["bmk"].get(y, np.nan)) for y in blocks["act"]])
    # Ohio PERS: year col 1, actual col 2, benchmark col 3
    add("Ohio PERS", _year_rows(_rows(wb, "Ohio PERS"), 1, 2, 3))
    # Wisconsin SWIB Core Fund: year col 0, actual col 1, benchmark col 2
    add("SWIB Core", _year_rows(_rows(wb, "Wisconsin SWIB"), 0, 1, 2))
    # Iowa PERS: year col 1, actual col 4, benchmark col 5
    add("Iowa PERS", _year_rows(_rows(wb, "Iowa PERS"), 1, 4, 5))
    # CT TRS: year col 0, actual col 1, benchmark col 2
    add("CT TRS", _year_rows(_rows(wb, "CT TRS"), 0, 1, 2))
    # Maine: year col 1, actual col 2, benchmark col 4
    add("Maine PERS", _year_rows(_rows(wb, "Maine Returns"), 1, 2, 4))
    return pd.DataFrame(panel).sort_values(["fund", "year"]).reset_index(drop=True)
