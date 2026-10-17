"""Load the author's calculation workbook into a tidy panel of annual returns.

The workbook (one sheet per fund, annual actual and benchmark returns plus the sheet's own
calculations) is not distributed with this repository. Supply its path; the loader returns
one row per fund-year with the fund's short name, and a metadata frame with each sheet's
terminal K and confidence so the published appendix can be reproduced from the sheets.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_workbook_panel", "load_panel", "log_excess_matrix"]

_SKIP = {"Dist", "Stat", "Main"}
_SUFFIX = re.compile(r"\s+\d{2}-\d{2}(Q\d)?$")


def _find_header(rows):
    for i, r in enumerate(rows[:6]):
        s = [str(c).lower() if c is not None else "" for c in r]
        if any("actual" in c for c in s) and any("benchmark" in c for c in s):
            return i, s
    return None, None


def load_workbook_panel(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, meta). panel: fund, year, actual, benchmark. meta: per-sheet terminal values."""
    from openpyxl import load_workbook  # optional dependency

    wb = load_workbook(path, read_only=True, data_only=True)
    records, meta = [], []
    for name in wb.sheetnames:
        if name in _SKIP or "Chart" in name:
            continue
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        hi, hdr = _find_header(rows)
        if hi is None:
            continue

        def col(pred):
            for j, c in enumerate(hdr):
                if pred(c):
                    return j
            return None

        c_act = col(lambda c: c.startswith("actual"))
        c_bmk = col(lambda c: c.startswith("benchmark"))
        c_k = col(lambda c: c.strip() == "k")
        c_cis = col(lambda c: "confidence" in c)
        c_te = col(lambda c: "tracking" in c)
        c_cor = col(lambda c: "correlation" in c)
        key = _SUFFIX.sub("", name).strip()
        last = None
        for r in rows[hi + 1:]:
            try:
                y = int(r[0])
            except (TypeError, ValueError):
                continue
            if not 1980 <= y <= 2030:
                continue
            a, b = r[c_act], r[c_bmk]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a) < 1.5 and abs(b) < 1.5:
                records.append(dict(fund=key, year=y, actual=float(a), benchmark=float(b)))
                last = r
        if last is not None:
            def num(j):
                v = last[j] if j is not None else None
                return float(v) if isinstance(v, (int, float)) else np.nan
            meta.append(dict(fund=key, sheet=name, sheet_K=num(c_k), sheet_CIS=num(c_cis),
                             sheet_TE=num(c_te), sheet_rho=num(c_cor), last_year=int(last[0])))
    panel = pd.DataFrame(records).sort_values(["fund", "year"]).reset_index(drop=True)
    return panel, pd.DataFrame(meta).set_index("fund")


def log_excess_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Year x fund matrix of log excess returns, NaN where a fund has no observation."""
    d = np.log1p(panel["actual"]) - np.log1p(panel["benchmark"])
    return panel.assign(d=d).pivot(index="year", columns="fund", values="d").sort_index()


def load_panel(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the panel from the author's workbook (.xlsx) or from the distributed CSV.

    With a CSV (data/panel_returns.csv), the per-sheet terminal values are read from
    sheet_terminal_values.csv in the same folder when it exists, so every script runs on the
    distributed files exactly as it runs on the workbook.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return load_workbook_panel(path)
    panel = pd.read_csv(path).sort_values(["fund", "year"]).reset_index(drop=True)
    meta_path = path.with_name("sheet_terminal_values.csv")
    if meta_path.exists():
        meta = pd.read_csv(meta_path).set_index("fund")
    else:
        meta = pd.DataFrame(index=pd.Index(sorted(panel["fund"].unique()), name="fund"))
    return panel, meta
