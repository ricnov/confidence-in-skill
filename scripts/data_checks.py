#!/usr/bin/env python3
"""Data checks on the panel: benchmark columns that are the S&P 500, and sheets that duplicate each other.

    python scripts/data_checks.py data/panel_returns.csv
"""
import sys

import numpy as np
import pandas as pd

from cis.panel import load_panel, SP500_FY_JUNE

panel, _ = load_panel(sys.argv[1] if len(sys.argv) > 1 else "data/panel_returns.csv")
B = panel.pivot(index="year", columns="fund", values="benchmark")
A = panel.pivot(index="year", columns="fund", values="actual")
sp = pd.Series(SP500_FY_JUNE)

print("== benchmark columns that coincide with the S&P 500 total return (June fiscal years), 3+ years ==")
for f in B.columns:
    b = B[f].dropna()
    common = b.index.intersection(sp.index)
    hits = [int(y) for y in common if abs(b[y] - sp[y]) < 0.0016]
    if len(hits) >= 3:
        print("  %-16s %2d of %2d years: %s" % (f, len(hits), len(b), hits))

print("\n== pairs of sheets with identical values in 5+ common years ==")
keys = list(B.columns)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        db = (B[keys[i]] - B[keys[j]]).dropna()
        da = (A[keys[i]] - A[keys[j]]).dropna()
        eq_b = int((db.abs() < 1e-9).sum()); eq_a = int((da.abs() < 1e-9).sum())
        if eq_b >= 5 or eq_a >= 5:
            print("  %-10s %-10s common years %2d | identical benchmark %2d | identical actual %2d | mean |actual diff| %.2f%%" %
                  (keys[i], keys[j], len(db), eq_b, eq_a, 100 * da.abs().mean()))

print("\n== benchmark summary of the flagged plans against the peer-average benchmark ==")
peer = np.log1p(B).mean(axis=1)
for f in ["LASERS", "DAL F&P", "NYSLRS", "WV ERS", "PA SERS", "PA PSERS", "NJ PERS", "NJ TPAF", "NJDRPB"]:
    b = np.log1p(B[f]).dropna(); a = np.log1p(A[f]).dropna()
    print("  %-10s n %2d | benchmark mean %5.2f%% sd %5.2f%% | corr with peer benchmark %.2f | plan-benchmark corr %.2f | TE %.2f%%" %
          (f, len(b), 100 * b.mean(), 100 * b.std(ddof=1), np.corrcoef(b, peer.loc[b.index])[0, 1],
           np.corrcoef(a, b)[0, 1], 100 * (a - b).std(ddof=1)))
