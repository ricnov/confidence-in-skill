#!/usr/bin/env python3
"""Is there skill anywhere in the panel, and does it change?  (Section 4.4 of the paper.)

    python scripts/run_dispersion_tests.py data/panel_returns.csv [B]      (or the author's workbook .xlsx)

Runs the year-cluster sign-flip null for the panel-level statistics in cis.dispersion, on the
full panel and on the panel without its extreme plans, and writes results/dispersion_tests.csv.
Seed 20261011 and B = 50,000 are the paper's.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cis.panel import load_panel, log_excess_matrix
from cis.dispersion import null_functionals, summarise

if len(sys.argv) < 2:
    sys.exit(__doc__)
B = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

panel, _ = load_panel(sys.argv[1])
mat = log_excess_matrix(panel)
sets = {"all 47": [], "without NYSLRS and LASERS": ["NYSLRS", "LASERS"],
        "without NYSLRS, LASERS, DAL F&P and WV ERS": ["NYSLRS", "LASERS", "DAL F&P", "WV ERS"]}
tables = []
for label, excl in sets.items():
    sub = mat.drop(columns=excl)
    null = null_functionals(sub, B=B, seed=20261011)
    s = summarise(sub, null)
    s.insert(0, "panel", label)
    s.insert(1, "funds", sub.shape[1])
    tables.append(s)
    print("\n== %s (%d funds) ==" % (label, sub.shape[1]))
    print(s.drop(columns=["panel", "funds"]).round(4).to_string())
res = pd.concat(tables)
out = Path("results"); out.mkdir(exist_ok=True)
res.to_csv(out / "dispersion_tests.csv")
print("\nwrote results/dispersion_tests.csv")
