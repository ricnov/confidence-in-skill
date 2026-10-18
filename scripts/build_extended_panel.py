#!/usr/bin/env python3
"""The extended panel: the 42 programmes of the article's workbook, with SWIB replaced by the Core Fund
series of the second workbook (the first workbook's SWIB sheet is wrong before 2014; SWIB's own 2011
release confirms 1.4% against 0.9%), plus the seven further plans and Maine PERS from the second
workbook (Maine's fiscal 2020, whose benchmark is recorded as 0.0, is dropped). 50 programmes.

    python scripts/build_extended_panel.py <second_workbook.xlsx> [data/panel_returns.csv]
"""
import sys
from pathlib import Path

import pandas as pd

from cis.panel import load_panel, clean_panel
from cis.panel_extra import load_extra_workbook

extra = load_extra_workbook(sys.argv[1])
base, _ = load_panel(sys.argv[2] if len(sys.argv) > 2 else "data/panel_returns.csv")
core = clean_panel(base)
core = core[core["fund"] != "SWIB"]
extra = extra[~((extra["fund"] == "Maine PERS") & (extra["year"] == 2020))]
extra = extra.replace({"fund": {"SWIB Core": "SWIB"}})
ext = pd.concat([core, extra], ignore_index=True).sort_values(["fund", "year"]).reset_index(drop=True)
out = Path("data/panel_returns_extended.csv")
ext.to_csv(out, index=False)
print("wrote", out, "programmes", ext["fund"].nunique(), "plan-years", len(ext))
