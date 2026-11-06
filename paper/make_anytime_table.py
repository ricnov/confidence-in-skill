"""Table rows for the anytime-valid confidence (paper, Section 5 / Appendix A).

    python paper/make_anytime_table.py [results/anytime_valid_extended.json]

Writes paper/table_anytime_rows.tex defining \\AnytimeRows: one row per programme whose confidence reached 90%
at some look on either reading (the framework's path from the tenth year, or the anytime-valid confidence):
years of record, mean excess return, the framework's final confidence, the year its path first crossed 90%,
the anytime-valid confidence at the end of the record and the years it first reached 90% and 95%.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "anytime_valid_extended.json")
J = json.load(open(src))
P = pd.DataFrame(J["per_plan"]).T
P = P[(P["naive_first_90"].notna()) | (P["anytime_first_90"].notna())].sort_values("t", ascending=False)


def yr(v):
    return "--" if v is None or (isinstance(v, float) and pd.isna(v)) else "%d" % int(v)


def pct(p):
    return "%d" % round(100 * float(p))


lines = []
for f, r in P.iterrows():
    lines.append(" & ".join([f.replace("&", "\\&"), "%d" % int(r["n"]), "$%+d$" % round(float(r["mean_bp"])),
                             pct(r["naive_final"]), yr(r["naive_first_90"]),
                             pct(r["C_one_final"]), yr(r["anytime_first_90"]), yr(r["anytime_first_95"])]) + " \\\\")
(ROOT / "paper" / "table_anytime_rows.tex").write_text("\\newcommand{\\AnytimeRows}{%\n" + "\n".join(lines) + "\n}\n")
print("wrote table_anytime_rows.tex with %d rows" % len(lines))
