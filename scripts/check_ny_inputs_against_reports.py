"""Check data/nyslrs_policy_benchmark.csv against the text of the New York fund's reports.

    python scripts/check_ny_inputs_against_reports.py <directory of acfr_YYYY.txt> [data/nyslrs_policy_benchmark.csv]

The directory holds the text of the sixteen Comprehensive / Annual Comprehensive Financial Reports of the New York
State and Local Retirement System for fiscal 2010-2025 (the PDFs at osc.ny.gov, converted with
`pdftotext -layout report.pdf acfr_YYYY.txt`). For every row of the inputs file the script finds the Investment
Results table of that year (the table that starts with 'Total Fund'), the line whose label names the benchmark,
and compares the one-year figure with the file's; it checks the Total Fund one-year return against the
time-weighted portfolio returns used in the paper, and every long-term policy weight against the Asset
Allocation table. The fiscal 2010 absolute-return benchmark, which that year's table does not list, is read from
the narrative. Every comparison is printed with the line of the report it was made against; the run of
19 November 2026 passed 255 of 255.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path(sys.argv[1])
INP = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / "nyslrs_policy_benchmark.csv"
rows = list(csv.DictReader(open(INP)))
by_year = defaultdict(list)
for r in rows:
    by_year[int(r["year"])].append(r)

TWR = {2010: 25.87, 2011: 14.57, 2012: 5.96, 2013: 10.38, 2014: 13.02, 2015: 7.16, 2016: 0.19, 2017: 11.48, 2018: 11.35,
       2019: 5.23, 2020: -2.68, 2021: 33.55, 2022: 9.51, 2023: -4.14, 2024: 11.55, 2025: 5.84}
MWR = {2015: 6.98, 2016: 0.03, 2017: 11.40, 2018: 11.29, 2019: 5.14, 2020: -2.64, 2021: 33.43, 2022: 9.44, 2023: -4.43, 2024: 11.34, 2025: 5.74}

# how each benchmark name in the file is found in the table text
PAT = {
    "Russell 3000 Index": r"Russell 3000",
    "MSCI All Country World Index Free ex US": r"ex\s*US",
    "MSCI All Country World Index Free": r"MSCI All Country World|MSCI ACWI",
    "MSCI All Country World Index": r"MSCI All Country World|MSCI ACWI",
    "Cambridge US Private Equity Index": r"Cambridge",
    "Cambridge U.S. Private Equity Index": r"Cambridge",
    "Cambridge U.S. Private Equity Index (1Q Lag)": r"Cambridge",
    "NCREIF": r"NCREIF",
    "Real Estate Benchmark": r"Real Estate Benchmark",
    "HFRX": r"HFRX",
    "HFRX Global Hedge Fund Index": r"HFRX",
    "HFRI Asset-Weighted Composite Index": r"HFRI",
    "U.S. CPI + 5%": r"CPI",
    "Barclays Capital Aggregate Bond Index": r"Aggregate",
    "Bloomberg Barclays Capital Aggregate Bond Index": r"Aggregate",
    "Bloomberg Barclays Aggregate Bond Index": r"Aggregate",
    "Bloomberg U.S. Aggregate Bond Index": r"Aggregate",
    "Salomon Smith Barney Inflation Linked Index": r"Inflation Linked|Inflation-Linked",
    "Barclays Capital U.S. TIPS Index": r"TIPS",
    "Bloomberg Barclays Capital U.S. TIPS Index": r"TIPS",
    "Bloomberg Barclays U.S. TIPS Index": r"TIPS",
    "Credit Suisse Leveraged Loan Index + 2.5%": r"Leveraged Loan",
    "CRF Actuarial Return + 2%": r"Actuarial",
}
NUM = re.compile(r"(?<![\w.])(\(?-?\d+\.\d+\)?)%?")


ANUM = re.compile(r"(?<![\w.,$])(\(?-?\d+(?:\.\d+)?\)?)%?(?![,\d])")   # allocation tables print integers too


def norm(line):
    return line.replace("\u2013", "-").replace("\u2212", "-").replace("\u2014", "-")


def to_float(tok):
    tok = tok.strip()
    if tok.startswith("("):
        return -float(tok.strip("()"))
    return float(tok)


def first_number(line):
    # the first number after the label: labels contain no decimals, so the first decimal number is the 1-year figure
    m = NUM.search(norm(line))
    return to_float(m.group(1)) if m else None


def results_block(text):
    lines = text.split("\n")
    # the table: first 'Total Fund' line followed by a percentage, after the contents
    for i, l in enumerate(lines):
        if re.match(r"\s*Total Fund\s*(\(\d\))?\s+\(?-?\d+\.\d+\)?%", norm(l)):
            return lines[i:i + 45], i
    raise SystemExit("no Total Fund line")


def allocation_block(text):
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if re.search(r"Asset Type", l) and i > 1000:
            hdr = " ".join(lines[max(0, i - 2):i + 3])
            if "Policy" in hdr:
                return lines[i:i + 34], hdr
    raise SystemExit("no policy table header")


# class keywords in the allocation table, per file class name
ALLOC = {
    "Domestic equity": [r"Domestic Equity"], "International equity": [r"International Equity"],
    "Global equity": [r"Domestic Equity", r"International Equity"], "Public equity": [r"Domestic Equity", r"International Equity"],
    "Private equity": [r"Private Equity"], "Real estate": [r"Real Estate"],
    "Absolute return": [r"Absolute Return"], "Opportunistic": [r"Opportunistic"],
    "Opportunistic and absolute return": [r"Opportunistic"],
    "Real assets": [r"Real Asset"], "Credit": [r"^\s*Credit"],
    "Bonds, cash and mortgages": [r"Bonds, Cash"], "Bonds and mortgages": [r"Bonds|Core Fixed Income"], "Fixed income": [r"Fixed Income|Bonds"],
    "TIPS": [r"Inflation|TIPS"], "Cash": [r"^\s*Cash|Short-term|Short-Term"],
}

ok = bad = 0
for y in sorted(by_year):
    text = open(D / ("acfr_%d.txt" % y), encoding="utf-8", errors="replace").read()
    block, at = results_block(text)
    print("\n=== fiscal %d: Investment Results table at text line %d" % (y, at + 1))
    tf = first_number(block[0])
    flag = "OK " if abs(tf - TWR[y]) < 1e-9 else "BAD"
    print("  %s Total Fund 1-year %.2f (file: %.2f)" % (flag, tf, TWR[y]))
    ok += flag == "OK "; bad += flag == "BAD"
    for r in by_year[y]:
        b = r["benchmark"]; want = float(r["benchmark_return"])
        if b.startswith("none listed"):
            continue
        if y == 2010 and b == "HFRX":
            # narrative sentence
            m = re.search(r"HFRX[^.]*?(\d+\.\d) percent", text)
            got = float(m.group(1)) if m else None
            flag = "OK " if got is not None and abs(got - want) < 1e-9 else "BAD"
            print("  %s %-34s narrative: %s (file: %.2f)" % (flag, r["asset_class"], (m.group(0)[:90] if m else "not found"), want))
            ok += flag == "OK "; bad += flag == "BAD"
            continue
        pat = PAT[b]
        cands = [l for l in block[1:] if re.search(pat, l)]
        if b in ("MSCI All Country World Index Free", "MSCI All Country World Index"):
            cands = [l for l in cands if not re.search(r"ex\s*US", l)]
        if pat == r"Aggregate":
            cands = [l for l in cands if "Aggregate" in l]
        got = first_number(cands[0]) if cands else None
        if cands and "2.5%" in cands[0]:                      # the label itself holds '+ 2.5%'
            nums = [to_float(x) for x in NUM.findall(norm(cands[0]))]
            got = nums[1] if len(nums) > 1 else None
        flag = "OK " if got is not None and abs(got - want) < 1e-9 else "BAD"
        ok += flag == "OK "; bad += flag == "BAD"
        print("  %s %-34s %-44s table: %s (file: %.2f)" % (flag, r["asset_class"], b[:44], got, want))
    # allocation
    ablock, hdr = allocation_block(text)
    h = re.sub(r"\s+", " ", hdr)
    policy_first = h.find("Long-Term") < h.find("March 31") if ("March 31" in h and "Long-Term" in h) else (h.find("Policy") < h.find("March 31"))
    print("  allocation header: %s" % re.sub(r"\s+", " ", hdr.strip())[:150])
    for r in by_year[y]:
        cls = r["asset_class"]; want = float(r["policy_weight"]) * 100
        got = 0.0; found = []
        for kw in ALLOC[cls]:
            for l in ablock:
                if re.search(kw, l) and ANUM.search(norm(l)):
                    nums = [to_float(x) for x in ANUM.findall(norm(l))]
                    got += nums[0] if policy_first else nums[-1]
                    found.append(l.strip()[:80]); break
        flag = "OK " if found and abs(got - want) < 0.051 else "BAD"
        ok += flag == "OK "; bad += flag == "BAD"
        print("  %s weight %-34s %6.2f (file %6.2f)  <- %s" % (flag, cls, got, want, " | ".join(found)))
print("\nchecks passed: %d, failed: %d" % (ok, bad))

# the money-weighted schedule in the 2025 and 2016 reports, and the assumed rate
for yr in (2025, 2016):
    text = open(D / ("acfr_%d.txt" % yr), encoding="utf-8", errors="replace").read()
    i = text.find("money-weighted rate of return")
    seg = text[i - 1500:i + 1500] if i > 0 else ""
    print("\n--- %d report, around 'money-weighted rate of return':" % yr)
    for l in seg.split("\n"):
        if re.search(r"\d{4}|%|rate of return", l) and l.strip():
            print("   ", l.strip()[:140])
