"""The cleaned panel and the New York fund's benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cis.core import skill_from_returns
from cis.panel import load_panel, clean_panel, NY_POLICY_BENCHMARK, DATA_PROBLEMS

ROOT = Path(__file__).resolve().parents[1]


def _ny(panel):
    d = panel[panel["fund"] == "NYSLRS"].sort_values("year")
    return skill_from_returns(d["actual"].values, d["benchmark"].values)


def test_policy_benchmark_constant_is_the_sum_of_the_distributed_inputs():
    inp = pd.read_csv(ROOT / "data" / "nyslrs_policy_benchmark.csv")
    assert np.allclose(inp.groupby("year")["policy_weight"].sum(), 1.0)
    s = (inp["policy_weight"] * inp["benchmark_return"] / 100).groupby(inp["year"]).sum()
    assert set(s.index) == set(NY_POLICY_BENCHMARK)
    for y, v in NY_POLICY_BENCHMARK.items():
        assert s[y] == pytest.approx(v, abs=1e-6)
    assert NY_POLICY_BENCHMARK[2010] == pytest.approx(0.3099, abs=5e-5)   # the sheet has 0.2528


def test_clean_panel_measures_new_york_against_each_benchmark():
    panel, _ = load_panel(ROOT / "data" / "panel_returns.csv")
    cl = clean_panel(panel)
    assert cl["fund"].nunique() == 42 and not set(cl["fund"]) & set(DATA_PROBLEMS)
    assert round(_ny(cl)["t"], 2) == 1.64                                  # policy benchmark (default)
    assert round(_ny(clean_panel(panel, ny_benchmark="NYSLRS"))["t"], 2) == 4.24   # the sheet's column
    assert round(_ny(clean_panel(panel, ny_benchmark="NYSCRF"))["t"], 2) == 0.15   # the Fund against itself
    with pytest.raises(ValueError):
        clean_panel(panel, ny_benchmark="other")
    ext = pd.read_csv(ROOT / "data" / "panel_returns_extended.csv")
    ny = ext[ext["fund"] == "NYSLRS"].set_index("year")["benchmark"]
    assert all(ny[y] == pytest.approx(v, abs=1e-6) for y, v in NY_POLICY_BENCHMARK.items())
