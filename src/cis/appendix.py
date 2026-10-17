"""The published cross-section: Appendix III of Muralidhar (2026), 47 legible rows.

The appendix reports, per fund, the start year, the number of years, the annualised
return, the excess return over benchmark, the tracking error and the confidence in skill
rounded to whole percentage points. It does not report sigma_P, sigma_L or rho, so the
paper's own statistic cannot be evaluated forward; everything here works from the
published columns alone.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .core import k_from_cis, cis_from_k
from .multiple import bh_q

__all__ = ["load_published", "model_consistent_from_appendix", "implied_drag", "summary"]

_DATA = Path(__file__).resolve().parents[2] / "data" / "appendix_III_published.csv"


def load_published(path: str | Path | None = None) -> pd.DataFrame:
    """Load the transcribed appendix (fund, start, years, ann_return, excess, tracking_error, cis)."""
    df = pd.read_csv(path or _DATA).set_index("fund")
    return df


def model_consistent_from_appendix(df: pd.DataFrame) -> pd.DataFrame:
    """K* = sqrt(T) gap*/sigma_R with gap* = log(1+g_P) - log(1+g_L) from the appendix's own columns.

    Adds columns K_star, CIS_star, p_t (one-sided Student-t, T-1 df), q (BH over all rows)
    and the published-column p-value p_pub = 1 - CIS.
    """
    out = df.copy()
    T = out["years"].astype(float)
    g_p = out["ann_return"]
    g_l = g_p - out["excess"]
    gap_star = np.log1p(g_p) - np.log1p(g_l)
    out["gap_star"] = gap_star
    out["K_star"] = np.sqrt(T) * gap_star / out["tracking_error"]
    out["CIS_star"] = cis_from_k(out["K_star"])
    out["p_t"] = stats.t.sf(out["K_star"], df=T - 1)
    out["q"] = bh_q(out["p_t"].values)
    out["p_pub"] = 1 - out["cis"]
    return out


def implied_drag(df: pd.DataFrame) -> pd.DataFrame:
    """Recover the drag term the paper applied, fund by fund, by inverting the published confidence.

    gap_impl = Phi^{-1}(CIS) sigma_R / sqrt(T); drag = gap* - gap_impl = (sigma_P^2 - sigma_L^2)/2
    as applied. Rows censored at 0% or 100% are dropped.
    """
    d = model_consistent_from_appendix(df)
    d = d[(d["cis"] > 0) & (d["cis"] < 1)].copy()
    d["gap_impl"] = k_from_cis(d["cis"]) * d["tracking_error"] / np.sqrt(d["years"].astype(float))
    d["drag"] = d["gap_star"] - d["gap_impl"]
    d["variance_diff"] = 2 * d["drag"]
    d["drag_share"] = (d["drag"].abs() / d["gap_star"].abs()).replace(np.inf, np.nan)
    return d


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers of the cross-section under the global null of no skill."""
    cis = df["cis"].values
    m = cis.size
    T = df["years"].astype(float).values
    expected_t = stats.t.sf(stats.norm.ppf(0.90), df=T - 1).sum()
    return dict(
        n_funds=int(m),
        n_above_90=int((cis > 0.90).sum()),
        expected_above_90_normal=0.10 * m,
        expected_above_90_t=float(expected_t),
        mean_cis=float(cis.mean()),
        ks_p=float(stats.kstest(np.clip(cis, 1e-9, 1 - 1e-9), "uniform").pvalue),
        binomial_p_13=float(stats.binom.sf(12, m, 0.10)),
    )
