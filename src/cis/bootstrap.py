"""Dependence-aware inference for the cross-section: a year-cluster wild bootstrap.

Under the global null of no skill anywhere, each calendar year's cross-section of demeaned
log excess returns is multiplied by one random sign common to all funds in that year
(Rademacher weights). This imposes the null, keeps every fund's sample size and the
cross-sectional dependence intact, and avoids the duplicated-year artefacts that a
resampling bootstrap produces on short annual series.

The outputs are the null distribution of (i) the number of funds above a confidence
threshold and (ii) the maximum and minimum t-statistic, from which single-step and
step-down (Westfall–Young) adjusted p-values follow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["observed_t", "wild_bootstrap", "adjusted_p"]


def observed_t(mat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-fund t-statistics and sample sizes from a year x fund matrix with NaNs."""
    V = mat.values
    n = np.sum(~np.isnan(V), axis=0)
    mu = np.nanmean(V, axis=0)
    sd = np.nanstd(V, axis=0, ddof=1)
    return np.sqrt(n) * mu / sd, n


def wild_bootstrap(mat: pd.DataFrame, B: int = 20000, threshold: float = 0.90,
                   seed: int | None = 0) -> dict:
    """Null distribution of the count above `threshold` and of the extreme t-statistics.

    Returns a dict with arrays `count_normal`, `count_t`, `max_t`, `min_t`, the matrix `T`
    (B x m) of bootstrap t-statistics, and the observed `t` and `n`.
    """
    rng = np.random.default_rng(seed)
    V = mat.values
    ny, m = V.shape
    n = np.sum(~np.isnan(V), axis=0)
    D0 = V - np.nanmean(V, axis=0)                    # impose zero skill fund by fund
    T = np.empty((B, m))
    count_normal = np.empty(B, dtype=int)
    count_t = np.empty(B, dtype=int)
    for b in range(B):
        s = rng.choice([-1.0, 1.0], size=(ny, 1))
        Db = D0 * s
        mu = np.nanmean(Db, axis=0)
        sd = np.nanstd(Db, axis=0, ddof=1)
        tb = np.sqrt(n) * mu / sd
        T[b] = tb
        count_normal[b] = np.sum(stats.norm.cdf(tb) > threshold)
        count_t[b] = np.sum(stats.t.cdf(tb, df=n - 1) > threshold)
    t_obs, _ = observed_t(mat)
    return dict(T=T, count_normal=count_normal, count_t=count_t,
                max_t=T.max(axis=1), min_t=T.min(axis=1), t=t_obs, n=n,
                funds=list(mat.columns))


def adjusted_p(boot: dict) -> pd.DataFrame:
    """Single-step (max-t) and step-down adjusted p-values for every fund, both sides."""
    T, t = boot["T"], boot["t"]
    maxt, mint = boot["max_t"], boot["min_t"]
    m = t.size
    single_up = np.array([(maxt >= t[i]).mean() for i in range(m)])
    single_dn = np.array([(mint <= t[i]).mean() for i in range(m)])
    # step-down on the outperformance side (Westfall–Young)
    order = np.argsort(-t)
    step_up = np.empty(m)
    prev = 0.0
    for r, i in enumerate(order):
        sub = order[r:]
        p = (T[:, sub].max(axis=1) >= t[i]).mean()
        prev = max(prev, p)
        step_up[i] = prev
    return pd.DataFrame({"t": t, "n": boot["n"], "p_single_up": single_up,
                         "p_stepdown_up": step_up, "p_single_down": single_dn},
                        index=boot["funds"]).sort_values("t", ascending=False)
