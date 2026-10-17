"""Panel-level tests under the dependence-aware null: is there skill anywhere, and does it change?

All statistics are functionals of the year x fund matrix of log excess returns, and their null
distributions come from the same year-cluster sign-flip bootstrap as `cis.bootstrap` (one random
sign per calendar year, common to all funds, after demeaning each fund), so dependence between
funds is respected throughout. The functionals:

  count      number of funds whose normal-reference confidence exceeds `threshold`
  max_t      largest t-statistic;   min_t  smallest
  sd_mean    cross-sectional standard deviation of the fund mean log excess returns
  sd_t       cross-sectional standard deviation of the t-statistics
  sum_t2     sum of squared t-statistics (an overall "is there skill anywhere" statistic)
  half_chi2  sum over funds of the squared z-statistic for second-half minus first-half mean
  max_half   largest |z| of that half-sample difference

The half-sample split divides each fund's own observations into its first floor(n/2) and the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["half_masks", "functionals", "null_functionals", "summarise"]


def half_masks(mat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrices (year x fund) marking each fund's first and second half of observations."""
    V = mat.values
    A = np.zeros_like(V)
    Bm = np.zeros_like(V)
    for j in range(V.shape[1]):
        ii = np.where(~np.isnan(V[:, j]))[0]
        k = len(ii) // 2
        A[ii[:k], j] = 1.0
        Bm[ii[k:], j] = 1.0
    return A, Bm


def _moments(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted (0/1) sample size, mean and unbiased variance down the columns; X may hold NaN."""
    Z = np.where(np.isnan(X), 0.0, X) * W
    n = W.sum(axis=0)
    mean = Z.sum(axis=0) / n
    var = ((Z - mean * W) ** 2).sum(axis=0) / (n - 1)
    return n, mean, var


def functionals(X: np.ndarray, A: np.ndarray, Bm: np.ndarray, threshold: float = 0.90) -> dict:
    """All statistics for one year x fund matrix X (NaN where a fund has no observation)."""
    n = np.sum(~np.isnan(X), axis=0)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    t = np.sqrt(n) * mu / sd
    na, ma, va = _moments(X, A)
    nb, mb, vb = _moments(X, Bm)
    z = (mb - ma) / np.sqrt(va / na + vb / nb)
    return dict(count=int(np.sum(stats.norm.cdf(t) > threshold)), max_t=t.max(), min_t=t.min(),
                sd_mean=np.std(mu, ddof=1), sd_t=np.std(t, ddof=1), sum_t2=float(np.sum(t ** 2)),
                half_chi2=float(np.sum(z ** 2)), max_half=float(np.abs(z).max()),
                t=t, z=z, mu=mu, se=sd / np.sqrt(n))


def null_functionals(mat: pd.DataFrame, B: int = 20000, seed: int | None = 0,
                     threshold: float = 0.90) -> pd.DataFrame:
    """Null distribution (B rows) of every scalar functional under the year-cluster sign flip."""
    rng = np.random.default_rng(seed)
    V = mat.values
    ny, m = V.shape
    D0 = V - np.nanmean(V, axis=0)
    A, Bm = half_masks(mat)
    keys = ["count", "max_t", "min_t", "sd_mean", "sd_t", "sum_t2", "half_chi2", "max_half"]
    out = np.empty((B, len(keys)))
    for b in range(B):
        s = rng.choice([-1.0, 1.0], size=(ny, 1))
        f = functionals(D0 * s, A, Bm, threshold)
        out[b] = [f[k] for k in keys]
    return pd.DataFrame(out, columns=keys)


def summarise(mat: pd.DataFrame, null: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Observed value, null mean, null root-mean-square, 95th (5th for min_t) percentile, and
    P(null at least as extreme) per statistic. For sd_mean, observed^2 - null_rms^2 is the
    method-of-moments estimate of the variance of true mean excess returns across funds."""
    A, Bm = half_masks(mat)
    obs = functionals(mat.values, A, Bm, threshold)
    rows = []
    for k in null.columns:
        o = obs[k]
        if k == "min_t":
            p = float((null[k] <= o).mean())
            q = float(np.percentile(null[k], 5))
        else:
            p = float((null[k] >= o).mean())
            q = float(np.percentile(null[k], 95))
        rows.append(dict(statistic=k, observed=o, null_mean=float(null[k].mean()),
                         null_rms=float(np.sqrt(np.mean(null[k] ** 2))), null_q=q, p=p))
    return pd.DataFrame(rows).set_index("statistic")
