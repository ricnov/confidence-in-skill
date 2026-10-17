"""Core statistics of the Ambarish–Seigel / Muralidhar–U "confidence in skill" measure.

Model: portfolio P and benchmark L follow correlated geometric Brownian motions. The
relative value R = P/L has log drift gap = (mu_P - sigma_P^2/2) - (mu_L - sigma_L^2/2)
and diffusion sigma_R^2 = sigma_P^2 + sigma_L^2 - 2 rho sigma_P sigma_L (the squared
tracking error). The paper's statistic is

    K   = sqrt(T) * gap / sigma_R,
    CIS = Phi(K).

With the numerator and denominator taken as their sample counterparts on the log
excess return d_t = log(1 + r_P,t) - log(1 + r_L,t), K is exactly the one-sample
t-statistic sqrt(n) * mean(d) / sd(d), so CIS = 1 - p for a one-sided test evaluated
with the normal reference distribution. `t_statistic` and `k_from_annualised` make
that identity explicit; `test_core.py` pins it to 1e-10.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "tracking_error",
    "k_from_annualised",
    "cis_from_k",
    "k_from_cis",
    "log_excess",
    "t_statistic",
    "p_value",
    "skill_from_returns",
]


def tracking_error(sigma_p: float, sigma_l: float, rho: float) -> float:
    """Equation (7) of the paper: sigma_R^2 = sigma_P^2 + sigma_L^2 - 2 rho sigma_P sigma_L."""
    v = sigma_p**2 + sigma_l**2 - 2.0 * rho * sigma_p * sigma_l
    return float(np.sqrt(max(v, 0.0)))


def k_from_annualised(T: float, gap: float, sigma_r: float) -> float:
    """Equation (9b): K = sqrt(T) * gap / sigma_R, with T in years and annualised inputs."""
    return float(np.sqrt(T) * gap / sigma_r)


def cis_from_k(k, reference: str = "normal", df: float | None = None):
    """Confidence in skill: Phi(K) (the paper) or the Student-t CDF with `df` degrees of freedom."""
    if reference == "normal":
        return stats.norm.cdf(k)
    if reference == "t":
        if df is None:
            raise ValueError("df is required for the t reference distribution")
        return stats.t.cdf(k, df=df)
    raise ValueError(f"unknown reference distribution {reference!r}")


def k_from_cis(cis, reference: str = "normal", df: float | None = None):
    """Invert `cis_from_k`; used to recover the gap implied by a published confidence value."""
    if reference == "normal":
        return stats.norm.ppf(cis)
    if reference == "t":
        return stats.t.ppf(cis, df=df)
    raise ValueError(f"unknown reference distribution {reference!r}")


def log_excess(r_p, r_l) -> np.ndarray:
    """d_t = log(1 + r_P,t) - log(1 + r_L,t) for simple periodic returns."""
    r_p = np.asarray(r_p, dtype=float)
    r_l = np.asarray(r_l, dtype=float)
    return np.log1p(r_p) - np.log1p(r_l)


def t_statistic(d) -> float:
    """One-sample t-statistic sqrt(n) * mean(d) / sd(d) with the sample (ddof=1) standard deviation."""
    d = np.asarray(d, dtype=float)
    n = d.size
    return float(np.sqrt(n) * d.mean() / d.std(ddof=1))


def p_value(t: float, n: int, side: str = "upper", reference: str = "t") -> float:
    """One-sided p-value of the t-statistic. `side='upper'` tests skill, `'lower'` tests lack of it."""
    if reference == "t":
        sf = stats.t.sf(t, df=n - 1)
        cdf = stats.t.cdf(t, df=n - 1)
    elif reference == "normal":
        sf, cdf = stats.norm.sf(t), stats.norm.cdf(t)
    else:
        raise ValueError(reference)
    return float(sf if side == "upper" else cdf)


def skill_from_returns(r_p, r_l, periods_per_year: float = 1.0) -> dict:
    """Every version of the statistic for one fund, from its periodic simple returns.

    Returns a dict with the paper's statistic as implemented (annualised geometric returns
    minus sigma^2/2), the arithmetic-mean version the derivation implies for mu, and the
    model-consistent t-statistic on the log excess series, together with the drag term
    (sigma_P^2 - sigma_L^2)/2 that separates the first two.
    """
    r_p = np.asarray(r_p, dtype=float)
    r_l = np.asarray(r_l, dtype=float)
    n = r_p.size
    T = n / periods_per_year
    h = periods_per_year
    g_p = np.prod(1 + r_p) ** (1 / T) - 1          # annualised geometric return
    g_l = np.prod(1 + r_l) ** (1 / T) - 1
    s_p = r_p.std(ddof=1) * np.sqrt(h)
    s_l = r_l.std(ddof=1) * np.sqrt(h)
    rho = float(np.corrcoef(r_p, r_l)[0, 1]) if n > 2 else np.nan
    te = tracking_error(s_p, s_l, rho)
    gap_paper = (g_p - s_p**2 / 2) - (g_l - s_l**2 / 2)
    gap_arith = (r_p.mean() * h - s_p**2 / 2) - (r_l.mean() * h - s_l**2 / 2)
    d = log_excess(r_p, r_l)
    t = t_statistic(d)
    k_paper = k_from_annualised(T, gap_paper, te)
    k_arith = k_from_annualised(T, gap_arith, te)
    return dict(
        n=n, T=T, g_p=g_p, g_l=g_l, sigma_p=s_p, sigma_l=s_l, rho=rho, tracking_error=te,
        gap_paper=gap_paper, k_paper=k_paper, cis_paper=float(stats.norm.cdf(k_paper)),
        gap_arith=gap_arith, k_arith=k_arith, cis_arith=float(stats.norm.cdf(k_arith)),
        gap_star=float(d.mean() * h), sigma_star=float(d.std(ddof=1) * np.sqrt(h)),
        drag=float((s_p**2 - s_l**2) / 2),
        t=t, cis_star=float(stats.norm.cdf(t)),
        p_t=p_value(t, n, "upper", "t"), p_t_under=p_value(t, n, "lower", "t"),
    )
