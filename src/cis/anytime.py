"""The anytime-valid confidence in skill (paper, Section 5).

For annual log excess returns d_1, ..., d_n i.i.d. N(mu, sigma^2), both parameters unknown, the direction
u_n = d_{1:n}/||d_{1:n}|| has a law under H0: mu = 0 that does not depend on sigma. The likelihood ratio of u_n
under an information ratio delta = mu/sigma against delta = 0, averaged over a half-normal prior
delta ~ |N(0, a^2)|, is

    M_n = 2 (1 + n a^2)^{-1/2} E_R[ exp(kappa^2 R^2 / 2) Phi(kappa R) ],   kappa = a B_n / (sqrt(A_n) sqrt(1 + n a^2)),

with A_n = sum d_i^2, B_n = sum d_i and R ~ chi_n. Since u_{n-1} is a function of u_n, (M_n) is a nonnegative
martingale with M_0 = 1 under H0 whatever sigma, and Ville's inequality gives P(sup_n M_n >= 1/alpha) <= alpha.
The two-sided version (full normal prior) is (1 + n a^2)^{-1/2} (1 - kappa^2)^{-n/2}. An average of e-processes
over several prior scales is an e-process; the paper uses the equal-weight mixture over a = 0.25, 0.5, 1.
"""
import numpy as np
from scipy import stats
from scipy.special import gammaln

DEFAULT_SCALES = (0.25, 0.5, 1.0)
_R = np.linspace(1e-4, 16.0, 4001)   # quadrature grid for the chi variable


def e_process(x, a=DEFAULT_SCALES):
    """One-sided (delta > 0) and two-sided e-process values M_2..M_n for the sequence x (NaN before n = 2).

    a is one prior scale or a tuple of scales; a tuple gives the equal-weight mixture."""
    if np.ndim(a) > 0 and len(a) > 1:
        ones, twos = zip(*(e_process(x, a=float(a_)) for a_ in a))
        return np.mean(np.vstack(ones), axis=0), np.mean(np.vstack(twos), axis=0)
    a = float(np.ravel(a)[0])
    x = np.asarray(x, float)
    n_all = len(x)
    one = np.full(n_all, np.nan)
    two = np.full(n_all, np.nan)
    A = np.cumsum(x ** 2)
    B = np.cumsum(x)
    for n in range(2, n_all + 1):
        kap = a * B[n - 1] / (np.sqrt(A[n - 1]) * np.sqrt(1 + n * a ** 2))
        logpdf = (n - 1) * np.log(_R) - _R ** 2 / 2 - ((n / 2 - 1) * np.log(2) + gammaln(n / 2))
        w = np.exp(logpdf + kap ** 2 * _R ** 2 / 2)
        one[n - 1] = 2 / np.sqrt(1 + n * a ** 2) * np.trapezoid(w * stats.norm.cdf(kap * _R), _R)
        two[n - 1] = (1 + n * a ** 2) ** -0.5 * (1 - kap ** 2) ** (-n / 2)
    return one, two


def confidence_path(M):
    """Anytime-valid confidence C_n = max(0, 1 - 1 / max_{k <= n} M_k); NaN where M is undefined."""
    M = np.asarray(M, float)
    Mstar = np.fmax.accumulate(np.where(np.isnan(M), -np.inf, M))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(Mstar > 1, 1 - 1 / Mstar, 0.0)
    C[np.isnan(M) & ~(Mstar > 1)] = np.nan
    return C


def naive_path(x):
    """The framework's confidence Phi(t_n) on the expanding window (NaN before n = 2)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for n in range(2, len(x) + 1):
        m = x[:n].mean(); s = x[:n].std(ddof=1)
        out[n - 1] = stats.norm.cdf(np.sqrt(n) * m / s) if s > 0 else np.nan
    return out


def first_cross(path, level, start_index=0):
    """Index of the first element at or above level, at or after start_index; None if never."""
    idx = np.where(np.nan_to_num(np.asarray(path, float)[start_index:], nan=-1) >= level)[0]
    return int(idx[0] + start_index) if len(idx) else None
