import numpy as np
import pytest
from scipy import stats

import cis
from cis.core import log_excess, t_statistic, skill_from_returns, tracking_error, k_from_annualised
from cis.multiple import bh_q, bh_reject, holm_reject, bh_thresholds


def _returns(seed=0, n=25):
    rng = np.random.default_rng(seed)
    r_l = rng.normal(0.07, 0.11, n)
    r_p = r_l + rng.normal(0.003, 0.015, n)
    return r_p, r_l


def test_k_is_the_t_statistic_of_the_log_excess_return():
    """Proposition 1: sqrt(T) gap*/sigma_R* == sqrt(n) mean(d)/sd(d) to machine precision."""
    for seed in range(20):
        for h in (1.0, 12.0):
            r_p, r_l = _returns(seed, n=37)
            d = log_excess(r_p, r_l)
            n = d.size
            T = n / h
            gap_star = d.mean() * h                       # annualised mean log excess
            sigma_star = d.std(ddof=1) * np.sqrt(h)       # annualised sd of log excess
            k = k_from_annualised(T, gap_star, sigma_star)
            assert abs(k - t_statistic(d)) < 1e-10


def test_geometric_return_is_the_mean_log_return():
    """Equation (6): log(1+g) == (h/n) sum log(1+r_t), exactly."""
    r_p, _ = _returns(3, n=30)
    g = np.prod(1 + r_p) ** (1 / 30) - 1
    assert abs(np.log1p(g) - np.log1p(r_p).mean()) < 1e-12


def test_double_count_equals_half_the_variance_difference_to_first_order():
    s = skill_from_returns(*_returns(5, n=200))
    # gap_paper ~ gap_arith - ... : geometric minus arithmetic numerator differs by about -drag
    assert abs((s["gap_paper"] - s["gap_star"]) + s["drag"]) < 2e-3


def test_cis_is_one_minus_p():
    r_p, r_l = _returns(7)
    s = skill_from_returns(r_p, r_l)
    assert abs(s["cis_star"] - (1 - stats.norm.sf(s["t"]))) < 1e-12


def test_tracking_error_formula_matches_direct_sd_for_simple_excess():
    r_p, r_l = _returns(11, n=40)
    te_formula = tracking_error(r_p.std(ddof=1), r_l.std(ddof=1), np.corrcoef(r_p, r_l)[0, 1])
    assert abs(te_formula - (r_p - r_l).std(ddof=1)) < 1e-12


def test_bh_thresholds_for_the_published_panel():
    assert np.allclose(bh_thresholds(47, 0.10), [0.10 / 47, 0.20 / 47, 0.30 / 47])
    assert round(1 - 0.10 / 47, 4) == 0.9979


def test_bh_and_holm_on_a_known_vector():
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.2, 0.5])
    assert bh_reject(p, 0.05) == 2      # thresholds 0.0083, 0.0167, 0.025: 0.03 fails rank 3
    assert holm_reject(p, 0.05) == 2    # 0.01 <= 0.05/5, 0.03 > 0.05/4
    q = bh_q(p)
    assert q[0] == pytest.approx(0.006) and np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_published_appendix_reproduces_the_note():
    """The numbers in the note computed from the published column alone."""
    df = cis.load_published()
    assert len(df) == 47
    s = cis.summary(df)
    assert s["n_above_90"] == 13
    assert round(s["mean_cis"], 3) == 0.603
    assert round(s["ks_p"], 3) == 0.014
    assert round(s["expected_above_90_t"], 1) == 5.1
    mc = cis.model_consistent_from_appendix(df)
    assert round(mc.loc["NYSLRS", "K_star"], 2) == 3.68
    assert round(mc.loc["LASERS", "K_star"], 2) == -4.24
    assert round(mc.loc["NYSLRS", "q"], 3) == 0.058
    # face value: one outperformer and one underperformer at 10% FDR
    assert bh_reject(mc["p_pub"], 0.10) == 1
    assert bh_reject(df["cis"], 0.10) == 1
    assert bh_reject(mc["p_t"], 0.10) == 1
    drag = cis.implied_drag(df)
    assert len(drag) == 45
    assert set(drag.index[drag["drag"].abs() > drag["gap_star"].abs()]) == {
        "NC TERS", "NJ PERS", "NJ TPAF", "NYSTRS", "LAFPP"}


def test_dispersion_functionals_match_direct_computation():
    """cis.dispersion: the half-sample z is the Welch statistic; sum_t2 and sd_t agree with the t's."""
    import pandas as pd
    from cis.dispersion import half_masks, functionals
    rng = np.random.default_rng(1)
    years = np.arange(2000, 2020)
    V = rng.normal(0.0, 0.02, size=(20, 5))
    V[:6, 1] = np.nan                       # fund 1 starts later
    V[15:, 3] = np.nan                      # fund 3 ends earlier
    mat = pd.DataFrame(V, index=years, columns=list("abcde"))
    A, Bm = half_masks(mat)
    f = functionals(mat.values, A, Bm)
    for j in range(5):
        x = V[:, j][~np.isnan(V[:, j])]
        k = len(x) // 2
        a, b = x[:k], x[k:]
        z = (b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        assert abs(f["z"][j] - z) < 1e-12
        assert abs(f["t"][j] - np.sqrt(len(x)) * x.mean() / x.std(ddof=1)) < 1e-12
    assert abs(f["sum_t2"] - np.sum(f["t"] ** 2)) < 1e-12
    assert abs(f["sd_t"] - np.std(f["t"], ddof=1)) < 1e-12
    assert (A + Bm == (~np.isnan(V)).astype(float)).all()
