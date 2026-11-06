"""The anytime-valid confidence: scale invariance, the martingale identity E[M_n] = 1, and the running maximum."""
import numpy as np
from scipy import stats, integrate

from cis.anytime import e_process, confidence_path, naive_path


def test_scale_invariance():
    x = np.random.default_rng(0).standard_normal(15) + 0.3
    one, two = e_process(x, a=0.5)
    one7, two7 = e_process(7.0 * x, a=0.5)
    assert np.allclose(one[1:], one7[1:]) and np.allclose(two[1:], two7[1:])


def test_two_sided_has_unit_expectation_under_the_null():
    # Under mu = 0, B_n^2 / A_n = n Z with Z ~ Beta(1/2, (n-1)/2), so E[(1 - c Z)^(-n/2)] = (1 - c)^(-1/2) with
    # c = n a^2 / (1 + n a^2), and E[M_n] = (1 + n a^2)^(-1/2) (1 - c)^(-1/2) = 1. Check by quadrature.
    for n, a in ((5, 0.5), (16, 0.3), (27, 1.0)):
        c = n * a ** 2 / (1 + n * a ** 2)
        val, _ = integrate.quad(lambda z: (1 - c * z) ** (-n / 2) * stats.beta.pdf(z, 0.5, (n - 1) / 2), 0, 1, limit=200)
        EM = (1 + n * a ** 2) ** -0.5 * val
        assert abs(EM - 1) < 1e-6, (n, a, EM)


def test_one_sided_has_unit_expectation_by_simulation():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6000, 8))
    EM = np.mean([e_process(X[i], a=0.5)[0][-1] for i in range(len(X))])
    assert abs(EM - 1) < 0.08, EM


def test_confidence_is_monotone_and_bounded():
    x = np.random.default_rng(2).standard_normal(20) + 0.6
    one, _ = e_process(x)
    C = confidence_path(one)
    C = C[~np.isnan(C)]
    assert np.all(np.diff(C) >= -1e-12) and C.min() >= 0 and C.max() < 1
    assert np.isnan(naive_path(x)[0]) and naive_path(x)[-1] == stats.norm.cdf(np.sqrt(20) * x.mean() / x.std(ddof=1))


def test_closed_form_matches_quadrature():
    """The closed form (Student-t cdf) agrees with the chi quadrature to the quadrature's accuracy."""
    from cis.anytime import e_process_quadrature
    rng = np.random.default_rng(5)
    for a in (0.25, 0.5, 1.0, (0.25, 0.5, 1.0)):
        x = rng.normal(0.4, 1.3, 30)
        one, two = e_process(x, a=a)
        q1, q2 = e_process_quadrature(x, a=a)
        assert np.nanmax(np.abs(one / q1 - 1)) < 1e-5
        assert np.nanmax(np.abs(two / q2 - 1)) < 1e-10
