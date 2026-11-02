import numpy as np
import pytest

from ashdisperse.spectral.cheb import (cheb_deriv, cheb_eval, cheb_fit,
                                       cheb_lobatto, standard_chop,
                                       truncateCoeffs)


def test_cheb_fit_eval_roundtrip_polynomial():
    x = cheb_lobatto(9)
    f = 3 * x**4 - x**2 + 0.5 * x - 2
    c = cheb_fit(f)
    xt = np.linspace(-1, 1, 101)
    assert np.allclose(cheb_eval(c, xt), 3 * xt**4 - xt**2 + 0.5 * xt - 2, atol=1e-13)


def test_cheb_deriv_exact_for_polynomial():
    x = cheb_lobatto(12)
    c = cheb_fit(np.sin(x) * 0 + x**5 - 2 * x**3)
    dc = cheb_deriv(c)
    xt = np.linspace(-1, 1, 51)
    assert np.allclose(cheb_eval(dc, xt), 5 * xt**4 - 6 * xt**2, atol=1e-12)


def test_cheb_fit_smooth_function_converges():
    x = cheb_lobatto(33)
    c = cheb_fit(np.exp(x) * np.cos(3 * x))
    xt = np.linspace(-1, 1, 201)
    assert np.max(np.abs(cheb_eval(c, xt) - np.exp(xt) * np.cos(3 * xt))) < 1e-13


@pytest.mark.parametrize("rate", [0.5, 0.8])
def test_standard_chop_geometric_series(rate):
    # geometric decay to a noise floor of 1e-16: resolved
    n = 256
    a = np.maximum(rate ** np.arange(n), 1e-16 * np.abs(np.sin(np.arange(n)) + 1.1))
    cut = standard_chop(a, 1e-10)
    expected = int(np.log(1e-10) / np.log(rate))
    assert cut < n
    assert expected - 5 <= cut <= expected + 40


def test_standard_chop_algebraic_tail_not_resolved():
    n = 256
    a = 1.0 / (1 + np.arange(n)) ** 2  # reaches only ~1.5e-5
    assert standard_chop(a, 1e-10) == n


def test_standard_chop_short_series_not_resolved():
    assert standard_chop(np.array([1.0, 1e-20, 1e-20]), 1e-10) == 3


def test_truncate_coeffs_complex():
    n = 128
    c = (0.5 ** np.arange(n)) * (1 + 1j)
    t = truncateCoeffs(c.astype(np.complex128), 1e-10, 10.0)
    assert 30 <= t.size < n
    assert np.all(t == c[: t.size])


def test_solve_inplace_matches_numpy():
    from ashdisperse.spectral.lapack import solve_inplace
    rng = np.random.default_rng(3)
    for N, nrhs in ((5, 1), (33, 2), (64, 3)):
        L = rng.random((N, N)) + 1j * rng.random((N, N)) + N * np.eye(N)
        B = rng.random((nrhs, N)) + 1j * rng.random((nrhs, N))
        X = B.copy()
        solve_inplace(L.copy(), X)
        assert np.allclose(L @ X.T, B.T, rtol=0, atol=1e-12)


def test_solve_inplace_singular():
    from ashdisperse.spectral.lapack import solve_inplace
    L = np.zeros((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        solve_inplace(L, np.ones((1, 4), dtype=np.complex128))
