"""
Unit tests for factor covariance model functions (src/factor_models.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.factor_models import (
    estimate_factor_betas,
    estimate_gjr_garch,
    estimate_symmetric_garch,
    factor_arch_covariance,
    factor_dcc_covariance,
    factor_double_arch_covariance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_returns(T: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.01


def make_market(T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T) * 0.012


# ---------------------------------------------------------------------------
# estimate_gjr_garch tests
# ---------------------------------------------------------------------------

def test_gjr_garch_shape():
    r = make_market(300)
    h, s = estimate_gjr_garch(r)
    assert h.shape == (300,)
    assert s.shape == (300,)


def test_gjr_garch_positive_variance():
    r = make_market(300)
    h, _ = estimate_gjr_garch(r)
    assert np.all(h > 0), "Conditional variances must be positive"


def test_gjr_garch_no_nan():
    r = make_market(300)
    h, s = estimate_gjr_garch(r)
    assert not np.any(np.isnan(h))
    assert not np.any(np.isnan(s))


def test_gjr_garch_standardized_residuals_unit_std():
    """Standardized residuals should have approximate unit variance."""
    r = make_market(1000)
    _, s = estimate_gjr_garch(r)
    assert 0.5 < np.std(s) < 1.5, f"Std of standardized residuals = {np.std(s):.3f}"


def test_gjr_garch_dtype():
    r = make_market(200)
    h, s = estimate_gjr_garch(r)
    assert h.dtype == np.float64
    assert s.dtype == np.float64


# ---------------------------------------------------------------------------
# estimate_symmetric_garch tests
# ---------------------------------------------------------------------------

def test_symmetric_garch_shape():
    r = make_market(300)
    h, s = estimate_symmetric_garch(r)
    assert h.shape == (300,)
    assert s.shape == (300,)


def test_symmetric_garch_positive_variance():
    r = make_market(300)
    h, _ = estimate_symmetric_garch(r)
    assert np.all(h > 0)


def test_symmetric_garch_no_nan():
    r = make_market(300)
    h, s = estimate_symmetric_garch(r)
    assert not np.any(np.isnan(h))
    assert not np.any(np.isnan(s))


# ---------------------------------------------------------------------------
# estimate_factor_betas tests
# ---------------------------------------------------------------------------

def test_estimate_factor_betas_shape():
    n = 5
    T = 300
    stock_ret = make_returns(T, n)
    mkt = make_market(T)
    betas, resid = estimate_factor_betas(stock_ret, mkt)
    assert betas.shape == (n,)
    assert resid.shape == (T, n)


def test_estimate_factor_betas_residuals_orthogonal():
    """Residuals must be orthogonal to market (OLS property)."""
    n = 3
    T = 400
    rng = np.random.default_rng(0)
    mkt = rng.standard_normal(T) * 0.01
    stock_ret = np.outer(mkt, [0.8, 1.0, 1.2]) + rng.standard_normal((T, n)) * 0.005
    betas, resid = estimate_factor_betas(stock_ret, mkt)
    for i in range(n):
        corr = np.corrcoef(mkt, resid[:, i])[0, 1]
        assert abs(corr) < 0.02, f"Residual {i} correlated with market: {corr:.4f}"


def test_estimate_factor_betas_known_beta():
    """Known betas must be recovered approximately."""
    T = 2000
    rng = np.random.default_rng(5)
    mkt = rng.standard_normal(T) * 0.01
    true_betas = np.array([0.8, 1.2, 1.5])
    noise = rng.standard_normal((T, 3)) * 0.003
    stock_ret = np.outer(mkt, true_betas) + noise
    betas, _ = estimate_factor_betas(stock_ret, mkt)
    np.testing.assert_allclose(betas, true_betas, atol=0.05,
                               err_msg="Estimated betas far from true betas")


# ---------------------------------------------------------------------------
# factor_arch_covariance tests
# ---------------------------------------------------------------------------

def test_factor_arch_covariance_shape():
    T, n = 200, 5
    betas = np.ones(n) * 1.0
    h_market = np.ones(T) * 1e-4
    idio_var = np.ones(n) * 5e-5
    H = factor_arch_covariance(betas, h_market, idio_var)
    assert H.shape == (T, n, n)


def test_factor_arch_covariance_positive_diagonal():
    T, n = 100, 4
    betas = np.array([0.8, 1.0, 1.2, 0.9])
    h_market = np.ones(T) * 1e-4
    idio_var = np.ones(n) * 1e-5
    H = factor_arch_covariance(betas, h_market, idio_var)
    for t in range(T):
        assert np.all(np.diag(H[t]) > 0), f"Non-positive diagonal at t={t}"


def test_factor_arch_covariance_symmetry():
    T, n = 50, 3
    betas = np.array([0.8, 1.0, 1.2])
    h_market = np.ones(T) * 1e-4
    idio_var = np.ones(n) * 1e-5
    H = factor_arch_covariance(betas, h_market, idio_var)
    for t in range(T):
        np.testing.assert_allclose(H[t], H[t].T, atol=1e-12, err_msg=f"Not symmetric at t={t}")


# ---------------------------------------------------------------------------
# factor_double_arch_covariance tests
# ---------------------------------------------------------------------------

def test_factor_double_arch_shape():
    T, n = 200, 5
    betas = np.ones(n) * 1.0
    h_market = np.ones(T) * 1e-4
    h_idio = np.ones((T, n)) * 5e-5
    H = factor_double_arch_covariance(betas, h_market, h_idio)
    assert H.shape == (T, n, n)


def test_factor_double_arch_positive_diagonal():
    T, n = 100, 3
    betas = np.array([1.0, 1.2, 0.9])
    h_market = np.ones(T) * 1e-4
    h_idio = np.ones((T, n)) * 1e-5
    H = factor_double_arch_covariance(betas, h_market, h_idio)
    for t in range(T):
        assert np.all(np.diag(H[t]) > 0)


def test_factor_double_arch_symmetry():
    T, n = 50, 3
    betas = np.array([0.8, 1.0, 1.2])
    h_market = np.ones(T) * 1e-4
    h_idio = np.ones((T, n)) * 1e-5
    H = factor_double_arch_covariance(betas, h_market, h_idio)
    for t in range(T):
        np.testing.assert_allclose(H[t], H[t].T, atol=1e-12)


# ---------------------------------------------------------------------------
# factor_dcc_covariance tests
# ---------------------------------------------------------------------------

def test_factor_dcc_shape():
    T, n = 100, 3
    betas = np.array([0.8, 1.0, 1.2])
    h_market = np.ones(T) * 1e-4
    h_idio = np.ones((T, n)) * 1e-5
    # Identity residual correlation matrices (n+1) x (n+1)
    Rt_resid = np.zeros((T, n + 1, n + 1))
    for t in range(T):
        Rt_resid[t] = np.eye(n + 1)
    H = factor_dcc_covariance(betas, h_market, h_idio, Rt_resid)
    assert H.shape == (T, n, n)


def test_factor_dcc_symmetry():
    T, n = 50, 3
    betas = np.array([0.8, 1.0, 1.2])
    h_market = np.ones(T) * 1e-4
    h_idio = np.ones((T, n)) * 1e-5
    Rt_resid = np.zeros((T, n + 1, n + 1))
    for t in range(T):
        Rt_resid[t] = np.eye(n + 1)
    H = factor_dcc_covariance(betas, h_market, h_idio, Rt_resid)
    for t in range(T):
        np.testing.assert_allclose(H[t], H[t].T, atol=1e-10, err_msg=f"Not symmetric at t={t}")


def test_factor_dcc_no_nan():
    T, n = 100, 3
    betas = np.array([0.8, 1.0, 1.2])
    h_market = np.ones(T) * 1e-4
    h_idio = np.ones((T, n)) * 1e-5
    Rt_resid = np.zeros((T, n + 1, n + 1))
    for t in range(T):
        Rt_resid[t] = np.eye(n + 1)
    H = factor_dcc_covariance(betas, h_market, h_idio, Rt_resid)
    assert not np.any(np.isnan(H)), "NaN found in Factor DCC covariance"
