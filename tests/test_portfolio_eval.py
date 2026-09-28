"""
Unit tests for portfolio evaluation functions (src/portfolio_eval.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.portfolio_eval import (
    compute_pair_realized_vol,
    evaluate_all_pairs,
    hedge_ratio,
    min_variance_weights,
)


# ---------------------------------------------------------------------------
# min_variance_weights tests
# ---------------------------------------------------------------------------

def test_min_variance_weights_equal_var_zero_cov():
    """When h_ii = h_jj = 1 and h_ij = 0, weight must be 0.5."""
    w = min_variance_weights(1.0, 1.0, 0.0)
    assert abs(w - 0.5) < 1e-10, f"Expected 0.5, got {w}"


def test_min_variance_weights_zero_denom_fallback():
    """Denominator = 0 must return fallback 0.5."""
    w = min_variance_weights(1.0, 1.0, 1.0)  # 1+1-2*1 = 0
    assert w == 0.5


def test_min_variance_weights_formula():
    """Verify the formula w = (h_jj - h_ij) / (h_ii + h_jj - 2*h_ij)."""
    h_ii, h_jj, h_ij = 2.0, 4.0, 1.0
    expected = (h_jj - h_ij) / (h_ii + h_jj - 2 * h_ij)  # (4-1)/(2+4-2) = 3/4 = 0.75
    w = min_variance_weights(h_ii, h_jj, h_ij)
    assert abs(w - expected) < 1e-10


def test_min_variance_weights_negative_cov():
    """Negative covariance is allowed (short position possible)."""
    w = min_variance_weights(1.0, 1.0, -0.5)
    # denom = 1+1-2*(-0.5) = 3; w = (1-(-0.5))/3 = 0.5
    expected = (1.0 - (-0.5)) / (1.0 + 1.0 - 2 * (-0.5))
    assert abs(w - expected) < 1e-10


# ---------------------------------------------------------------------------
# hedge_ratio tests
# ---------------------------------------------------------------------------

def test_hedge_ratio_basic():
    assert abs(hedge_ratio(0.5, 1.0) - 0.5) < 1e-10


def test_hedge_ratio_zero():
    assert abs(hedge_ratio(0.0, 1.0) - 0.0) < 1e-10


def test_hedge_ratio_zero_denom():
    """h_jj near zero must return 0 (fallback)."""
    assert hedge_ratio(0.5, 0.0) == 0.0


def test_hedge_ratio_formula():
    h_ij, h_jj = 3.0, 6.0
    assert abs(hedge_ratio(h_ij, h_jj) - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# compute_pair_realized_vol tests
# ---------------------------------------------------------------------------

def test_compute_pair_realized_vol_returns_dict():
    T = 200
    rng = np.random.default_rng(0)
    ri = rng.standard_normal(T) * 0.01
    rj = rng.standard_normal(T) * 0.01
    H_series = np.zeros((T, 2, 2))
    H_series[:, 0, 0] = 1e-4
    H_series[:, 1, 1] = 1e-4
    H_series[:, 0, 1] = 0.5e-4
    H_series[:, 1, 0] = 0.5e-4
    result = compute_pair_realized_vol(ri, rj, H_series, (0, 1))
    assert "minvar_vol" in result
    assert "hedge_vol" in result


def test_compute_pair_realized_vol_non_negative():
    T = 200
    rng = np.random.default_rng(1)
    ri = rng.standard_normal(T) * 0.01
    rj = rng.standard_normal(T) * 0.01
    H_series = np.zeros((T, 2, 2))
    H_series[:, 0, 0] = 1e-4
    H_series[:, 1, 1] = 1e-4
    H_series[:, 0, 1] = 0.0
    H_series[:, 1, 0] = 0.0
    result = compute_pair_realized_vol(ri, rj, H_series, (0, 1))
    assert result["minvar_vol"] >= 0
    assert result["hedge_vol"] >= 0


def test_compute_pair_realized_vol_diagonal_h():
    """With diagonal H (zero covariance), minvar weight = 0.5."""
    T = 100
    rng = np.random.default_rng(2)
    ri = rng.standard_normal(T) * 0.01
    rj = rng.standard_normal(T) * 0.01
    H_series = np.zeros((T, 2, 2))
    H_series[:, 0, 0] = 1e-4
    H_series[:, 1, 1] = 1e-4  # equal variances, zero cov -> w = 0.5
    result = compute_pair_realized_vol(ri, rj, H_series, (0, 1))
    # minvar portfolio = 0.5*ri + 0.5*rj
    expected_vol = float(np.std(0.5 * ri + 0.5 * rj, ddof=1))
    assert abs(result["minvar_vol"] - expected_vol) < 1e-10


# ---------------------------------------------------------------------------
# evaluate_all_pairs tests
# ---------------------------------------------------------------------------

def test_evaluate_all_pairs_count():
    n, T = 4, 200
    rng = np.random.default_rng(5)
    returns = rng.standard_normal((T, n)) * 0.01
    H_series = np.zeros((T, n, n))
    for t in range(T):
        H_series[t] = np.eye(n) * 1e-4
    result = evaluate_all_pairs(returns, H_series)
    n_pairs = n * (n - 1) // 2
    assert len(result["minvar_vols"]) == n_pairs
    assert len(result["hedge_vols"]) == n_pairs


def test_evaluate_all_pairs_averages():
    n, T = 3, 200
    rng = np.random.default_rng(6)
    returns = rng.standard_normal((T, n)) * 0.01
    H_series = np.zeros((T, n, n))
    for t in range(T):
        H_series[t] = np.eye(n) * 1e-4
    result = evaluate_all_pairs(returns, H_series)
    # avg should equal mean of individual pair vols
    np.testing.assert_allclose(result["avg_minvar_vol"], np.mean(result["minvar_vols"]), atol=1e-12)
    np.testing.assert_allclose(result["avg_hedge_vol"], np.mean(result["hedge_vols"]), atol=1e-12)


def test_evaluate_all_pairs_n18():
    """n=18 must produce exactly 153 pairs."""
    n, T = 18, 100
    rng = np.random.default_rng(7)
    returns = rng.standard_normal((T, n)) * 0.01
    H_series = np.zeros((T, n, n))
    for t in range(T):
        H_series[t] = np.eye(n) * 1e-4
    result = evaluate_all_pairs(returns, H_series)
    assert len(result["minvar_vols"]) == 153


def test_evaluate_all_pairs_non_negative():
    n, T = 4, 100
    rng = np.random.default_rng(8)
    returns = rng.standard_normal((T, n)) * 0.01
    H_series = np.zeros((T, n, n))
    for t in range(T):
        H_series[t] = np.eye(n) * 1e-4
    result = evaluate_all_pairs(returns, H_series)
    assert result["avg_minvar_vol"] >= 0
    assert result["avg_hedge_vol"] >= 0
    assert np.all(result["minvar_vols"] >= 0)
    assert np.all(result["hedge_vols"] >= 0)
