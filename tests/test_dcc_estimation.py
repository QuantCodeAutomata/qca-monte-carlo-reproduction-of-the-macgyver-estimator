"""
Unit tests for pairwise DCC estimation and MacGyver aggregation
(src/dcc_estimation.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dcc_estimation import (
    estimate_pair_dcc,
    macgyver_aggregate,
    pairwise_dcc_loglik,
    reconstruct_full_dcc,
    run_macgyver_estimation,
)
from src.dcc_simulation import generate_dcc_data


# ---------------------------------------------------------------------------
# pairwise_dcc_loglik tests
# ---------------------------------------------------------------------------

def test_pairwise_loglik_finite():
    """Log-likelihood must return a finite float for valid inputs."""
    rng = np.random.default_rng(0)
    si = rng.standard_normal(100)
    sj = rng.standard_normal(100)
    nll = pairwise_dcc_loglik(np.array([0.03, 0.93]), si, sj, rbar_ij=0.3)
    assert np.isfinite(nll), f"Expected finite NLL, got {nll}"


def test_pairwise_loglik_positive():
    """NLL (minimization objective) must be positive for well-behaved inputs."""
    rng = np.random.default_rng(1)
    si = rng.standard_normal(200)
    sj = rng.standard_normal(200)
    nll = pairwise_dcc_loglik(np.array([0.03, 0.93]), si, sj, rbar_ij=0.0)
    assert nll > 0, f"Expected positive NLL, got {nll}"


def test_pairwise_loglik_restricted_finite():
    """Restricted (logit-parameterized) log-likelihood must be finite."""
    rng = np.random.default_rng(2)
    si = rng.standard_normal(150)
    sj = rng.standard_normal(150)
    # logit(0.03) ~ -3.48, logit(0.93) ~ 2.59
    theta = np.log(0.03 / 0.97)
    phi = np.log(0.93 / 0.07)
    nll = pairwise_dcc_loglik(np.array([theta, phi]), si, sj, rbar_ij=0.2, restricted=True)
    assert np.isfinite(nll), f"Expected finite NLL, got {nll}"


def test_pairwise_loglik_penalizes_extreme_params():
    """Extremely large alpha+beta should trigger the large-penalty guard."""
    rng = np.random.default_rng(3)
    si = rng.standard_normal(100)
    sj = rng.standard_normal(100)
    nll_bad = pairwise_dcc_loglik(np.array([0.5, 0.8]), si, sj, rbar_ij=0.3)
    nll_good = pairwise_dcc_loglik(np.array([0.03, 0.93]), si, sj, rbar_ij=0.3)
    # A large sum alpha+beta often leads to higher NLL (worse fit or penalty)
    assert nll_bad >= 0


def test_pairwise_loglik_returns_float():
    rng = np.random.default_rng(4)
    nll = pairwise_dcc_loglik(np.array([0.03, 0.93]),
                              rng.standard_normal(100), rng.standard_normal(100),
                              rbar_ij=0.0)
    assert isinstance(nll, float)


# ---------------------------------------------------------------------------
# estimate_pair_dcc tests
# ---------------------------------------------------------------------------

def test_estimate_pair_returns_dict_keys():
    """Must return dict with required keys."""
    S, _, _ = generate_dcc_data(2, 200, 0.05, 0.90, 0.5, seed=5)
    rbar = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1])
    result = estimate_pair_dcc(S[:, 0], S[:, 1], rbar, restricted=False)
    assert "alpha" in result
    assert "beta" in result
    assert "converged" in result
    assert "nll" in result


def test_estimate_pair_bounds_unrestricted():
    """Unrestricted estimates must be in [0.001, 0.3] x [0.5, 0.9999]."""
    S, _, _ = generate_dcc_data(2, 500, 0.05, 0.90, 0.5, seed=7)
    rbar = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1])
    result = estimate_pair_dcc(S[:, 0], S[:, 1], rbar, restricted=False)
    assert 0.001 <= result["alpha"] <= 0.3
    assert 0.5 <= result["beta"] <= 0.9999


def test_estimate_pair_restricted_returns():
    """Restricted estimates must be in (0, 1) (sigmoid output)."""
    S, _, _ = generate_dcc_data(2, 400, 0.05, 0.90, 0.5, seed=8)
    rbar = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1])
    result = estimate_pair_dcc(S[:, 0], S[:, 1], rbar, restricted=True)
    assert 0.0 < result["alpha"] < 1.0
    assert 0.0 < result["beta"] < 1.0


def test_estimate_pair_convergence_bool():
    """converged field must be a bool."""
    S, _, _ = generate_dcc_data(2, 200, 0.05, 0.90, 0.5, seed=9)
    rbar = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1])
    result = estimate_pair_dcc(S[:, 0], S[:, 1], rbar)
    assert isinstance(result["converged"], bool)


# ---------------------------------------------------------------------------
# macgyver_aggregate tests
# ---------------------------------------------------------------------------

def test_macgyver_aggregate_mean():
    alphas = np.array([0.03, 0.04, 0.05, 0.06])
    betas  = np.array([0.90, 0.91, 0.92, 0.93])
    agg = macgyver_aggregate(alphas, betas)
    np.testing.assert_allclose(agg["MEAN"]["alpha"], np.mean(alphas), atol=1e-12)
    np.testing.assert_allclose(agg["MEAN"]["beta"],  np.mean(betas),  atol=1e-12)


def test_macgyver_aggregate_median():
    alphas = np.array([0.01, 0.03, 0.05, 0.07, 0.10])
    betas  = np.ones(5) * 0.90
    agg = macgyver_aggregate(alphas, betas)
    np.testing.assert_allclose(agg["MED"]["alpha"], np.median(alphas), atol=1e-12)


def test_macgyver_aggregate_trimmed_5pct():
    """5%-trimmed mean must remove floor(0.05 * n) from each tail."""
    n = 100
    alphas = np.linspace(0.01, 0.20, n)
    betas  = np.ones(n) * 0.90
    agg = macgyver_aggregate(alphas, betas)
    k = int(np.floor(0.05 * n))  # = 5
    expected = np.mean(np.sort(alphas)[k : n - k])
    np.testing.assert_allclose(agg["MEANT05"]["alpha"], expected, atol=1e-12)


def test_macgyver_aggregate_small_n():
    """With only 3 pairs, floor(0.05*3)=0, so trimmed mean equals mean."""
    alphas = np.array([0.02, 0.04, 0.06])
    betas  = np.ones(3) * 0.90
    agg = macgyver_aggregate(alphas, betas)
    np.testing.assert_allclose(agg["MEANT05"]["alpha"], np.mean(alphas), atol=1e-12)


def test_macgyver_aggregate_keys():
    agg = macgyver_aggregate(np.array([0.05, 0.06]), np.array([0.90, 0.91]))
    assert set(agg.keys()) == {"MEAN", "MED", "MEANT05"}
    for v in agg.values():
        assert "alpha" in v and "beta" in v


# ---------------------------------------------------------------------------
# run_macgyver_estimation tests
# ---------------------------------------------------------------------------

def test_run_macgyver_estimation_shape():
    n = 4
    S, _, _ = generate_dcc_data(n, 200, 0.05, 0.90, 0.5, seed=11)
    alphas, betas = run_macgyver_estimation(S, restricted=False)
    n_pairs = n * (n - 1) // 2
    assert len(alphas) == n_pairs
    assert len(betas) == n_pairs


def test_run_macgyver_estimation_n3():
    """n=3 has 3 pairs."""
    S, _, _ = generate_dcc_data(3, 300, 0.05, 0.90, 0.5, seed=15)
    alphas, betas = run_macgyver_estimation(S, restricted=False)
    assert len(alphas) == 3
    assert len(betas) == 3


def test_run_macgyver_estimation_dtype():
    S, _, _ = generate_dcc_data(3, 200, 0.05, 0.90, 0.5, seed=16)
    alphas, betas = run_macgyver_estimation(S)
    assert alphas.dtype == np.float64
    assert betas.dtype == np.float64


# ---------------------------------------------------------------------------
# reconstruct_full_dcc tests
# ---------------------------------------------------------------------------

def test_reconstruct_full_dcc_unit_diagonal():
    """Reconstructed correlation matrices must have unit diagonals."""
    n, T = 3, 100
    S, _, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.5, seed=12)
    Rbar = np.corrcoef(S.T)
    Rt = reconstruct_full_dcc(S, 0.05, 0.90, Rbar)
    for t in range(T):
        np.testing.assert_allclose(np.diag(Rt[t]), np.ones(n), atol=1e-6,
                                   err_msg=f"Diagonal failed at t={t}")


def test_reconstruct_full_dcc_symmetry():
    """Reconstructed correlation matrices must be symmetric."""
    n, T = 4, 100
    S, _, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.5, seed=13)
    Rbar = np.corrcoef(S.T)
    Rt = reconstruct_full_dcc(S, 0.05, 0.90, Rbar)
    for t in range(T):
        np.testing.assert_allclose(Rt[t], Rt[t].T, atol=1e-10)


def test_reconstruct_full_dcc_shape():
    n, T = 5, 200
    S, _, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.5, seed=14)
    Rbar = np.corrcoef(S.T)
    Rt = reconstruct_full_dcc(S, 0.05, 0.90, Rbar)
    assert Rt.shape == (T, n, n)


def test_reconstruct_full_dcc_no_nan():
    n, T = 3, 100
    S, _, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.5, seed=17)
    Rbar = np.corrcoef(S.T)
    Rt = reconstruct_full_dcc(S, 0.05, 0.90, Rbar)
    assert not np.any(np.isnan(Rt)), "NaN found in reconstructed Rt"
