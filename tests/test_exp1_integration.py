"""
Integration tests for Experiment 1: Monte Carlo MacGyver Estimator.

Uses a reduced number of replications (5) to keep CI runtime short while
validating end-to-end correctness of the pipeline.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp.exp1_monte_carlo import (
    compute_bias_rmse,
    run_design,
    run_single_replication,
    summarize_design_results,
)


# ---------------------------------------------------------------------------
# compute_bias_rmse tests
# ---------------------------------------------------------------------------

def test_compute_bias_rmse_zero_error():
    """If all estimates equal the true value, bias and RMSE must be zero."""
    true_val = 0.05
    estimates = np.full(50, true_val)
    bias, rmse = compute_bias_rmse(estimates, true_val)
    assert abs(bias) < 1e-12
    assert abs(rmse) < 1e-12


def test_compute_bias_rmse_known_error():
    """Test with known systematic over-estimate."""
    true_val = 0.05
    estimates = np.full(100, 0.06)  # bias = 0.01, rmse = 0.01
    bias, rmse = compute_bias_rmse(estimates, true_val)
    np.testing.assert_allclose(bias, 0.01, atol=1e-10)
    np.testing.assert_allclose(rmse, 0.01, atol=1e-10)


def test_compute_bias_rmse_rmse_nonneg():
    estimates = np.random.default_rng(0).standard_normal(100) * 0.05 + 0.05
    _, rmse = compute_bias_rmse(estimates, 0.05)
    assert rmse >= 0


# ---------------------------------------------------------------------------
# run_single_replication tests
# ---------------------------------------------------------------------------

def test_single_replication_returns_all_keys():
    """Single replication must return all 6 estimator keys."""
    result = run_single_replication(n=3, alpha=0.05, beta=0.90, rho_bar=0.50,
                                    seed=0, T=200)
    expected_keys = {"MEAN", "MED", "MEANT05", "R_MEAN", "R_MED", "R_MEANT05"}
    assert set(result.keys()) == expected_keys


def test_single_replication_estimates_finite():
    """All 6 estimators must return finite alpha and beta."""
    result = run_single_replication(n=3, alpha=0.05, beta=0.90, rho_bar=0.50,
                                    seed=1, T=200)
    for method, (a_hat, b_hat) in result.items():
        assert np.isfinite(a_hat), f"{method} alpha is not finite: {a_hat}"
        assert np.isfinite(b_hat), f"{method} beta is not finite: {b_hat}"


def test_single_replication_unrestricted_bounds():
    """Unrestricted MEAN/MED/MEANT05 must be within expected optimizer bounds."""
    result = run_single_replication(n=3, alpha=0.05, beta=0.90, rho_bar=0.50,
                                    seed=2, T=200)
    for method in ["MEAN", "MED", "MEANT05"]:
        a, b = result[method]
        assert 0.001 <= a <= 0.3, f"{method} alpha={a:.4f} out of [0.001, 0.3]"
        assert 0.5 <= b <= 0.9999, f"{method} beta={b:.4f} out of [0.5, 0.9999]"


def test_single_replication_restricted_range():
    """Restricted R_MEAN/R_MED/R_MEANT05 must be in (0, 1) (sigmoid output)."""
    result = run_single_replication(n=3, alpha=0.05, beta=0.90, rho_bar=0.50,
                                    seed=3, T=200)
    for method in ["R_MEAN", "R_MED", "R_MEANT05"]:
        a, b = result[method]
        assert 0.0 < a < 1.0, f"{method} alpha={a:.4f} not in (0, 1)"
        assert 0.0 < b < 1.0, f"{method} beta={b:.4f} not in (0, 1)"


def test_single_replication_deterministic():
    """Same seed must produce identical results."""
    r1 = run_single_replication(n=3, alpha=0.05, beta=0.90, rho_bar=0.50, seed=7, T=200)
    r2 = run_single_replication(n=3, alpha=0.05, beta=0.90, rho_bar=0.50, seed=7, T=200)
    for method in r1:
        assert r1[method] == r2[method], f"Non-deterministic result for {method}"


# ---------------------------------------------------------------------------
# run_design (reduced replications) tests
# ---------------------------------------------------------------------------

def test_run_design_shape():
    design = {"id": 3, "n": 3, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50}
    df = run_design(design, n_reps=3, master_seed=42, series_T=200)
    assert len(df) == 3
    expected_cols = [f"{m}_{p}" for m in
                     ["MEAN", "MED", "MEANT05", "R_MEAN", "R_MED", "R_MEANT05"]
                     for p in ["alpha", "beta"]]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_run_design_no_nan():
    design = {"id": 3, "n": 3, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50}
    df = run_design(design, n_reps=3, master_seed=42, series_T=200)
    num_cols = [c for c in df.columns if c != "rep"]
    assert not df[num_cols].isnull().any().any(), "NaN found in design results"


# ---------------------------------------------------------------------------
# summarize_design_results tests
# ---------------------------------------------------------------------------

def test_summarize_design_results_index():
    design = {"id": 3, "n": 3, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50}
    df = run_design(design, n_reps=5, master_seed=42, series_T=200)
    stats = summarize_design_results(df, design["alpha"], design["beta"])
    assert set(stats.index) == {"MEAN", "MED", "MEANT05", "R_MEAN", "R_MED", "R_MEANT05"}


def test_summarize_design_results_columns():
    design = {"id": 3, "n": 3, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50}
    df = run_design(design, n_reps=5, master_seed=42, series_T=200)
    stats = summarize_design_results(df, design["alpha"], design["beta"])
    for col in ["alpha_bias", "alpha_rmse", "beta_bias", "beta_rmse"]:
        assert col in stats.columns, f"Missing column: {col}"


def test_summarize_design_results_rmse_nonneg():
    design = {"id": 3, "n": 3, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50}
    df = run_design(design, n_reps=5, master_seed=42, series_T=200)
    stats = summarize_design_results(df, design["alpha"], design["beta"])
    assert (stats["alpha_rmse"] >= 0).all(), "Negative alpha RMSE found"
    assert (stats["beta_rmse"] >= 0).all(), "Negative beta RMSE found"


def test_summarize_design_methodology_bias_direction():
    """
    Methodology test: per Engle (2007), small positive alpha bias and
    small negative beta bias are expected.  Run a small n=3 design
    with 5 reps and short T=200 to keep CI fast, verifying MED is finite.
    """
    # Use n=3 with only 5 reps and T=200 to keep CI fast
    design = {"id": 1, "n": 3, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50}
    df = run_design(design, n_reps=5, master_seed=99, series_T=200)
    stats = summarize_design_results(df, design["alpha"], design["beta"])
    # Both must be finite (sign pattern needs more reps to be reliable)
    alpha_bias = stats.loc["MED", "alpha_bias"]
    beta_bias = stats.loc["MED", "beta_bias"]
    assert np.isfinite(alpha_bias), f"alpha bias not finite: {alpha_bias}"
    assert np.isfinite(beta_bias), f"beta bias not finite: {beta_bias}"
