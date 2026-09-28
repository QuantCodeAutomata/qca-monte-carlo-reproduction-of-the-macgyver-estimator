"""
Unit tests for the DCC simulation module (src/dcc_simulation.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dcc_simulation import build_equicorr_matrix, generate_dcc_data


# ---------------------------------------------------------------------------
# build_equicorr_matrix tests
# ---------------------------------------------------------------------------

def test_build_equicorr_matrix_diagonal():
    """Diagonal entries must all be 1."""
    R = build_equicorr_matrix(5, 0.5)
    np.testing.assert_allclose(np.diag(R), np.ones(5), atol=1e-10)


def test_build_equicorr_matrix_off_diagonal():
    """Off-diagonal entries must equal rho."""
    rho = 0.3
    R = build_equicorr_matrix(4, rho)
    mask = ~np.eye(4, dtype=bool)
    np.testing.assert_allclose(R[mask], rho, atol=1e-10)


def test_build_equicorr_matrix_symmetry():
    R = build_equicorr_matrix(6, 0.4)
    np.testing.assert_allclose(R, R.T, atol=1e-10)


def test_build_equicorr_matrix_shape():
    for n in [2, 3, 10]:
        R = build_equicorr_matrix(n, 0.2)
        assert R.shape == (n, n)


def test_build_equicorr_matrix_zero_rho():
    R = build_equicorr_matrix(4, 0.0)
    np.testing.assert_allclose(R, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# generate_dcc_data tests
# ---------------------------------------------------------------------------

def test_generate_dcc_data_shape():
    n, T = 5, 100
    S, Rt, Qt = generate_dcc_data(n, T, 0.05, 0.90, 0.50, seed=0)
    assert S.shape == (T, n), f"S shape mismatch: {S.shape}"
    assert Rt.shape == (T, n, n)
    assert Qt.shape == (T, n, n)


def test_generate_dcc_data_correlation_diag():
    """Rt matrices must have unit diagonals (property of correlation matrices)."""
    n, T = 3, 200
    _, Rt, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.50, seed=1)
    for t in range(T):
        np.testing.assert_allclose(np.diag(Rt[t]), np.ones(n), atol=1e-6,
                                   err_msg=f"Diagonal check failed at t={t}")


def test_generate_dcc_data_correlation_symmetric():
    """Rt matrices must be symmetric."""
    n, T = 3, 50
    _, Rt, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.50, seed=2)
    for t in range(T):
        np.testing.assert_allclose(Rt[t], Rt[t].T, atol=1e-10)


def test_generate_dcc_data_stationarity_violation():
    """alpha + beta >= 1 must raise AssertionError."""
    with pytest.raises(AssertionError):
        generate_dcc_data(3, 100, 0.6, 0.5, 0.5, seed=0)  # sum = 1.1


def test_generate_dcc_data_stationarity_boundary():
    """alpha + beta = 1.0 must raise AssertionError."""
    with pytest.raises(AssertionError):
        generate_dcc_data(3, 100, 0.1, 0.9, 0.5, seed=0)


def test_generate_dcc_data_reproducible():
    """Same seed must produce identical output."""
    S1, Rt1, Q1 = generate_dcc_data(3, 50, 0.05, 0.90, 0.5, seed=99)
    S2, Rt2, Q2 = generate_dcc_data(3, 50, 0.05, 0.90, 0.5, seed=99)
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(Rt1, Rt2)


def test_generate_dcc_data_different_seeds():
    """Different seeds must produce different output."""
    S1, _, _ = generate_dcc_data(3, 50, 0.05, 0.90, 0.5, seed=1)
    S2, _, _ = generate_dcc_data(3, 50, 0.05, 0.90, 0.5, seed=2)
    assert not np.allclose(S1, S2)


def test_generate_dcc_data_large_rho():
    """High equicorrelation must produce highly correlated output."""
    n, T = 5, 1000
    S, _, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.80, seed=10)
    sample_corr = np.corrcoef(S.T)
    off_diag = sample_corr[~np.eye(n, dtype=bool)]
    assert np.mean(off_diag) > 0.4, f"Expected avg off-diag corr > 0.4, got {np.mean(off_diag):.3f}"


def test_generate_dcc_data_low_rho():
    """Low equicorrelation must produce weakly correlated output."""
    n, T = 5, 1000
    S, _, _ = generate_dcc_data(n, T, 0.05, 0.90, 0.10, seed=20)
    sample_corr = np.corrcoef(S.T)
    off_diag = sample_corr[~np.eye(n, dtype=bool)]
    assert np.mean(off_diag) < 0.4, f"Expected avg off-diag corr < 0.4, got {np.mean(off_diag):.3f}"


def test_generate_dcc_data_no_nan():
    """Output must contain no NaN values."""
    S, Rt, Qt = generate_dcc_data(5, 200, 0.05, 0.90, 0.5, seed=30)
    assert not np.any(np.isnan(S))
    assert not np.any(np.isnan(Rt))
    assert not np.any(np.isnan(Qt))


def test_generate_dcc_data_dtype():
    """Output arrays must be float64."""
    S, Rt, Qt = generate_dcc_data(3, 50, 0.05, 0.90, 0.5, seed=0)
    assert S.dtype == np.float64
    assert Rt.dtype == np.float64
    assert Qt.dtype == np.float64


def test_generate_dcc_data_designs():
    """All 10 paper designs must generate without error."""
    designs = [
        (3,  0.05, 0.90, 0.50), (5,  0.05, 0.90, 0.50),
        (10, 0.05, 0.90, 0.50), (20, 0.05, 0.90, 0.50),
        (30, 0.05, 0.90, 0.50), (50, 0.05, 0.90, 0.50),
        (10, 0.05, 0.94, 0.50), (10, 0.02, 0.97, 0.50),
        (10, 0.05, 0.90, 0.20), (10, 0.05, 0.90, 0.80),
    ]
    for n, a, b, rho in designs:
        S, Rt, Qt = generate_dcc_data(n, 100, a, b, rho, seed=42)
        assert S.shape == (100, n)
