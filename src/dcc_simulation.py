"""
DCC Simulation Module.

Generates synthetic standardized returns from a Gaussian DCC(alpha, beta) process
using the correlation-targeted parameterization of Engle (2007).
"""

import numpy as np
from typing import Tuple


def build_equicorr_matrix(n: int, rho: float) -> np.ndarray:
    """Build an n x n equicorrelation matrix with rho off-diagonal.

    Args:
        n:   Dimension of the matrix.
        rho: Off-diagonal correlation value.

    Returns:
        R: (n, n) equicorrelation matrix with ones on the diagonal.
    """
    R = np.full((n, n), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    return R


def generate_dcc_data(
    n: int,
    T: int,
    alpha: float,
    beta: float,
    rho_bar: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate T x n standardized returns from a Gaussian DCC(alpha, beta) process.

    Uses the correlation-targeted DCC recursion:
        Q_t = (1 - alpha - beta) * Rbar + alpha * s_{t-1} s_{t-1}' + beta * Q_{t-1}
        R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        s_t ~ N(0, R_t)

    Initialization:
        Q_0 = Rbar  (exact unconditional target)
        s_0 ~ N(0, Rbar)  using the provided seed (fixed choice documented here)

    Args:
        n:       Number of series.
        T:       Sample size.
        alpha:   DCC ARCH parameter.
        beta:    DCC GARCH parameter.
        rho_bar: Off-diagonal equicorrelation for Rbar.
        seed:    Random seed for reproducibility.

    Returns:
        S:         (T, n) array of standardized returns.
        Rt_series: (T, n, n) array of true conditional correlation matrices.
        Q_series:  (T, n, n) array of Q_t matrices.
    """
    assert alpha + beta < 1.0, (
        f"DCC stationarity requires alpha + beta < 1, got {alpha + beta:.4f}"
    )
    assert alpha >= 0 and beta >= 0, "alpha and beta must be non-negative"

    rng = np.random.default_rng(seed)
    Rbar = build_equicorr_matrix(n, rho_bar)

    # Pre-allocate output arrays
    S = np.empty((T, n), dtype=np.float64)
    Rt_series = np.empty((T, n, n), dtype=np.float64)
    Q_series = np.empty((T, n, n), dtype=np.float64)

    c = 1.0 - alpha - beta  # scalar target weight

    # Initialization: Q_0 = Rbar, s_0 ~ N(0, Rbar)
    Q_prev = Rbar.copy()
    L0 = np.linalg.cholesky(Rbar)
    s_prev = L0 @ rng.standard_normal(n)

    for t in range(T):
        # DCC recursion
        Q_t = c * Rbar + alpha * np.outer(s_prev, s_prev) + beta * Q_prev

        # Convert to proper correlation matrix: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        diag_q = np.diag(Q_t)
        diag_q = np.maximum(diag_q, 1e-10)  # numerical guard
        inv_sqrt_diag = 1.0 / np.sqrt(diag_q)
        R_t = inv_sqrt_diag[:, None] * Q_t * inv_sqrt_diag[None, :]

        # Enforce symmetry and unit diagonal
        R_t = 0.5 * (R_t + R_t.T)
        np.fill_diagonal(R_t, 1.0)

        # Clip off-diagonal entries to valid correlation range
        off = np.triu_indices(n, k=1)
        R_t[off] = np.clip(R_t[off], -1.0 + 1e-8, 1.0 - 1e-8)
        R_t[off[1], off[0]] = R_t[off]

        # Draw s_t ~ N(0, R_t) via Cholesky
        try:
            L_t = np.linalg.cholesky(R_t)
        except np.linalg.LinAlgError:
            # Fallback: add small ridge to diagonal
            R_t_reg = R_t + 1e-6 * np.eye(n)
            np.fill_diagonal(R_t_reg, 1.0)
            L_t = np.linalg.cholesky(R_t_reg)

        s_t = L_t @ rng.standard_normal(n)

        # Store
        S[t] = s_t
        Rt_series[t] = R_t
        Q_series[t] = Q_t

        # Update for next step
        Q_prev = Q_t
        s_prev = s_t

    return S, Rt_series, Q_series
