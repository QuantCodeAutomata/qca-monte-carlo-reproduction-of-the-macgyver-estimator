"""
Factor Covariance Models.

Implements FACTOR ARCH, FACTOR DOUBLE ARCH, and FACTOR DCC covariance models
from Engle (2007).  Uses the arch library for GARCH estimation.

# Using arch arch_model — Context7 confirmed for GJR-GARCH / symmetric GARCH
"""

from typing import Tuple

import numpy as np
import statsmodels.api as sm
from arch import arch_model


# ---------------------------------------------------------------------------
# Univariate volatility estimation
# ---------------------------------------------------------------------------

def estimate_gjr_garch(
    returns: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a GJR-GARCH(1,1) model (threshold GARCH) for a univariate series.

    Model: h_t = omega + theta*r_{t-1}^2 + gamma*r_{t-1}^2*I(r_{t-1}<0) + phi*h_{t-1}
    Corresponds to arch_model(mean='Zero', vol='GARCH', p=1, o=1, q=1).

    Starting NaN values in conditional variance are backfilled with the
    unconditional variance estimate.

    Args:
        returns: (T,) return series (already demeaned or zero-mean assumed).
        verbose: If True, print arch fit summary.

    Returns:
        h:          (T,) conditional variance series.
        std_resids: (T,) standardized residuals r_t / sqrt(h_t).
    """
    r = np.asarray(returns, dtype=np.float64)
    am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = am.fit(disp="off" if not verbose else "on", show_warning=False)

    h = res.conditional_volatility ** 2
    h = np.asarray(h, dtype=np.float64)

    # Backfill any leading NaNs with the unconditional variance
    nan_mask = np.isnan(h)
    if nan_mask.any():
        unconditional = float(np.var(r))
        h[nan_mask] = unconditional

    # Guard against zero variance
    h = np.maximum(h, 1e-12)
    std_resids = r / np.sqrt(h)

    return h, std_resids


def estimate_symmetric_garch(
    returns: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a symmetric GARCH(1,1) model for market or idiosyncratic series.

    Model: h_t = omega + alpha*r_{t-1}^2 + beta*h_{t-1}
    Corresponds to arch_model(mean='Zero', vol='GARCH', p=1, q=1).

    Args:
        returns: (T,) return series.
        verbose: If True, print arch fit summary.

    Returns:
        h:          (T,) conditional variance series.
        std_resids: (T,) standardized residuals.
    """
    r = np.asarray(returns, dtype=np.float64)
    am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="normal")
    res = am.fit(disp="off" if not verbose else "on", show_warning=False)

    h = res.conditional_volatility ** 2
    h = np.asarray(h, dtype=np.float64)

    # Backfill any leading NaNs
    nan_mask = np.isnan(h)
    if nan_mask.any():
        h[nan_mask] = float(np.var(r))

    h = np.maximum(h, 1e-12)
    std_resids = r / np.sqrt(h)

    return h, std_resids


# ---------------------------------------------------------------------------
# Factor model estimation
# ---------------------------------------------------------------------------

def estimate_factor_betas(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """OLS estimation of one-factor betas without intercept.

    Model: r_{i,t} = beta_i * r_{m,t} + epsilon_{i,t}

    Args:
        stock_returns:  (T, n) stock return matrix.
        market_returns: (T,) market return series.

    Returns:
        betas:     (n,) OLS factor loadings.
        residuals: (T, n) OLS residuals.
    """
    T, n = stock_returns.shape
    market_col = market_returns.reshape(-1, 1).astype(np.float64)

    betas = np.empty(n, dtype=np.float64)
    residuals = np.empty((T, n), dtype=np.float64)

    for i in range(n):
        res = sm.OLS(stock_returns[:, i], market_col).fit()
        betas[i] = float(res.params[0])
        residuals[:, i] = stock_returns[:, i] - betas[i] * market_returns

    return betas, residuals


# ---------------------------------------------------------------------------
# Factor covariance constructors
# ---------------------------------------------------------------------------

def factor_arch_covariance(
    betas: np.ndarray,
    h_market: np.ndarray,
    idio_var: np.ndarray,
) -> np.ndarray:
    """Compute T x n x n FACTOR ARCH covariance matrices.

    H_t = beta * beta' * h_{m,t} + diag(d_i^2)

    Off-diagonal: Cov(r_i, r_j, t) = beta_i * beta_j * h_{m,t}
    Diagonal:     Var(r_i, t)       = beta_i^2 * h_{m,t} + d_i^2

    Args:
        betas:     (n,) factor loadings.
        h_market:  (T,) market conditional variances.
        idio_var:  (n,) constant idiosyncratic variances (d_i^2).

    Returns:
        H: (T, n, n) covariance matrices.
    """
    T = len(h_market)
    n = len(betas)
    beta_outer = np.outer(betas, betas)  # (n, n), time-invariant structure

    H = h_market[:, None, None] * beta_outer[None, :, :]  # (T, n, n)
    # Add constant idiosyncratic variance on diagonal
    diag_idx = np.arange(n)
    H[:, diag_idx, diag_idx] += idio_var

    return H


def factor_double_arch_covariance(
    betas: np.ndarray,
    h_market: np.ndarray,
    h_idio: np.ndarray,
) -> np.ndarray:
    """Compute T x n x n FACTOR DOUBLE ARCH covariance matrices.

    H_t[i, j] = beta_i * beta_j * h_{m,t}   (off-diagonal)
    H_t[i, i] = beta_i^2 * h_{m,t} + d_{i,t}^2  (diagonal)

    Uses standard GARCH(1,1) for idiosyncratic volatility (see exp2 for estimation).

    Args:
        betas:    (n,) factor loadings.
        h_market: (T,) market conditional variances.
        h_idio:   (T, n) idiosyncratic conditional variances.

    Returns:
        H: (T, n, n) covariance matrices.
    """
    T = len(h_market)
    n = len(betas)
    beta_outer = np.outer(betas, betas)

    H = h_market[:, None, None] * beta_outer[None, :, :]
    # Replace diagonal with beta_i^2 * h_{m,t} + d_{i,t}^2
    diag_idx = np.arange(n)
    H[:, diag_idx, diag_idx] = betas[None, :] ** 2 * h_market[:, None] + h_idio

    return H


def factor_dcc_covariance(
    betas: np.ndarray,
    h_market: np.ndarray,
    h_idio: np.ndarray,
    Rt_resid: np.ndarray,
) -> np.ndarray:
    """Compute T x n x n FACTOR DCC covariance matrices.

    Covariance formula from Engle (2007) including residual correlations
    between idiosyncratic innovations and the market innovation:

        Cov(r_i, r_j, t) = beta_i*beta_j*h_{m,t}
                         + d_{i,t}*d_{j,t}*rho^e_{ij,t}
                         + beta_i*d_{j,t}*sqrt(h_{m,t})*rho^e_{mj,t}
                         + beta_j*d_{i,t}*sqrt(h_{m,t})*rho^e_{mi,t}

        Var(r_i, t) = beta_i^2*h_{m,t} + d_{i,t}^2
                    + 2*beta_i*d_{i,t}*sqrt(h_{m,t})*rho^e_{mi,t}

    The Rt_resid matrix is (T, n+1, n+1) over [e_1,...,e_n, e_m] where
    index n (last) corresponds to the market residual.

    Args:
        betas:     (n,) factor loadings.
        h_market:  (T,) market conditional variances.
        h_idio:    (T, n) idiosyncratic conditional variances.
        Rt_resid:  (T, n+1, n+1) DCC residual correlation matrices.

    Returns:
        H: (T, n, n) stock covariance matrices.
    """
    T = len(h_market)
    n = len(betas)
    m_idx = n  # market index in Rt_resid

    d_t = np.sqrt(h_idio)          # (T, n) idiosyncratic std devs
    sqrt_hm = np.sqrt(h_market)    # (T,)

    H = np.empty((T, n, n), dtype=np.float64)

    for t in range(T):
        hm_t = h_market[t]
        sqrt_hm_t = sqrt_hm[t]
        d = d_t[t]                      # (n,)
        R = Rt_resid[t]                 # (n+1, n+1)

        rho_ee = R[:n, :n]              # (n, n) idio-idio correlations
        rho_me = R[m_idx, :n]           # (n,)  market-idio correlations

        # Covariance matrix
        # H_ij = beta_i*beta_j*hm + d_i*d_j*rho_ij
        #       + beta_i*d_j*sqrt(hm)*rho_mj + beta_j*d_i*sqrt(hm)*rho_mi
        H_t = (
            np.outer(betas, betas) * hm_t
            + np.outer(d, d) * rho_ee
            + np.outer(betas * sqrt_hm_t, d) * rho_me[None, :]
            + np.outer(d, betas * sqrt_hm_t) * rho_me[:, None]
        )

        H[t] = H_t

    return H
