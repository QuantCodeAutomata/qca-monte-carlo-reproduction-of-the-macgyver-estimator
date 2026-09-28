"""
Portfolio Evaluation Module.

Constructs minimum-variance and hedge portfolios from covariance forecasts
and computes realized volatility, following Engle (2007) economic evaluation.
"""

import itertools
from typing import Dict, Tuple

import numpy as np


def min_variance_weights(h_ii: float, h_jj: float, h_ij: float) -> float:
    """Compute the minimum-variance portfolio weight on asset i.

    w = (h_jj - h_ij) / (h_ii + h_jj - 2 * h_ij)

    The portfolio return is: r_port = w * r_i + (1 - w) * r_j.
    If the denominator is near zero, return 0.5 as a safe fallback.

    Args:
        h_ii: Conditional variance of asset i.
        h_jj: Conditional variance of asset j.
        h_ij: Conditional covariance of assets i and j.

    Returns:
        w: Weight on asset i.
    """
    denom = h_ii + h_jj - 2.0 * h_ij
    if abs(denom) < 1e-12:
        return 0.5
    return float((h_jj - h_ij) / denom)


def hedge_ratio(h_ij: float, h_jj: float) -> float:
    """Compute the OLS hedge ratio beta = h_ij / h_jj.

    The hedge portfolio return is: r_port = r_i - beta * r_j.
    If h_jj is near zero, return 0.

    Args:
        h_ij: Conditional covariance of assets i and j.
        h_jj: Conditional variance of asset j (the hedging instrument).

    Returns:
        beta: Hedge ratio.
    """
    if abs(h_jj) < 1e-12:
        return 0.0
    return float(h_ij / h_jj)


def compute_pair_realized_vol(
    ri: np.ndarray,
    rj: np.ndarray,
    H_series: np.ndarray,
    pair_idx: Tuple[int, int],
) -> Dict[str, float]:
    """Compute realized volatility for min-variance and hedge portfolios of a pair.

    Uses t-1 covariance forecast to form portfolio weight at time t,
    then evaluates realized return at time t.

    Args:
        ri:       (T,) return series for asset i.
        rj:       (T,) return series for asset j.
        H_series: (T, n, n) covariance forecast matrices.
        pair_idx: (i_idx, j_idx) indices into the n dimension of H_series.

    Returns:
        Dict with 'minvar_vol' and 'hedge_vol' (both std dev of portfolio returns).
    """
    i_idx, j_idx = pair_idx
    T = len(ri)

    minvar_returns = np.empty(T, dtype=np.float64)
    hedge_returns = np.empty(T, dtype=np.float64)

    for t in range(T):
        h_ii = float(H_series[t, i_idx, i_idx])
        h_jj = float(H_series[t, j_idx, j_idx])
        h_ij = float(H_series[t, i_idx, j_idx])

        w = min_variance_weights(h_ii, h_jj, h_ij)
        minvar_returns[t] = w * ri[t] + (1.0 - w) * rj[t]

        beta_h = hedge_ratio(h_ij, h_jj)
        hedge_returns[t] = ri[t] - beta_h * rj[t]

    return {
        "minvar_vol": float(np.std(minvar_returns, ddof=1)),
        "hedge_vol": float(np.std(hedge_returns, ddof=1)),
    }


def evaluate_all_pairs(
    returns: np.ndarray,
    H_series: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Evaluate all n(n-1)/2 pairs and aggregate realized volatilities.

    Args:
        returns:  (T, n) return matrix.
        H_series: (T, n, n) covariance forecast matrices.

    Returns:
        Dict with keys:
            'minvar_vols':   (n_pairs,) realized vol for min-var portfolios.
            'hedge_vols':    (n_pairs,) realized vol for hedge portfolios.
            'avg_minvar_vol': float mean across pairs.
            'avg_hedge_vol':  float mean across pairs.
    """
    n = returns.shape[1]
    pairs = list(itertools.combinations(range(n), 2))
    n_pairs = len(pairs)

    minvar_vols = np.empty(n_pairs, dtype=np.float64)
    hedge_vols = np.empty(n_pairs, dtype=np.float64)

    for k, (i, j) in enumerate(pairs):
        result = compute_pair_realized_vol(
            returns[:, i], returns[:, j], H_series, (i, j)
        )
        minvar_vols[k] = result["minvar_vol"]
        hedge_vols[k] = result["hedge_vol"]

    return {
        "minvar_vols": minvar_vols,
        "hedge_vols": hedge_vols,
        "avg_minvar_vol": float(np.mean(minvar_vols)),
        "avg_hedge_vol": float(np.mean(hedge_vols)),
    }
