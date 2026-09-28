"""
Pairwise DCC Estimation and MacGyver Aggregation.

Implements Engle (2007) MacGyver procedure:
  1. Estimate all n(n-1)/2 bivariate DCC models via pairwise Gaussian log-likelihood.
  2. Aggregate pairwise alpha/beta estimates using MEAN, MED, and 5%-trimmed mean.

Custom — Context7 found no library equivalent (Engle 2007 MacGyver pairwise aggregation).

Performance: the DCC recursion inner loop is JIT-compiled via numba (@njit) to
achieve ~50-100x speedup over pure-Python iteration.
"""

import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Numba-JIT DCC recursion + log-likelihood core
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dcc_nll_core(
    alpha: float,
    beta: float,
    rbar_ij: float,
    si: np.ndarray,
    sj: np.ndarray,
) -> float:
    """JIT-compiled DCC bivariate correlation negative log-likelihood.

    DCC recursion (correlation-targeted, Engle 2007):
        q_ii_t = (1-a-b)*1    + a*si_{t-1}^2       + b*q_ii_{t-1}
        q_jj_t = (1-a-b)*1    + a*sj_{t-1}^2       + b*q_jj_{t-1}
        q_ij_t = (1-a-b)*rbar + a*si_{t-1}*sj_{t-1} + b*q_ij_{t-1}
        rho_t  = q_ij_t / sqrt(q_ii_t * q_jj_t)

    L = 0.5 * sum_t [log(1-rho_t^2) + (si_t^2+sj_t^2-2*rho_t*si_t*sj_t)/(1-rho_t^2)]

    Args:
        alpha:    DCC ARCH parameter.
        beta:     DCC GARCH parameter.
        rbar_ij:  Unconditional target correlation.
        si, sj:   (T,) standardized return arrays.

    Returns:
        Negative log-likelihood (returns 1e10 on numerical failure).
    """
    LARGE = 1e10
    T = len(si)
    c = 1.0 - alpha - beta

    q_ii = 1.0
    q_jj = 1.0
    q_ij = rbar_ij
    nll = 0.0

    for t in range(T):
        if q_ii < 1e-8 or q_jj < 1e-8:
            return LARGE

        denom = (q_ii * q_jj) ** 0.5
        rho = q_ij / denom
        if rho >  0.9999:
            rho =  0.9999
        if rho < -0.9999:
            rho = -0.9999

        omr2 = 1.0 - rho * rho
        if omr2 < 1e-8:
            return LARGE

        sit = si[t]
        sjt = sj[t]

        nll += 0.5 * (
            np.log(omr2)
            + (sit * sit + sjt * sjt - 2.0 * rho * sit * sjt) / omr2
        )

        q_ii_new = c + alpha * sit * sit   + beta * q_ii
        q_jj_new = c + alpha * sjt * sjt   + beta * q_jj
        q_ij     = c * rbar_ij + alpha * sit * sjt + beta * q_ij

        q_ii = q_ii_new if q_ii_new > 1e-10 else 1e-10
        q_jj = q_jj_new if q_jj_new > 1e-10 else 1e-10

    return nll


# ---------------------------------------------------------------------------
# Pairwise DCC log-likelihood (Python wrapper used by scipy.optimize)
# ---------------------------------------------------------------------------

def pairwise_dcc_loglik(
    params: np.ndarray,
    si: np.ndarray,
    sj: np.ndarray,
    rbar_ij: float,
    restricted: bool = False,
) -> float:
    """Negative bivariate DCC correlation log-likelihood for a pair (i, j).

    Delegates hot computation to the numba-JIT `_dcc_nll_core`.

    Args:
        params:     [alpha, beta] for unrestricted; [theta, phi] for restricted where
                    alpha = sigmoid(theta), beta = sigmoid(phi).
        si, sj:     T-length standardized return arrays for the pair.
        rbar_ij:    Unconditional (sample) correlation between si and sj.
        restricted: If True, apply sigmoid transformation to params.

    Returns:
        Negative log-likelihood (scalar) for minimization.
    """
    if restricted:
        theta, phi = float(params[0]), float(params[1])
        alpha = np.exp(theta) / (1.0 + np.exp(theta))
        beta  = np.exp(phi)  / (1.0 + np.exp(phi))
    else:
        alpha, beta = float(params[0]), float(params[1])

    return _dcc_nll_core(alpha, beta, float(rbar_ij), si, sj)


# ---------------------------------------------------------------------------
# Pairwise DCC estimation
# ---------------------------------------------------------------------------

def estimate_pair_dcc(
    si: np.ndarray,
    sj: np.ndarray,
    rbar_ij: float,
    restricted: bool = False,
    x0_alpha: float = 0.03,
    x0_beta: float = 0.93,
    max_iter: int = 500,
) -> Dict:
    """Estimate bivariate DCC parameters for a single pair.

    Unrestricted: L-BFGS-B with bounds [(0.001, 0.3), (0.5, 0.9999)].
    Restricted:   Nelder-Mead over logit-transformed parameters (no sum constraint).

    Per Engle (2007): if the optimizer does not converge, the final iterate is
    kept as the estimate rather than discarding or repairing it.

    Args:
        si, sj:     T-length standardized return arrays.
        rbar_ij:    Sample correlation of the pair (used as DCC target).
        restricted: If True, optimize over logit-parameterized alpha and beta.
        x0_alpha:   Starting value for alpha.
        x0_beta:    Starting value for beta.
        max_iter:   Maximum optimizer iterations.

    Returns:
        Dict with keys: alpha, beta, converged (bool), nll (float), message (str).
    """
    if restricted:
        # Map starting values to logit space: theta = log(a/(1-a))
        theta0 = np.log(x0_alpha / (1.0 - x0_alpha))
        phi0 = np.log(x0_beta / (1.0 - x0_beta))
        x0 = np.array([theta0, phi0], dtype=np.float64)

        # L-BFGS-B over logit space (unconstrained — no alpha+beta<1 restriction
        # per Engle 2007 section on restricted parameterization)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                pairwise_dcc_loglik,
                x0,
                args=(si, sj, rbar_ij, True),
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-5},
            )

        theta_hat, phi_hat = result.x
        alpha_hat = float(np.exp(theta_hat) / (1.0 + np.exp(theta_hat)))
        beta_hat = float(np.exp(phi_hat) / (1.0 + np.exp(phi_hat)))

    else:
        x0 = np.array([x0_alpha, x0_beta], dtype=np.float64)
        bounds = [(0.001, 0.3), (0.5, 0.9999)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                pairwise_dcc_loglik,
                x0,
                args=(si, sj, rbar_ij, False),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-5},
            )

        alpha_hat, beta_hat = float(result.x[0]), float(result.x[1])

    return {
        "alpha": alpha_hat,
        "beta": beta_hat,
        "converged": bool(result.success),
        "nll": float(result.fun),
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# MacGyver aggregation
# ---------------------------------------------------------------------------

def macgyver_aggregate(
    alpha_estimates: np.ndarray,
    beta_estimates: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Aggregate pairwise DCC estimates using MEAN, MED, and 5%-trimmed mean.

    Trimming rule: remove floor(0.05 * n_pairs) estimates from each tail.
    This is applied separately to alpha and beta arrays.

    Args:
        alpha_estimates: (n_pairs,) array of pairwise alpha estimates.
        beta_estimates:  (n_pairs,) array of pairwise beta estimates.

    Returns:
        Dict with keys 'MEAN', 'MED', 'MEANT05', each containing
        {'alpha': float, 'beta': float}.
    """
    n_pairs = len(alpha_estimates)
    k = int(np.floor(0.05 * n_pairs))  # number to trim from each tail

    def trimmed_mean(arr: np.ndarray, k: int) -> float:
        sorted_arr = np.sort(arr)
        if k == 0:
            return float(np.mean(sorted_arr))
        return float(np.mean(sorted_arr[k : n_pairs - k]))

    return {
        "MEAN": {
            "alpha": float(np.mean(alpha_estimates)),
            "beta": float(np.mean(beta_estimates)),
        },
        "MED": {
            "alpha": float(np.median(alpha_estimates)),
            "beta": float(np.median(beta_estimates)),
        },
        "MEANT05": {
            "alpha": trimmed_mean(alpha_estimates, k),
            "beta": trimmed_mean(beta_estimates, k),
        },
    }


# ---------------------------------------------------------------------------
# Full MacGyver estimation run
# ---------------------------------------------------------------------------

def run_macgyver_estimation(
    S: np.ndarray,
    restricted: bool = False,
    x0_alpha: float = 0.03,
    x0_beta: float = 0.93,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run all n(n-1)/2 pairwise DCC estimations on T x n standardized returns.

    Args:
        S:          (T, n) standardized return matrix.
        restricted: If True, use logit-parameterized restricted estimation.
        x0_alpha:   Starting value for alpha.
        x0_beta:    Starting value for beta.
        max_iter:   Maximum optimizer iterations per pair.

    Returns:
        alpha_estimates: (n_pairs,) array of pairwise alpha estimates.
        beta_estimates:  (n_pairs,) array of pairwise beta estimates.
    """
    n = S.shape[1]
    pairs = list(itertools.combinations(range(n), 2))
    n_pairs = len(pairs)

    alpha_estimates = np.empty(n_pairs, dtype=np.float64)
    beta_estimates = np.empty(n_pairs, dtype=np.float64)

    for k, (i, j) in enumerate(pairs):
        si = S[:, i].astype(np.float64)
        sj = S[:, j].astype(np.float64)
        rbar_ij = float(np.corrcoef(si, sj)[0, 1])
        # Guard against degenerate correlation
        rbar_ij = float(np.clip(rbar_ij, -0.9999, 0.9999))

        result = estimate_pair_dcc(
            si, sj, rbar_ij,
            restricted=restricted,
            x0_alpha=x0_alpha,
            x0_beta=x0_beta,
            max_iter=max_iter,
        )
        alpha_estimates[k] = result["alpha"]
        beta_estimates[k] = result["beta"]

    return alpha_estimates, beta_estimates


# ---------------------------------------------------------------------------
# Full DCC state reconstruction from common parameters
# ---------------------------------------------------------------------------

def reconstruct_full_dcc(
    S: np.ndarray,
    alpha: float,
    beta: float,
    Rbar: np.ndarray,
) -> np.ndarray:
    """Reconstruct T x n x n DCC correlation matrices using common parameters.

    Uses Q_t = (1-alpha-beta)*Rbar + alpha*s_{t-1}s_{t-1}' + beta*Q_{t-1}
    with Q_0 = Rbar.

    Args:
        S:     (T, n) standardized returns.
        alpha: Common DCC ARCH parameter.
        beta:  Common DCC GARCH parameter.
        Rbar:  (n, n) unconditional correlation target matrix.

    Returns:
        Rt_series: (T, n, n) array of conditional correlation matrices.
    """
    T, n = S.shape
    c = 1.0 - alpha - beta

    Rt_series = np.empty((T, n, n), dtype=np.float64)
    Q_t = Rbar.copy()

    # Initial shock: use zeros (s_0 = 0 vectors) for reconstruction
    s_prev = np.zeros(n, dtype=np.float64)

    for t in range(T):
        Q_t = c * Rbar + alpha * np.outer(s_prev, s_prev) + beta * Q_t

        # Convert to correlation matrix
        diag_q = np.maximum(np.diag(Q_t), 1e-10)
        inv_sqrt = 1.0 / np.sqrt(diag_q)
        R_t = inv_sqrt[:, None] * Q_t * inv_sqrt[None, :]

        # Enforce symmetry and unit diagonal
        R_t = 0.5 * (R_t + R_t.T)
        np.fill_diagonal(R_t, 1.0)

        Rt_series[t] = R_t
        s_prev = S[t]

    return Rt_series
