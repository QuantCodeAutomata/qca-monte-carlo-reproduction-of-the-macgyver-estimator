"""
Experiment 2: Empirical Reproduction of DCC, Factor Covariance Models,
and Economic Evaluation on 18 U.S. Large-Cap Stocks (Engle 2007).

Six covariance models:
  1. CONST       - Full-sample constant covariance
  2. HIST100     - Rolling 100-day historical covariance
  3. DCC         - DCC-MacGyver on GJR-GARCH standardized stock returns
  4. FACTOR_ARCH - One-factor model, constant idiosyncratic variance
  5. DOUBLE_ARCH - One-factor model, GARCH(1,1) idiosyncratic variance
  6. FACTOR_DCC  - Factor model + DCC on standardized residuals

Economic evaluation: minimum-variance and hedge portfolios for all 153 pairs.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data.data_loader import compute_returns, load_stock_data
from src.dcc_estimation import (
    macgyver_aggregate,
    reconstruct_full_dcc,
    run_macgyver_estimation,
)
from src.factor_models import (
    estimate_factor_betas,
    estimate_gjr_garch,
    estimate_symmetric_garch,
    factor_arch_covariance,
    factor_dcc_covariance,
    factor_double_arch_covariance,
)
from src.portfolio_eval import evaluate_all_pairs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOCKS = [
    "AA", "AXP", "BA", "CAT", "DD", "DIS", "GE", "GM", "IBM", "IP",
    "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "MSFT",
]
MARKET = "SPY"  # S&P 500 ETF as market proxy


# ---------------------------------------------------------------------------
# Helper: benchmark covariance models
# ---------------------------------------------------------------------------

def build_const_covariance(returns: np.ndarray) -> np.ndarray:
    """Full-sample constant covariance matrix, broadcast to T x n x n.

    Args:
        returns: (T, n) return matrix.

    Returns:
        H: (T, n, n) constant covariance matrices (same matrix repeated).
    """
    T, n = returns.shape
    H_const = np.cov(returns.T)  # (n, n)
    return np.broadcast_to(H_const, (T, n, n)).copy()


def build_hist100_covariance(
    returns: np.ndarray,
    window: int = 100,
) -> Tuple[np.ndarray, int]:
    """Rolling 100-day historical covariance.

    For each date t >= window, uses returns[t-window:t] to estimate covariance.
    For t < window, uses the full-sample covariance as a placeholder
    (but the evaluation period starts from t=window).

    Args:
        returns: (T, n) return matrix.
        window:  Rolling window length.

    Returns:
        H:          (T, n, n) covariance forecast matrices.
        eval_start: Index from which forecasts are based on exactly `window` obs.
    """
    T, n = returns.shape
    H = np.empty((T, n, n), dtype=np.float64)

    # Placeholders for first `window` periods
    H_full = np.cov(returns.T)
    for t in range(window):
        H[t] = H_full

    for t in range(window, T):
        window_data = returns[t - window : t]
        H[t] = np.cov(window_data.T)

    return H, window


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exp2(
    api_key: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Dict:
    """Run the full empirical experiment.

    Args:
        api_key:     Massive API key (defaults to MASSIVE_TOKEN env var).
        results_dir: Path to results directory (default: project/results).

    Returns:
        Dict with performance table, win-fraction matrices, and DCC parameters.
    """
    rdir = Path(results_dir) if results_dir else Path(__file__).parent.parent / "results"
    rdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and align data
    # ------------------------------------------------------------------
    print("Loading data via Massive API...")
    all_tickers = STOCKS + [MARKET]
    prices = load_stock_data(
        all_tickers, start_date="1994-01-01", end_date="2004-12-31", api_key=api_key
    )
    returns_all = compute_returns(prices, log_returns=False)

    # Keep only stocks present in the loaded data (GM may be unavailable)
    available_stocks = [s for s in STOCKS if s in returns_all.columns]
    dropped = [s for s in STOCKS if s not in available_stocks]
    if dropped:
        print(f"  WARNING: tickers not available and dropped: {dropped}", flush=True)

    stock_ret = returns_all[available_stocks].values.astype(np.float64)
    market_ret = returns_all[MARKET].values.astype(np.float64)  # (T,)
    dates = returns_all.index
    T, n = stock_ret.shape
    n_pairs = n * (n - 1) // 2
    print(f"Data aligned: T={T}, n_stocks={n} (dropped {len(dropped)}), n_pairs={n_pairs}", flush=True)

    # ------------------------------------------------------------------
    # 2. Model 1: CONST
    # ------------------------------------------------------------------
    print("\nModel 1: CONST")
    H_const = build_const_covariance(stock_ret)

    # ------------------------------------------------------------------
    # 3. Model 2: HIST100
    # ------------------------------------------------------------------
    print("Model 2: HIST100")
    H_hist100, hist_start = build_hist100_covariance(stock_ret)
    eval_start = hist_start  # All models evaluated from this index
    print(f"  Evaluation period starts at index {eval_start} ({dates[eval_start].date()})")

    # ------------------------------------------------------------------
    # 4. Model 3: DCC on GJR-GARCH standardized stock returns
    # ------------------------------------------------------------------
    print("Model 3: DCC MacGyver")
    h_stocks = np.empty((T, n), dtype=np.float64)
    S_stocks = np.empty((T, n), dtype=np.float64)

    for i, ticker in enumerate(available_stocks):
        h_i, s_i = estimate_gjr_garch(stock_ret[:, i])
        h_stocks[:, i] = h_i
        S_stocks[:, i] = s_i

    Rbar_stocks = np.corrcoef(S_stocks.T)
    print(f"  Estimating {n_pairs} pairwise DCC models (unrestricted)...")
    alpha_u, beta_u = run_macgyver_estimation(S_stocks, restricted=False)
    agg = macgyver_aggregate(alpha_u, beta_u)
    dcc_alpha = agg["MED"]["alpha"]
    dcc_beta  = agg["MED"]["beta"]
    print(f"  DCC median: alpha={dcc_alpha:.4f} (paper ~0.0157), beta={dcc_beta:.4f} (paper ~0.9755)")

    Rt_dcc = reconstruct_full_dcc(S_stocks, dcc_alpha, dcc_beta, Rbar_stocks)
    D_stocks = np.sqrt(h_stocks)  # (T, n) conditional std devs
    # H_t = D_t * R_t * D_t  (element-wise scaling)
    H_dcc = D_stocks[:, :, None] * Rt_dcc * D_stocks[:, None, :]

    # ------------------------------------------------------------------
    # 5. Model 4: FACTOR ARCH
    # ------------------------------------------------------------------
    print("Model 4: FACTOR ARCH")
    h_market, s_market = estimate_symmetric_garch(market_ret)
    betas, residuals = estimate_factor_betas(stock_ret, market_ret)
    idio_var_const = np.var(residuals, axis=0, ddof=1)
    H_factor_arch = factor_arch_covariance(betas, h_market, idio_var_const)

    # ------------------------------------------------------------------
    # 6. Model 5: FACTOR DOUBLE ARCH
    # ------------------------------------------------------------------
    print("Model 5: FACTOR DOUBLE ARCH")
    h_idio = np.empty((T, n), dtype=np.float64)
    std_resid_idio = np.empty((T, n), dtype=np.float64)

    for i in range(n):
        h_i, s_i = estimate_symmetric_garch(residuals[:, i])
        h_idio[:, i] = h_i
        std_resid_idio[:, i] = s_i

    H_factor_double_arch = factor_double_arch_covariance(betas, h_market, h_idio)

    # ------------------------------------------------------------------
    # 7. Model 6: FACTOR DCC
    # ------------------------------------------------------------------
    print("Model 6: FACTOR DCC")
    # Stack standardized idiosyncratic residuals + market residual: T x (n+1)
    e_all = np.column_stack([std_resid_idio, s_market])  # (T, n+1)
    Rbar_resid = np.corrcoef(e_all.T)

    print("  Estimating pairwise DCC on standardized residuals (unrestricted)...")
    alpha_r, beta_r = run_macgyver_estimation(e_all, restricted=False)
    agg_r = macgyver_aggregate(alpha_r, beta_r)
    fdcc_alpha = agg_r["MED"]["alpha"]
    fdcc_beta  = agg_r["MED"]["beta"]
    print(f"  Factor DCC median: alpha={fdcc_alpha:.4f} (paper ~0.009), beta={fdcc_beta:.4f} (paper ~0.925)")

    Rt_resid = reconstruct_full_dcc(e_all, fdcc_alpha, fdcc_beta, Rbar_resid)
    H_factor_dcc = factor_dcc_covariance(betas, h_market, h_idio, Rt_resid)

    # ------------------------------------------------------------------
    # 8. Economic evaluation over common evaluation period
    # ------------------------------------------------------------------
    print("\nEconomic evaluation...")
    H_models: Dict[str, np.ndarray] = {
        "CONST":       H_const[eval_start:],
        "HIST100":     H_hist100[eval_start:],
        "DCC":         H_dcc[eval_start:],
        "FACTOR_ARCH": H_factor_arch[eval_start:],
        "DOUBLE_ARCH": H_factor_double_arch[eval_start:],
        "FACTOR_DCC":  H_factor_dcc[eval_start:],
    }
    stock_ret_eval = stock_ret[eval_start:]
    T_eval = stock_ret_eval.shape[0]
    model_names = list(H_models.keys())

    perf_results: Dict = {}
    for model_name, H in H_models.items():
        res = evaluate_all_pairs(stock_ret_eval, H)
        perf_results[model_name] = res
        print(
            f"  {model_name:12s}: avg min-var vol={res['avg_minvar_vol']:.6f}, "
            f"avg hedge vol={res['avg_hedge_vol']:.6f}"
        )

    # Paper benchmarks for reference
    paper_minvar = {
        "CONST": 0.015764, "HIST100": 0.015889, "DCC": 0.015632,
        "FACTOR_ARCH": 0.015743, "DOUBLE_ARCH": 0.015628, "FACTOR_DCC": 0.015604,
    }
    paper_hedge = {
        "CONST": 0.018989, "HIST100": 0.019072, "DCC": 0.018844,
        "FACTOR_ARCH": 0.019026, "DOUBLE_ARCH": 0.018883, "FACTOR_DCC": 0.018804,
    }

    # ------------------------------------------------------------------
    # 9. Win-fraction matrices
    # ------------------------------------------------------------------
    N_models = len(model_names)
    wf_minvar = np.zeros((N_models, N_models), dtype=np.float64)
    wf_hedge  = np.zeros((N_models, N_models), dtype=np.float64)

    for ia, mA in enumerate(model_names):
        for ib, mB in enumerate(model_names):
            if ia != ib:
                wf_minvar[ia, ib] = np.mean(
                    perf_results[mA]["minvar_vols"] < perf_results[mB]["minvar_vols"]
                )
                wf_hedge[ia, ib] = np.mean(
                    perf_results[mA]["hedge_vols"] < perf_results[mB]["hedge_vols"]
                )

    wf_minvar_df = pd.DataFrame(wf_minvar, index=model_names, columns=model_names)
    wf_hedge_df  = pd.DataFrame(wf_hedge,  index=model_names, columns=model_names)

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    perf_df = pd.DataFrame({
        m: {
            "avg_minvar_vol": perf_results[m]["avg_minvar_vol"],
            "avg_hedge_vol":  perf_results[m]["avg_hedge_vol"],
        }
        for m in model_names
    }).T
    perf_df.to_csv(rdir / "exp2_performance_table.csv")
    wf_minvar_df.to_csv(rdir / "exp2_win_fractions_minvar.csv")
    wf_hedge_df.to_csv(rdir / "exp2_win_fractions_hedge.csv")

    # DCC parameter estimates
    dcc_params = pd.DataFrame({
        "DCC_on_returns":   {"alpha": dcc_alpha, "beta": dcc_beta},
        "FACTOR_DCC_resid": {"alpha": fdcc_alpha, "beta": fdcc_beta},
    }).T
    dcc_params.to_csv(rdir / "exp2_dcc_params.csv")

    # ------------------------------------------------------------------
    # 11. Plots
    # ------------------------------------------------------------------

    # Plot 1: Average DCC correlation over time
    eval_dates = dates[eval_start:]
    avg_corr_dcc = np.empty(T - eval_start, dtype=np.float64)
    triu = np.triu_indices(n, k=1)
    for t in range(eval_start, T):
        avg_corr_dcc[t - eval_start] = np.mean(Rt_dcc[t][triu])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eval_dates, avg_corr_dcc, linewidth=0.8, color="navy", label="Avg DCC correlation")
    ax.set_title("Average DCC Correlation Over Time (18 U.S. Large-Cap Stocks)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(rdir / "exp2_avg_dcc_correlation.png", dpi=150)
    plt.close()

    # Plot 2: Performance bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, paper_vals, title in zip(
        axes,
        ["avg_minvar_vol", "avg_hedge_vol"],
        [paper_minvar, paper_hedge],
        ["Avg Min-Var Portfolio Realized Vol", "Avg Hedge Portfolio Realized Vol"],
    ):
        vals = perf_df[metric]
        paper_v = [paper_vals.get(m, np.nan) for m in model_names]
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w / 2, vals.values, w, label="Reproduced", color="steelblue", edgecolor="black")
        ax.bar(x + w / 2, paper_v, w, label="Paper (target)", color="coral", edgecolor="black", alpha=0.7)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel("Realized Volatility")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Experiment 2: Economic Evaluation", fontsize=13)
    plt.tight_layout()
    plt.savefig(rdir / "exp2_performance_comparison.png", dpi=150)
    plt.close()

    # Plot 3: Win-fraction heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, wf, title in zip(
        axes,
        [wf_minvar_df, wf_hedge_df],
        ["Min-Var Win Fractions (row beats column)", "Hedge Win Fractions (row beats column)"],
    ):
        sns.heatmap(
            wf, annot=True, fmt=".2f", cmap="YlGnBu",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
        )
        ax.set_title(title)
    plt.suptitle("Experiment 2: Win-Fraction Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(rdir / "exp2_win_fractions.png", dpi=150)
    plt.close()

    # Plot 4: Cross-sectional std dev of DCC correlations over time
    std_corr_dcc = np.empty(T - eval_start, dtype=np.float64)
    for t in range(eval_start, T):
        std_corr_dcc[t - eval_start] = np.std(Rt_dcc[t][triu], ddof=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eval_dates, std_corr_dcc, linewidth=0.8, color="darkred",
            label="Std dev of DCC correlations")
    ax.set_title("Cross-sectional Std Dev of DCC Correlations Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std Dev of Correlations")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(rdir / "exp2_std_dcc_correlation.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {rdir}/")
    return {
        "perf_table": perf_df,
        "win_minvar": wf_minvar_df,
        "win_hedge":  wf_hedge_df,
        "dcc_alpha":  dcc_alpha,
        "dcc_beta":   dcc_beta,
        "fdcc_alpha": fdcc_alpha,
        "fdcc_beta":  fdcc_beta,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_exp2()
