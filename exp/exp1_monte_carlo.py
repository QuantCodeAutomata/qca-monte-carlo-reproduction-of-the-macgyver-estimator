"""
Experiment 1: Monte Carlo Reproduction of the MacGyver Estimator
for High-Dimensional DCC Parameters (Engle 2007).

Runs 100 independent Monte Carlo replications for each of 10 simulation
designs and reports bias / RMSE of six aggregate estimators:
    MEAN, MED, MEANT05 (unrestricted)
    R_MEAN, R_MED, R_MEANT05 (restricted / logit-parameterized)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.dcc_estimation import macgyver_aggregate, run_macgyver_estimation
from src.dcc_simulation import generate_dcc_data

MAX_WORKERS: int = min(4, os.cpu_count() or 1)

# ---------------------------------------------------------------------------
# Simulation designs (Table 1 in Engle 2007)
# ---------------------------------------------------------------------------

DESIGNS = [
    {"id": 1,  "n": 3,  "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50},
    {"id": 2,  "n": 5,  "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50},
    {"id": 3,  "n": 10, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50},
    {"id": 4,  "n": 20, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50},
    {"id": 5,  "n": 30, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50},
    {"id": 6,  "n": 50, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.50},
    {"id": 7,  "n": 10, "alpha": 0.05, "beta": 0.94, "rho_bar": 0.50},
    {"id": 8,  "n": 10, "alpha": 0.02, "beta": 0.97, "rho_bar": 0.50},
    {"id": 9,  "n": 10, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.20},
    {"id": 10, "n": 10, "alpha": 0.05, "beta": 0.90, "rho_bar": 0.80},
]

N_REPS = 100
T = 1000


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def _rep_worker(args: Tuple) -> Dict[str, Tuple[float, float]]:
    """Top-level picklable wrapper for ProcessPoolExecutor.

    Args:
        args: Tuple (n, alpha, beta, rho_bar, seed, T).

    Returns:
        Dict mapping estimator name to (alpha_hat, beta_hat).
    """
    n, alpha, beta, rho_bar, seed, series_T = args
    return run_single_replication(n, alpha, beta, rho_bar, seed, series_T)


def run_single_replication(
    n: int,
    alpha: float,
    beta: float,
    rho_bar: float,
    seed: int,
    T: int = T,
) -> Dict[str, Tuple[float, float]]:
    """Run a single Monte Carlo replication.

    Generates DCC data, estimates all pairwise DCCs (unrestricted and
    restricted), and returns the six MacGyver aggregate estimates.

    Args:
        n:       Cross-sectional dimension.
        alpha:   True DCC alpha.
        beta:    True DCC beta.
        rho_bar: Equicorrelation for Rbar.
        seed:    Random seed for this replication.
        T:       Sample size.

    Returns:
        Dict mapping estimator name to (alpha_hat, beta_hat).
    """
    S, _, _ = generate_dcc_data(n, T, alpha, beta, rho_bar, seed=seed)

    # Unrestricted pairwise estimation
    a_u, b_u = run_macgyver_estimation(S, restricted=False)
    agg_u = macgyver_aggregate(a_u, b_u)

    # Restricted (logit-parameterized) pairwise estimation
    a_r, b_r = run_macgyver_estimation(S, restricted=True)
    agg_r = macgyver_aggregate(a_r, b_r)

    return {
        "MEAN":     (agg_u["MEAN"]["alpha"],    agg_u["MEAN"]["beta"]),
        "MED":      (agg_u["MED"]["alpha"],     agg_u["MED"]["beta"]),
        "MEANT05":  (agg_u["MEANT05"]["alpha"], agg_u["MEANT05"]["beta"]),
        "R_MEAN":   (agg_r["MEAN"]["alpha"],    agg_r["MEAN"]["beta"]),
        "R_MED":    (agg_r["MED"]["alpha"],     agg_r["MED"]["beta"]),
        "R_MEANT05":(agg_r["MEANT05"]["alpha"], agg_r["MEANT05"]["beta"]),
    }


# ---------------------------------------------------------------------------
# Design runner
# ---------------------------------------------------------------------------

def run_design(
    design: Dict,
    n_reps: int = N_REPS,
    master_seed: int = 42,
    series_T: int = T,
) -> pd.DataFrame:
    """Run all replications for a single simulation design.

    Args:
        design:      Design dictionary with keys: n, alpha, beta, rho_bar.
        n_reps:      Number of Monte Carlo replications.
        master_seed: Master seed for deterministic sub-seed generation.
        series_T:    Sample size per replication (default: module-level T=1000).

    Returns:
        DataFrame of shape (n_reps, 12) with columns {METHOD}_alpha / {METHOD}_beta.
    """
    rng = np.random.default_rng(master_seed + design["id"])
    seeds = rng.integers(0, 2 ** 31, size=n_reps)

    worker_args: List[Tuple] = [
        (design["n"], design["alpha"], design["beta"],
         design["rho_bar"], int(seeds[rep]), series_T)
        for rep in range(n_reps)
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        replication_results = list(
            pool.map(_rep_worker, worker_args, chunksize=max(1, n_reps // (MAX_WORKERS * 2)))
        )

    rows = []
    for rep, r in enumerate(replication_results):
        row: Dict = {"rep": rep}
        for method, (a_hat, b_hat) in r.items():
            row[f"{method}_alpha"] = a_hat
            row[f"{method}_beta"] = b_hat
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_bias_rmse(
    estimates: np.ndarray, true_val: float
) -> Tuple[float, float]:
    """Compute Monte Carlo bias and RMSE.

    Args:
        estimates: (n_reps,) array of estimates.
        true_val:  True parameter value.

    Returns:
        (bias, rmse) tuple.
    """
    errors = estimates - true_val
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return bias, rmse


def summarize_design_results(
    df: pd.DataFrame,
    true_alpha: float,
    true_beta: float,
) -> pd.DataFrame:
    """Compute bias and RMSE for all six estimators across replications.

    Args:
        df:          Raw replication DataFrame from run_design().
        true_alpha:  Design true alpha.
        true_beta:   Design true beta.

    Returns:
        Summary DataFrame indexed by estimator with columns:
        alpha_bias, alpha_rmse, beta_bias, beta_rmse.
    """
    methods = ["MEAN", "MED", "MEANT05", "R_MEAN", "R_MED", "R_MEANT05"]
    rows = []
    for m in methods:
        a_bias, a_rmse = compute_bias_rmse(df[f"{m}_alpha"].values, true_alpha)
        b_bias, b_rmse = compute_bias_rmse(df[f"{m}_beta"].values, true_beta)
        rows.append({
            "estimator": m,
            "alpha_bias": a_bias,
            "alpha_rmse": a_rmse,
            "beta_bias":  b_bias,
            "beta_rmse":  b_rmse,
        })
    return pd.DataFrame(rows).set_index("estimator")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_designs(
    n_reps: int = N_REPS,
    master_seed: int = 42,
) -> Dict:
    """Run all 10 simulation designs.

    Args:
        n_reps:      Number of replications per design.
        master_seed: Master random seed.

    Returns:
        Dict keyed by design id with sub-keys: 'design', 'raw', 'stats'.
    """
    import time

    summary: Dict = {}
    for design in DESIGNS:
        t0 = time.time()
        print(
            f"  Design {design['id']:2d}: "
            f"n={design['n']:2d}, alpha={design['alpha']:.2f}, "
            f"beta={design['beta']:.2f}, rho_bar={design['rho_bar']:.2f} ...",
            flush=True,
        )
        df = run_design(design, n_reps=n_reps, master_seed=master_seed)
        elapsed = time.time() - t0
        summary[design["id"]] = {
            "design": design,
            "raw": df,
            "stats": summarize_design_results(
                df, design["alpha"], design["beta"]
            ),
        }
        print(f"    -> done in {elapsed:.1f}s", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # --- numba JIT warm-up (populates disk cache used by pool workers) ---
    from src.dcc_estimation import _dcc_nll_core
    _dummy = np.ones(10, dtype=np.float64)
    _dcc_nll_core(0.05, 0.90, 0.5, _dummy, _dummy)
    print("numba JIT cache warmed up.", flush=True)

    print("=" * 70)
    print("Experiment 1: Monte Carlo MacGyver Estimator (Engle 2007)")
    print(f"  Designs: {len(DESIGNS)} | Replications: {N_REPS} | T: {T}")
    print("=" * 70)

    _t_all = _time.time()
    summary = run_all_designs(n_reps=N_REPS, master_seed=42)
    print(f"\nTotal wall-clock: {_time.time()-_t_all:.1f}s", flush=True)

    # Print and collect tables
    bias_rows = []
    rmse_rows = []
    for did, data in summary.items():
        stats = data["stats"]
        d = data["design"]
        label = f"D{did}:n={d['n']},a={d['alpha']},b={d['beta']}"
        print(
            f"\nDesign {did}: n={d['n']}, alpha={d['alpha']}, "
            f"beta={d['beta']}, rho_bar={d['rho_bar']}"
        )
        print(stats.round(6).to_string())
        for est in stats.index:
            bias_rows.append({
                "design": label,
                "estimator": est,
                "alpha_bias": stats.loc[est, "alpha_bias"],
                "beta_bias": stats.loc[est, "beta_bias"],
            })
            rmse_rows.append({
                "design": label,
                "estimator": est,
                "alpha_rmse": stats.loc[est, "alpha_rmse"],
                "beta_rmse": stats.loc[est, "beta_rmse"],
            })

    bias_df = (
        pd.DataFrame(bias_rows)
        .pivot(index="design", columns="estimator", values=["alpha_bias", "beta_bias"])
    )
    rmse_df = (
        pd.DataFrame(rmse_rows)
        .pivot(index="design", columns="estimator", values=["alpha_rmse", "beta_rmse"])
    )

    bias_df.to_csv(results_dir / "exp1_bias_table.csv")
    rmse_df.to_csv(results_dir / "exp1_rmse_table.csv")

    # Summary averages across all designs
    methods = ["MEAN", "MED", "MEANT05", "R_MEAN", "R_MED", "R_MEANT05"]
    avg_rows = []
    for m in methods:
        avg_a_rmse = np.mean([summary[d]["stats"].loc[m, "alpha_rmse"] for d in summary])
        avg_b_rmse = np.mean([summary[d]["stats"].loc[m, "beta_rmse"] for d in summary])
        avg_a_bias = np.mean([summary[d]["stats"].loc[m, "alpha_bias"] for d in summary])
        avg_b_bias = np.mean([summary[d]["stats"].loc[m, "beta_bias"] for d in summary])
        avg_rows.append({
            "estimator": m,
            "avg_alpha_bias": avg_a_bias,
            "avg_alpha_rmse": avg_a_rmse,
            "avg_beta_bias": avg_b_bias,
            "avg_beta_rmse": avg_b_rmse,
        })
    avg_df = pd.DataFrame(avg_rows).set_index("estimator")
    avg_df.to_csv(results_dir / "exp1_avg_stats.csv")

    print("\n" + "=" * 70)
    print("Average statistics across all 10 designs:")
    print(avg_df.round(6).to_string())
    print("\nBenchmark (Engle 2007): MED alpha RMSE ~0.00501, beta RMSE ~0.01499")
    print("Benchmark (Engle 2007): R_MED alpha RMSE ~0.00532, beta RMSE ~0.01529")

    # Bar chart: average RMSE by estimator
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, param in zip(axes, ["alpha", "beta"]):
        avg_df[f"avg_{param}_rmse"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Average {param} RMSE across 10 designs")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Estimator")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Experiment 1: MacGyver Estimator Monte Carlo", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_avg_rmse.png", dpi=150)
    plt.close()

    # Heatmap of alpha RMSE per design
    rmse_alpha = pd.DataFrame(
        {d: summary[d]["stats"]["alpha_rmse"] for d in summary}
    ).T
    rmse_alpha.index = [f"D{d}" for d in summary]
    fig, ax = plt.subplots(figsize=(10, 6))
    import seaborn as sns
    sns.heatmap(rmse_alpha, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax)
    ax.set_title("Alpha RMSE by Design and Estimator")
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_alpha_rmse_heatmap.png", dpi=150)
    plt.close()

    print("\nResults saved to results/")
