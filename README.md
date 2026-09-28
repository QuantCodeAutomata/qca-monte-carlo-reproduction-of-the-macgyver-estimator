# MacGyver Estimator for High-Dimensional DCC Parameters

Reproduction of the simulation study and empirical analysis in **Engle (2007)**:
*"High Dimensional Dynamic Correlations"*, evaluating pairwise bivariate DCC
estimation followed by aggregation (the "MacGyver estimator") as a scalable
approach to estimating common DCC parameters in high-dimensional systems.

---

## Repository Structure

```
repo/
├── data/                   # Data loading utilities (Massive REST API)
│   └── data_loader.py
├── src/                    # Core implementations
│   ├── dcc_simulation.py   # DCC data generation (Engle 2007 DGP)
│   ├── dcc_estimation.py   # Pairwise DCC + MacGyver aggregation
│   ├── factor_models.py    # FACTOR ARCH / DOUBLE ARCH / FACTOR DCC
│   └── portfolio_eval.py   # Min-variance & hedge portfolio evaluation
├── exp/                    # Experiment scripts
│   ├── exp1_monte_carlo.py # Exp 1: Monte Carlo MacGyver
│   └── exp2_empirical.py   # Exp 2: Empirical 18-stock comparison
├── tests/                  # pytest test suite
│   ├── test_dcc_simulation.py
│   ├── test_dcc_estimation.py
│   ├── test_portfolio_eval.py
│   ├── test_factor_models.py
│   └── test_exp1_integration.py
├── results/                # Outputs: tables, plots, RESULTS.md
│   └── RESULTS.md
├── requirements.txt
└── README.md
```

---

## Experiments

### Experiment 1 — Monte Carlo (Synthetic Data)

Reproduces the Monte Carlo simulation study from Engle (2007), Table 1–3.

- **10 simulation designs**: varying n ∈ {3,5,10,20,30,50}, alpha ∈ {0.02,0.05},
  beta ∈ {0.90,0.94,0.97}, and ρ̄ ∈ {0.20,0.50,0.80}.
- **100 replications** per design, T=1000 each.
- **6 aggregation estimators**: MEAN, MED, MEANT05 (unrestricted) and
  R_MEAN, R_MED, R_MEANT05 (restricted/logit-parameterized).
- Reports **bias** and **RMSE** for alpha and beta.

```bash
python exp/exp1_monte_carlo.py
```

### Experiment 2 — Empirical (18 U.S. Large-Cap Stocks, 1994–2004)

Reproduces the empirical covariance model comparison from Engle (2007).
Requires a valid `MASSIVE_TOKEN` environment variable.

- **6 covariance models**: CONST, HIST100, DCC, FACTOR ARCH, DOUBLE ARCH, FACTOR DCC.
- **Economic evaluation**: realized volatility of min-variance and hedge portfolios
  across all 153 stock pairs.
- Outputs performance table and win-fraction matrices.

```bash
MASSIVE_TOKEN=your_key python exp/exp2_empirical.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Key Results (Experiment 1)

See `results/RESULTS.md` for full tables. Key finding:

> The **median** of unrestricted pairwise DCC estimates (MED) consistently
> achieves the best or near-best RMSE for both alpha and beta across all
> 10 simulation designs, confirming Engle's (2007) recommendation.

---

## References

Engle, R. F. (2007). *High Dimensional Dynamic Correlations*.
In R. Blundell, T. Persson, & W. K. Newey (Eds.), *The Econometrics of Panel Data:
Fundamentals and Recent Developments in Theory and Practice*.
Blackwell: Oxford.
