# Bolt's Performance Journal

This file tracks critical performance optimization insights, failures, and architectural findings.

## 2025-02-15 - Vectorized Monte Carlo and Bootstrap Resampling
**Learning:** Heuristic confidence validation using Monte Carlo simulations (3,000 runs) and Bootstrap resampling (5,000 runs) initially suffered from slow execution times (~2.0 seconds total) due to Python nested loops and redundant calculations. By fully vectorizing these simulations using multidimensional NumPy arrays, we avoided Python-level iterations, generating all paths and computing statistical properties (returns, trend line R² fit, volatility, win rates, and efficiency ratios) in parallel. Reconstructed path lengths must strictly match original non-vectorized shapes (length `n`) to maintain correct trend line and index boundaries.
**Action:** Always favor multidimensional NumPy array generation (`np.random.normal(..., size=(n_simulations, n_bars - 1))`) and vectorized aggregation (`axis=1`) over Python `for` loops when validating statistical models or conducting resampling.
