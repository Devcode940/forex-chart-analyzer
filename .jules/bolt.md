# ⚡ Bolt's Performance Journal

This journal tracks critical performance insights, architectural findings, and lessons learned while optimizing the Forex Chart Analyzer codebase.

## 2025-02-14 - Vectorizing Statistical Simulations & OLS R²
**Learning:** Sequential Python loops calling NumPy/SciPy functions (like `np.polyfit`, `np.std`, or custom Brownian motion loops) suffer from extreme overhead when executed thousands of times (e.g., 5,000 bootstrap resamples or 2,000 Monte Carlo paths). Vectorizing these computations entirely using multi-dimensional NumPy arrays and replacing `np.polyfit` with an algebraic Ordinary Least Squares (OLS) formula for simple linear regression reduces execution time by ~93% (from 1.5 seconds to 120ms for 5k bootstraps). It is crucial to maintain exact path lengths (`n` rather than `n+1` for reconstructed price series) to match downstream expectation matrices and statistical confidence bounds perfectly.
**Action:** Always favor high-dimensional NumPy matrix operations over custom loop-based simulation/resampling models in python-based quant/trading architectures. Precompute static/independent variables (like regression X values, deviations, and variances) outside any vectorized calculation block.
