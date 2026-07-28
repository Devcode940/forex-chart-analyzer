# Bolt's Optimization Journal

## 2025-07-28 - Vectorized Monte Carlo and Bootstrap Resampling
**Learning:** High-iteration Monte Carlo simulations and Bootstrap resampling loops in Python are slow and block CPU execution. Leveraging 2D NumPy operations to generate indices/paths in parallel and using vectorized statistical calculations (like vectorized OLS for trend R2 calculation) reduces execution time by ~11x (from ~3.24s down to ~0.29s for 5k sims/10k resamples).
**Action:** Identify loop-based simulations and metrics generation and rewrite using multi-dimensional NumPy arrays and parallelized ufunc execution instead of Python generators or explicit loops.
