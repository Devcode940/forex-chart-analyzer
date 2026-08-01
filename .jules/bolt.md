# Bolt's Performance Journal

This journal tracks critical learning insights, architecture findings, and performance optimizations.

## 2025-08-01 - Vectorizing Monte Carlo & Bootstrap Resampling in Technical Analysis
**Learning:** In statistical and simulation pipelines, executing iterative sampling or regression inside Python nested loops introduces heavy overhead. In `StatisticalValidator`, looping 5,000 to 10,000 times to generate Brownian Motion paths using a nested scalar loop, and calculating trend R² values sequentially via `np.polyfit`, created a major bottleneck (taking ~1.92 seconds). `np.polyfit` performs a full least-squares solve using SVD or QR decomposition under the hood, which is highly unnecessary for 1D trendlines. Instead, the algebraic Ordinary Least Squares (OLS) formula can be vectorized over 2D NumPy matrices using basic matrix arithmetic, and Geometric Brownian Motion can be computed via vectorized `np.cumprod`.
**Action:** Always pre-allocate simulation matrices and use NumPy vectorized operators (`np.dot`, `np.cumprod(..., axis=1)`) to eliminate loop-based computation. For simple regressions inside bootstrap resampling or Monte Carlo simulations, compute R² algebraically over 2D matrices rather than calling `np.polyfit` or scikit-learn in a loop.
