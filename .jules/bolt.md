# Bolt's Optimization Journal

## 2025-08-03 - Vectorized Bootstrap Resampling and R² Calculations
**Learning:** Sequential processing in Python loops for bootstrap resampling (e.g., executing `np.polyfit` 5,000–10,000 times) creates a massive performance bottleneck due to excessive overhead. By leveraging NumPy’s 2D array broadcasting, we can reconstruct all price paths simultaneously and compute the trend R² scores elegantly using the vectorized algebraic Ordinary Least Squares (OLS) formula, bypassing `np.polyfit` entirely.
**Action:** Always vectorize loops involving statistical sampling or path generation using multi-dimensional NumPy matrices to parallelize computations on the CPU.
