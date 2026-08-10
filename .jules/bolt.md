# Bolt's Performance Journal

This journal tracks critical optimization insights, failures, and architectural findings.

## 2025-02-14 - Vectorizing Bootstrap in Forex Chart Analyzer Pro
**Learning:**
Bootstrap confidence intervals and Monte Carlo simulations can easily become performance bottlenecks if implemented with nested Python loops, repeated array allocation/reconstruction, and sequential mathematical function calls (like `np.polyfit`). Since NumPy operations are heavily optimized in C, we can achieve substantial speedups (>25x) by representing the entire simulation/resampling workflow as a single 2D NumPy array execution. Additionally, R² trend calculations can be vectorized elegantly as the square of the Pearson correlation coefficient over 2D NumPy matrices, completely bypassing the need to invoke `np.polyfit` sequentially.
**Action:**
Always vectorize simulation-based algorithms with 2D/3D NumPy operations, keeping loop iterations out of raw Python. Standardize OLS trend metrics on algebraic Pearson correlation coefficient formulas when performing linear regressions on large batches.
