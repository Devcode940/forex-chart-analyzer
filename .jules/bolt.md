# Bolt's Performance Journal

## 2025-05-18 - Vectorize OLS Trend Strength Calculation in Bootstrap Metrics
**Learning:** Sequential calls to `np.polyfit` inside a 10,000-iteration loop incur significant Python overhead and overhead from generic polynomial fitting logic. Expressing linear regression R² via vectorized 2D matrix algebraic OLS formulas (`cov^2 / (var_x * var_y)`) computes all resampled trend strengths simultaneously across all bootstrap samples in a single matrix operation, reducing execution time from ~2.8s to ~0.07s (>37x speedup).
**Action:** Replace iterative 1D regression fittings in resampling loops with 2D matrix algebraic OLS formulas.
