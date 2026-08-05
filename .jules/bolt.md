# Bolt's Performance Journal

## 2025-02-15 - Vectorized Statistical Validation & OLS Trend R² Estimation
**Learning:** High-iteration Monte Carlo simulations and Bootstrap resamples suffer severely from Python loop overhead and repetitive function call overhead (such as `np.polyfit`). Since $R^2$ in simple linear regression is mathematically equivalent to the squared Pearson correlation coefficient, trend strength can be computed instantly across thousands of resampled paths in a single NumPy matrix expression: $\frac{SS_{xy}^2}{SS_{xx} \times SS_{yy}}$. This eliminates the need for loop-based OLS fitting entirely.
**Action:** When performing multi-run linear regressions or time-series path reconstructions, always vectorize the entire simulation in a single 2D NumPy array using `np.cumprod(..., axis=1)` and apply analytical algebraic formulas over the matrices rather than invoking sequential loop-based estimators.
