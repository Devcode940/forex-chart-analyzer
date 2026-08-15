## 2025-05-18 - Vectorize Bootstrap Resampling in StatisticalValidator

**Learning:** `StatisticalValidator.bootstrap_confidence_interval` ran 10,000 resamples in a sequential Python `for` loop, calling `np.polyfit` and Python generator expressions on each iteration, causing ~2.68s runtime. By converting resample path generation, win rate checks, volatility, path efficiency, and OLS linear regression $R^2$ into 2D NumPy matrix operations, execution time dropped from ~2.68s to ~0.23s (>11x speedup) with zero mathematical variance.

**Action:** For multi-iteration statistical resampling loops (Monte Carlo / Bootstrap), replace per-iteration loop calls with 2D array matrix operations and algebraic regression formulas.
