# Bolt Performance Journal

## 2026-08-12 - Vectorization of Bootstrap Confidence Intervals
**Learning:** Found a significant bottleneck in `StatisticalValidator.bootstrap_confidence_interval` where Monte Carlo/Bootstrap resamples were computed using a sequential Python loop. Specifically, `_calc_trend_r2` was invoking `np.polyfit` sequentially up to 10,000 times. Vectorizing both the random resampling, the Geometric Brownian Motion path reconstructions, and the metrics computation (volatility, R², efficiency ratio, and win rates) using 2D NumPy operations reduces execution time from ~6.0s down to ~0.13s for 10,000 resamples (a ~45x speedup).
**Action:** Replace the sequential `for _ in range(n_bootstrap)` loop in `analyzers/statistical_validator.py` with 2D NumPy vectorization while maintaining exact mathematical equivalence.
