# Bolt's Performance Journal

## 2025-02-14 - Vectorizing Resampling and Trend $R^2$ inside Bootstrap Confidence Interval
**Learning:** Sequential Python loops calling `np.polyfit` inside 10,000 bootstrap simulations are extremely slow. By generating all resample indices at once as a 2D matrix, we can vectorized the price reconstruction path, volatility calculation, and trend $R^2$ using the Pearson correlation squared formula.
**Action:** Replace the loop in `bootstrap_confidence_interval` with vectorized NumPy operations to achieve a >20x speedup with zero mathematical difference.
