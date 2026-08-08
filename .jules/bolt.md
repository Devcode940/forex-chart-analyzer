# Bolt's Journal - Forex Chart Analyzer Performance Learnings

## 2025-02-14 - Multi-run Regression and Sim Vectorization
**Learning:**
1. Streamlit apps execute the entire script from scratch on any tab transition or UI interaction. Expensive operations like ML training (RandomForest / GradientBoosting) and cross-validation must be cached using bounded `OrderedDict` caches with `feature_vector.tobytes()` as cache keys. Cache hits must fully restore the internal fitted state (e.g. fitted model attributes) to ensure other components referencing the instance have valid state.
2. Generating thousands of independent random paths (Monte Carlo/Bootstrap) with nested python loops is a massive performance bottleneck. These are 100% vectorizable with NumPy 2D matrices.
3. For simple linear regressions where the independent variable is a fixed linear range, calculating $R^2$ using `np.polyfit` inside loops has immense execution overhead. This can be elegantly computed as the square of the Pearson correlation coefficient over 2D NumPy matrices using basic algebraic formulas (covariance and variance), reducing computation time from seconds to milliseconds.

**Action:** Always vectorize iterative statistical computations (Monte Carlo, bootstrap) with 2D NumPy arrays and replace sequential regression fits with vectorized correlation-based $R^2$ formulas. Implement input-hash-based caching for models to bypass re-computation during Streamlit UI redraws.
