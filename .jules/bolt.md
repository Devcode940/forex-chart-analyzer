## 2025-02-12 - [Statistical Confidence Validator Vectorization]
**Learning:** Sequential OLS regression using `np.polyfit` inside a large loop (e.g. bootstrap resampling) incurs massive Python call overhead. Fully vectorizing OLS formulas and path simulation using 2D NumPy arrays leads to extreme performance gains.
**Action:** Reconstruct price paths via 2D cumprod and compute trend R² scores by vectorizing algebraic OLS formulas directly over matrices instead of using iterative loops or sequential function calls.
