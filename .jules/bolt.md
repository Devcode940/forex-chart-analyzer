## 2026-03-29 - Vectorized Bootstrap Linear Regression & Path Calculation
**Learning:** Sequential `np.polyfit` calls inside a Python bootstrap loop (e.g., 5,000–10,000 resamples) create significant overhead. Vectorizing $R^2$ trend strength via algebraic Pearson correlation squared ($r^2$) over a 2D matrix reduces computation time by ~30x (~1.5s to ~0.05s) while producing identical results.
**Action:** Replace `np.polyfit` in resampling/Monte Carlo loops with 2D matrix algebraic OLS / Pearson $r^2$ formulas (`cov(x, y)^2 / (var(x) * var(y))`).
