# Bolt's Performance Journal

## 2026-03-31 - Vectorized Bootstrap Resampling and Monte Carlo in StatisticalValidator
**Learning:** Sequential Python loops in bootstrap resampling (10,000 iterations) and Monte Carlo simulations (5,000 iterations) introduced massive runtime overhead (~3.5 seconds per run) due to loop interpretation and iterative calls to `np.polyfit`. Computing linear regression $R^2$ scores across 2D NumPy matrices using algebraic Pearson correlation squared ($r^2 = \frac{\text{Cov}(X,Y)^2}{\text{Var}(X)\text{Var}(Y)}$) bypassed sequential OLS fitting completely.
**Action:** Always vectorize multi-iteration simulation or resampling algorithms into 2D NumPy array operations (`axis=1`) and use algebraic Pearson correlation squared for linear $R^2$ trend calculations across matrices.
