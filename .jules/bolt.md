# Bolt's Journal - Critical Learnings

## 2025-05-18 - Vectorized Bootstrap OLS Pearson Correlation vs np.polyfit
**Learning:** Calling `np.polyfit` inside 10,000 bootstrap loop iterations creates severe Python overhead (~3 seconds). Computing simple linear regression $R^2$ as Pearson correlation squared over 2D matrices using `np.dot(y_dev, x_dev)` reduces execution time from 2983ms to 83ms (36x speedup).
**Action:** Always vectorize repeated 1D linear regression metrics into 2D matrix dot products instead of iterating `np.polyfit`.
