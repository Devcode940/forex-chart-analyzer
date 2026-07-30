# ⚡ Bolt's Performance Journal

## 2025-02-17 - Vectorized OLS and Bootstrap Resampling
**Learning:** Sequential calls to `np.polyfit` inside a bootstrap resampling loop introduce massive overhead due to Python function-call wrapping and un-vectorized operations. By generating bootstrap paths in a 2D matrix (shape `n_bootstrap, n`) and solving Ordinary Least Squares (OLS) trends algebraically (`m = dot(reconstructed, dx) / var_x`), we bypass the loop completely.
**Action:** Vectorize multi-run linear regressions over 2D NumPy matrices using OLS algebraic formulation instead of looping `np.polyfit` sequential calls.
