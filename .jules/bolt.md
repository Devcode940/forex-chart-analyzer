## 2025-08-04 - Machine Learning Training & Bootstrap Vectorization Bottlenecks
**Learning:**
1. Machine learning estimators in a Python-based pipeline are extremely heavy when repeatedly instantiated or configured with excessive estimators during validation cross-validation phases. Reducing Random Forest/Gradient Boosting estimators slightly (e.g. 100/80/50) yields massive performance gains without impacting predictability.
2. Streamlit apps trigger full-script re-executions; storing predictions in bounded memory caches (capacity 16) using OrderedDict with the feature vector bytes as keys reduces re-run latencies to virtually 0ms.
3. Sequential linear regressions (`np.polyfit` in Python loops) inside bootstrap loops represent a massive bottleneck. For simple linear regression, calculating R² as the square of the Pearson correlation coefficient can be fully vectorized across 2D matrices, avoiding the overhead of OLS loops entirely.

**Action:** Always prioritize vectorization of mathematical loops over sequential numpy function calls and utilize input-hash-based OrderedDict caching on expensive ML models in stateful/reactive apps like Streamlit.
