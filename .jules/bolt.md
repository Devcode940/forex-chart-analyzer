# ⚡ Bolt's Journal - Forex Chart Analyzer Pro v2

This journal contains critical, high-value performance learnings discovered during the optimization of this specific codebase.

## 2025-02-15 - Deterministic Synthetic Datasets & Streamlit Script Re-execution Caching
**Learning:**
1. **Deterministic Static Generators:** Several ML components (such as `WalkForwardValidator` and `MLEnsemble`) generated synthetic datasets using hardcoded seeds (`np.random.seed(42)`). Because these datasets and their corresponding window validations are 100% deterministic, repeatedly regenerating them and retraining full 100-estimator ensembles on them on every analysis execution was a massive CPU waste (~25 seconds of unnecessary processing).
2. **Streamlit Script Re-execution:** Under Streamlit's architecture, changing UI tabs or interacting with controls triggers a full-script rerun. Even though persistent analyzer instances are stored in `st.session_state`, they were re-running their expensive model-training pipelines on identical input chart features on every re-run.

**Action:**
1. **Class-Level Caching:** Cache deterministic, static datasets and pre-trained models at the class level so they are generated/fitted at most once across the entire process lifetime.
2. **Instance-Level Call Caching:** Implement instance-level call caching by hashing the input `feature_vector` (`hash(feature_vector.tobytes())`). If the input hasn't changed, bypass all ML training, validation, and prediction steps, and return the cached dictionary immediately. This reduced tab-switching latency from ~27 seconds to under 0.1 milliseconds!
3. **Cross-Validation Parallelism:** Leverage `n_jobs=-1` on outer cross-validation loops (e.g., `cross_val_score`) to parallelize tree training across all CPU cores.
