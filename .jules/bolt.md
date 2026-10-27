## 2026-10-27 - KMeans Subsampling in Image Processing
**Learning:** `KMeans.fit` in `extract_chart_colors` was running on thousands of HSV pixels with `n_init=10`, causing ~225ms latency per image. Subsampling valid HSV pixels to max 1,000 samples and using `n_init=1` with `k-means++` initialization reduced latency to ~18ms (~92% speedup) without affecting color clustering accuracy.
**Action:** Subsample input pixel space for unsupervised clustering in image processing pipelines when exact full-pixel coverage is redundant.
