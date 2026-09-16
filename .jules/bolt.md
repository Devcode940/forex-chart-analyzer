# Bolt's Performance Journal

## 2025-05-15 - Vectorizing Image Mask Column Slicing in IndicatorDetector
**Learning:** Extracting points from image masks by iteratively slicing a 2D array in a Python loop (e.g., 100 iterations per color mask call) creates noticeable function call overhead. Reshaping uniform 2D slice ranges into a 3D matrix `(h, num_slices, slice_width)` allows computing slice pixel counts and weighted y-position sums using 2D matrix multiplication (`row_counts.T @ np.arange(h)`), achieving >2x speed improvement for mask line point extraction.
**Action:** For image processing routines that slice binary masks into vertical/horizontal bins, use 3D array reshaping and matrix multiplication instead of looping over image slices in Python.
