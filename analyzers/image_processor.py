"""
Image Processor Module — Enhanced
Handles image upload, preprocessing, adaptive color segmentation (KMeans + HSV),
chart grid extraction, and robust price series extraction with noise filtering.
"""

import cv2
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans


class ImageProcessor:
    """Preprocesses forex chart images for analysis with adaptive color detection."""

    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.edges = None

    def load_image(self, uploaded_file) -> np.ndarray:
        """Load an uploaded image file into a numpy array."""
        bytes_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        self.original_image = np.array(pil_image)
        return self.original_image

    def preprocess(self, image: np.ndarray) -> dict:
        """Run the full preprocessing pipeline with softer edge thresholds."""
        self.original_image = image
        results = {}

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.gray_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        results["gray"] = self.gray_image

        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        results["blurred"] = blurred

        # Softer thresholds catch more chart features
        self.edges = cv2.Canny(blurred, 30, 150)
        results["edges"] = self.edges

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        results["threshold"] = thresh

        return results

    def extract_chart_colors(self, image: np.ndarray) -> dict:
        """Adaptive KMeans + fixed HSV for robust candle color detection.

        Uses KMeans clustering on HSV pixel space to discover dominant chart
        colors automatically, then applies widened HSV ranges for reliable
        green/red candle extraction across different chart themes.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # ── Adaptive KMeans clustering ──
        h, w = hsv.shape[:2]
        # Subsample for speed (every 4th pixel)
        sample_rate = 4
        pixels = hsv[::sample_rate, ::sample_rate].reshape(-1, 3).astype(np.float32)

        # Filter out near-black and near-white pixels (background)
        brightness = pixels[:, 2]
        valid_mask = (brightness > 20) & (brightness < 240)
        valid_pixels = pixels[valid_mask]

        if len(valid_pixels) < 10:
            # Fallback: use all pixels
            valid_pixels = pixels

        unique_pixels = np.unique(valid_pixels, axis=0)
        n_clusters = min(6, max(2, len(unique_pixels)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
        kmeans.fit(valid_pixels)
        centers = kmeans.cluster_centers_

        # Classify KMeans clusters into color families
        green_cluster_pixels = 0
        red_cluster_pixels = 0
        for center in centers:
            h_val, s_val, v_val = center
            # Green range
            if 35 <= h_val <= 90 and s_val >= 30 and v_val >= 30:
                mask = cv2.inRange(hsv, np.array([max(0, h_val - 15), max(0, s_val - 40), max(0, v_val - 40)]),
                                   np.array([min(179, h_val + 15), min(255, s_val + 40), min(255, v_val + 40)]))
                green_cluster_pixels += cv2.countNonZero(mask)
            # Red range (wraps around 0/180)
            if (h_val <= 15 or h_val >= 165) and s_val >= 30 and v_val >= 30:
                lower_h = max(0, h_val - 15) if h_val <= 15 else max(165, h_val - 15)
                upper_h = min(15, h_val + 15) if h_val <= 15 else min(180, h_val + 15)
                mask = cv2.inRange(hsv, np.array([lower_h, max(0, s_val - 40), max(0, v_val - 40)]),
                                   np.array([upper_h, min(255, s_val + 40), min(255, v_val + 40)]))
                red_cluster_pixels += cv2.countNonZero(mask)

        # ── Fixed HSV ranges (widened for robustness) ──
        green_mask = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([90, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([12, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([168, 30, 30]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2
        blue_mask = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([140, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([40, 255, 255]))

        # Merge adaptive + fixed masks (union for maximum recall)
        if green_cluster_pixels > 0:
            # If KMeans found green, widen the green mask slightly
            green_mask_extra = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([95, 255, 255]))
            green_mask = green_mask | green_mask_extra

        # Stats
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        total = green_pixels + red_pixels or 1

        return {
            "green_mask": green_mask,
            "red_mask": red_mask,
            "blue_mask": blue_mask,
            "yellow_mask": yellow_mask,
            "green_pixels": green_pixels,
            "red_pixels": red_pixels,
            "blue_pixels": cv2.countNonZero(blue_mask),
            "yellow_pixels": cv2.countNonZero(yellow_mask),
            "bullish_ratio": green_pixels / total,
            "bearish_ratio": 1 - green_pixels / total,
            "sentiment": "BULLISH" if green_pixels / total > 0.52 else "BEARISH" if green_pixels / total < 0.48 else "NEUTRAL",
            "kmeans_centers": centers.tolist(),
            "green_cluster_pixels": green_cluster_pixels,
            "red_cluster_pixels": red_cluster_pixels,
        }

    def detect_grid_lines(self, edges: np.ndarray) -> dict:
        """Improved Hough with multiple parameter sets for robust grid detection."""
        # Primary detection
        h_lines = []
        v_lines = []

        all_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60,
                                     minLineLength=80, maxLineGap=15)
        if all_lines is not None:
            for line in all_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 8 and abs(x2 - x1) > 30:
                    h_lines.append((x1, y1, x2, y2))
                elif abs(x2 - x1) < 8 and abs(y2 - y1) > 30:
                    v_lines.append((x1, y1, x2, y2))

        # Secondary detection — looser parameters for faint grids
        all_lines2 = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                                      minLineLength=50, maxLineGap=25)
        if all_lines2 is not None:
            for line in all_lines2:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 8 and abs(x2 - x1) > 20:
                    h_lines.append((x1, y1, x2, y2))
                elif abs(x2 - x1) < 8 and abs(y2 - y1) > 20:
                    v_lines.append((x1, y1, x2, y2))

        # Merge (deduplicate by proximity)
        h_lines = self._merge_lines(h_lines, horizontal=True)
        v_lines = self._merge_lines(v_lines, horizontal=False)

        return {
            "horizontal_lines": h_lines,
            "vertical_lines": v_lines,
            "h_line_count": len(h_lines),
            "v_line_count": len(v_lines)
        }

    def _filter_lines(self, lines, horizontal=True):
        """Filter lines by orientation."""
        if lines is None:
            return []
        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if horizontal:
                if abs(y2 - y1) < 8:
                    filtered.append((x1, y1, x2, y2))
            else:
                if abs(x2 - x1) < 8:
                    filtered.append((x1, y1, x2, y2))
        return filtered

    def _merge_lines(self, lines, horizontal=True, gap=10):
        """Merge nearby duplicate lines."""
        if not lines:
            return []
        # Sort by position
        key = (lambda l: l[1]) if horizontal else (lambda l: l[0])
        lines = sorted(lines, key=key)
        merged = [lines[0]]
        for line in lines[1:]:
            prev = merged[-1]
            if horizontal:
                if abs(line[1] - prev[1]) > gap:
                    merged.append(line)
            else:
                if abs(line[0] - prev[0]) > gap:
                    merged.append(line)
        return merged

    def extract_price_series(self, image: np.ndarray, color_mask: np.ndarray) -> list:
        """Extract price data points with IQR noise filtering for robustness."""
        h, w = color_mask.shape
        points = []
        num_slices = min(w, 250)
        slice_width = w / num_slices

        for i in range(num_slices):
            x_start = int(i * slice_width)
            x_end = int((i + 1) * slice_width)
            column = color_mask[:, x_start:x_end]
            nonzero_rows = np.where(column > 0)[0]

            if len(nonzero_rows) > 3:
                # IQR outlier removal
                q1, q3 = np.percentile(nonzero_rows, [25, 75])
                iqr = q3 - q1
                mask = (nonzero_rows >= q1 - 1.5 * iqr) & (nonzero_rows <= q3 + 1.5 * iqr)
                clean_rows = nonzero_rows[mask]

                if len(clean_rows) > 0:
                    top_y = int(clean_rows.min())
                    bottom_y = int(clean_rows.max())
                    center_y = (top_y + bottom_y) // 2
                    center_x = (x_start + x_end) // 2
                    points.append({
                        "x": center_x,
                        "center_y": center_y,
                        "top_y": top_y,
                        "bottom_y": bottom_y,
                        "height": bottom_y - top_y
                    })

        return points

    def get_price_scale_estimate(self, image, right_edge_fraction=0.1):
        """Basic right-axis price scale estimation.

        Returns a dict with estimated price range and y-to-price mapping.
        Full OCR integration is deferred to a future enhancement.
        """
        h, w = image.shape[:2]
        right_strip = image[:, int(w * (1 - right_edge_fraction)):]
        gray = cv2.cvtColor(right_strip, cv2.COLOR_RGB2GRAY)

        # Try to detect horizontal text regions as price labels
        edges = cv2.Canny(gray, 50, 150)
        horizontal = cv2.HoughLinesP(edges, 1, np.pi / 2, 30, minLineLength=20, maxLineGap=5)

        price_levels = []
        if horizontal is not None:
            for line in horizontal:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:
                    price_levels.append(y1)

        if len(price_levels) >= 2:
            price_levels = sorted(set(price_levels))
            y_range = price_levels[-1] - price_levels[0]
            if y_range > 0:
                return {
                    "estimated_price_range": y_range,
                    "y_pixel_to_price": 1.0 / y_range,
                    "detected_levels": price_levels,
                    "method": "hough_line_estimation"
                }

        return {
            "estimated_price_range": None,
            "y_pixel_to_price": None,
            "detected_levels": [],
            "method": "no_levels_found"
        }

    def get_image_stats(self, image: np.ndarray) -> dict:
        """Get basic image statistics."""
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
            "mean_brightness": np.mean(self.gray_image) if self.gray_image is not None else 0,
        }
