"""
Image Processor Module
Handles image upload, preprocessing, color segmentation, and chart grid extraction.
"""

import cv2
import numpy as np
from PIL import Image
import io


class ImageProcessor:
    """Preprocesses forex chart images for analysis."""

    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.gray_image = None
        self.edges = None

    def load_image(self, uploaded_file) -> np.ndarray:
        """Load an uploaded image file into a numpy array."""
        bytes_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        self.original_image = np.array(pil_image)
        return self.original_image

    def preprocess(self, image: np.ndarray) -> dict:
        """Run the full preprocessing pipeline."""
        self.original_image = image
        results = {}

        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.gray_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        results["gray"] = self.gray_image

        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        results["blurred"] = blurred

        # Edge detection (Canny)
        self.edges = cv2.Canny(blurred, 50, 150)
        results["edges"] = self.edges

        # Adaptive threshold for better line detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        results["threshold"] = thresh

        return results

    def extract_chart_colors(self, image: np.ndarray) -> dict:
        """Extract dominant colors to identify bullish/bearish candles and indicators."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Green (bullish) range
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Red (bearish) range
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 40, 40])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

        # Blue indicators
        blue_lower = np.array([100, 40, 40])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Yellow indicators
        yellow_lower = np.array([20, 40, 40])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        total_colored = green_pixels + red_pixels
        bullish_ratio = green_pixels / total_colored if total_colored > 0 else 0.5

        return {
            "green_mask": green_mask,
            "red_mask": red_mask,
            "blue_mask": blue_mask,
            "yellow_mask": yellow_mask,
            "green_pixels": green_pixels,
            "red_pixels": red_pixels,
            "blue_pixels": blue_pixels,
            "yellow_pixels": yellow_pixels,
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": 1 - bullish_ratio,
            "sentiment": "BULLISH" if bullish_ratio > 0.55 else "BEARISH" if bullish_ratio < 0.45 else "NEUTRAL"
        }

    def detect_grid_lines(self, edges: np.ndarray) -> dict:
        """Detect horizontal and vertical grid lines using Hough transforms."""
        # Horizontal lines
        horizontal_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 2, threshold=80,
            minLineLength=100, maxLineGap=10
        )

        # Vertical lines
        vertical_lines = cv2.HoughLinesP(
            edges, 1, 0, threshold=80,
            minLineLength=100, maxLineGap=10
        )

        h_lines = []
        v_lines = []

        if horizontal_lines is not None:
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:  # Nearly horizontal
                    h_lines.append((x1, y1, x2, y2))

        if vertical_lines is not None:
            for line in vertical_lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 5:  # Nearly vertical
                    v_lines.append((x1, y1, x2, y2))

        return {
            "horizontal_lines": h_lines,
            "vertical_lines": v_lines,
            "h_line_count": len(h_lines),
            "v_line_count": len(v_lines)
        }

    def extract_price_series(self, image: np.ndarray, color_mask: np.ndarray) -> list:
        """Extract approximate price data points from colored regions."""
        h, w = color_mask.shape
        points = []

        # Divide image into vertical slices
        num_slices = min(w, 200)
        slice_width = w / num_slices

        for i in range(num_slices):
            x_start = int(i * slice_width)
            x_end = int((i + 1) * slice_width)
            column = color_mask[:, x_start:x_end]

            # Find top and bottom of colored region in this slice
            rows = np.where(column > 0)
            if len(rows[0]) > 0:
                top_y = rows[0].min()
                bottom_y = rows[0].max()
                center_y = (top_y + bottom_y) // 2
                center_x = (x_start + x_end) // 2
                points.append({
                    "x": center_x,
                    "top_y": top_y,
                    "bottom_y": bottom_y,
                    "center_y": center_y,
                    "height": bottom_y - top_y
                })

        return points

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
