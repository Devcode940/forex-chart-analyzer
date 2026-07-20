"""
Support & Resistance Detector Module
Identifies key support and resistance zones from price data.
"""

import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN


class SRDetector:
    """Detects Support and Resistance levels from chart data."""

    def __init__(self):
        self.support_levels = []
        self.resistance_levels = []

    def detect(self, price_series: dict, image_height: int) -> dict:
        """Full support/resistance detection pipeline."""
        smoothed = np.array(price_series.get("smoothed", []))
        x_positions = price_series.get("x", [])
        highs = price_series.get("highs", [])
        lows = price_series.get("lows", [])

        if len(smoothed) < 5:
            return {"support": [], "resistance": [], "key_zones": []}

        # Method 1: Swing-based detection
        swing_sr = self._swing_based_detection(smoothed, x_positions)

        # Method 2: Cluster-based detection
        cluster_sr = self._cluster_based_detection(smoothed)

        # Method 3: Volume-based (approximation using price density)
        density_sr = self._density_based_detection(smoothed)

        # Merge all detected levels
        all_supports = swing_sr["supports"] + cluster_sr["supports"] + density_sr["supports"]
        all_resistances = swing_sr["resistances"] + cluster_sr["resistances"] + density_sr["resistances"]

        # Consolidate nearby levels
        self.support_levels = self._consolidate_levels(all_supports)
        self.resistance_levels = self._consolidate_levels(all_resistances)

        # Identify key zones (confluence areas)
        key_zones = self._find_confluence_zones()

        # Convert to image coordinates for visualization
        support_with_coords = []
        for sl in self.support_levels:
            support_with_coords.append({
                "price_level": sl["level"],
                "strength": sl["strength"],
                "touches": sl["touches"],
                "image_y": int(image_height - sl["level"]),
                "type": "SUPPORT"
            })

        resistance_with_coords = []
        for rl in self.resistance_levels:
            resistance_with_coords.append({
                "price_level": rl["level"],
                "strength": rl["strength"],
                "touches": rl["touches"],
                "image_y": int(image_height - rl["level"]),
                "type": "RESISTANCE"
            })

        return {
            "support": support_with_coords,
            "resistance": resistance_with_coords,
            "key_zones": key_zones
        }

    def _swing_based_detection(self, smoothed: np.ndarray, x_positions: list) -> dict:
        """Detect S/R from swing highs and lows."""
        order = max(3, len(smoothed) // 15)

        maxima_idx = argrelextrema(smoothed, np.greater, order=order)[0]
        minima_idx = argrelextrema(smoothed, np.less, order=order)[0]

        supports = [{"level": float(smoothed[i]), "strength": 0.6, "touches": 1} for i in minima_idx]
        resistances = [{"level": float(smoothed[i]), "strength": 0.6, "touches": 1} for i in maxima_idx]

        return {"supports": supports, "resistances": resistances}

    def _cluster_based_detection(self, smoothed: np.ndarray) -> dict:
        """Detect S/R using DBSCAN clustering of price levels."""
        # Sample price points
        prices = smoothed.reshape(-1, 1)

        # Normalize
        price_range = prices.max() - prices.min()
        if price_range < 1e-6:
            return {"supports": [], "resistances": []}

        normalized = (prices - prices.min()) / price_range

        # Cluster
        clustering = DBSCAN(eps=0.03, min_samples=max(3, len(smoothed) // 20)).fit(normalized)
        labels = clustering.labels_

        supports = []
        resistances = []

        for label in set(labels):
            if label == -1:
                continue
            cluster_points = prices[labels == label]
            cluster_level = float(np.mean(cluster_points))
            cluster_size = len(cluster_points)
            cluster_strength = min(1.0, cluster_size / (len(smoothed) * 0.1))

            level = {"level": cluster_level, "strength": cluster_strength, "touches": cluster_size}

            # Classify as support or resistance based on position relative to median
            median = np.median(smoothed)
            if cluster_level < median:
                supports.append(level)
            else:
                resistances.append(level)

        return {"supports": supports, "resistances": resistances}

    def _density_based_detection(self, smoothed: np.ndarray) -> dict:
        """Detect S/R based on price density (where price spent most time)."""
        # Create histogram
        n_bins = min(50, len(smoothed) // 2)
        if n_bins < 5:
            return {"supports": [], "resistances": []}

        hist, bin_edges = np.histogram(smoothed, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in the histogram
        threshold = np.mean(hist) + np.std(hist)
        peak_indices = np.where(hist > threshold)[0]

        median = np.median(smoothed)

        supports = []
        resistances = []

        for idx in peak_indices:
            level = float(bin_centers[idx])
            strength = min(1.0, hist[idx] / (np.max(hist) + 1e-6))
            touches = int(hist[idx])

            level_data = {"level": level, "strength": strength, "touches": touches}

            if level < median:
                supports.append(level_data)
            else:
                resistances.append(level_data)

        return {"supports": supports, "resistances": resistances}

    def _consolidate_levels(self, levels: list, tolerance: float = 0.03) -> list:
        """Merge nearby S/R levels into stronger zones."""
        if not levels:
            return []

        # Sort by level
        levels.sort(key=lambda x: x["level"])

        consolidated = []
        current_group = [levels[0]]

        for level in levels[1:]:
            if abs(level["level"] - current_group[-1]["level"]) / (current_group[-1]["level"] + 1e-6) < tolerance:
                current_group.append(level)
            else:
                # Merge group
                merged = {
                    "level": float(np.mean([l["level"] for l in current_group])),
                    "strength": min(1.0, sum(l["strength"] for l in current_group) / len(current_group) + 0.1 * len(current_group)),
                    "touches": sum(l.get("touches", 1) for l in current_group)
                }
                consolidated.append(merged)
                current_group = [level]

        # Don't forget last group
        merged = {
            "level": float(np.mean([l["level"] for l in current_group])),
            "strength": min(1.0, sum(l["strength"] for l in current_group) / len(current_group) + 0.1 * len(current_group)),
            "touches": sum(l.get("touches", 1) for l in current_group)
        }
        consolidated.append(merged)

        # Sort by strength
        consolidated.sort(key=lambda x: x["strength"], reverse=True)

        return consolidated[:6]  # Keep top 6 levels

    def _find_confluence_zones(self) -> list:
        """Find zones where support and resistance are close (confluence)."""
        zones = []
        for sl in self.support_levels:
            for rl in self.resistance_levels:
                distance = abs(sl["level"] - rl["level"]) / (sl["level"] + 1e-6)
                if distance < 0.02:
                    zones.append({
                        "level": (sl["level"] + rl["level"]) / 2,
                        "strength": sl["strength"] + rl["strength"],
                        "type": "CONFLUENCE"
                    })
        return zones
