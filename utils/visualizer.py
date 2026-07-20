"""
Visualizer Module
Creates annotated chart overlays with detected patterns, S/R levels, and SL/TP markers.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io


class Visualizer:
    """Creates visual overlays on forex chart images."""

    # Color scheme
    COLORS = {
        "support": (0, 255, 0),         # Green
        "resistance": (255, 0, 0),       # Red
        "confluence": (255, 165, 0),     # Orange
        "bullish_pattern": (0, 200, 100), # Teal Green
        "bearish_pattern": (255, 50, 50), # Bright Red
        "neutral_pattern": (255, 255, 0), # Yellow
        "sl_line": (255, 0, 255),        # Magenta
        "tp_line": (0, 255, 255),        # Cyan
        "trend_line": (100, 149, 237),   # Cornflower Blue
        "swing_high": (255, 100, 100),   # Light Red
        "swing_low": (100, 255, 100),    # Light Green
        "bos_bullish": (0, 255, 200),    # Aqua
        "bos_bearish": (255, 100, 0),    # Orange Red
        "text_bg": (0, 0, 0, 180),      # Semi-transparent black
        "fib_382": (255, 165, 0),       # Orange 38.2%
        "fib_500": (255, 255, 0),       # Yellow 50%
        "fib_618": (255, 100, 0),       # Deep Orange 61.8%
        "fib_786": (200, 50, 0),        # Dark Orange 78.6%
        "fib_ext": (0, 200, 255),       # Light Blue extensions
        "liquidity_buy": (255, 50, 50), # Red buy-side
        "liquidity_sell": (50, 255, 50),# Green sell-side
        "divergence_bull": (0, 255, 200),# Aqua bullish div
        "divergence_bear": (255, 50, 150),# Pink bearish div
        "candlestick_bull": (100, 255, 150),# Light green
        "candlestick_bear": (255, 100, 100),# Light red
    }

    def __init__(self):
        self.annotated_image = None

    def create_full_overlay(self, original_image: np.ndarray,
                            pattern_results: list,
                            sr_results: dict,
                            structure_results: dict,
                            sltp_results: dict,
                            regime_results: dict) -> np.ndarray:
        """Create a comprehensive annotated overlay."""
        # Convert to PIL for easier text rendering
        pil_img = Image.fromarray(original_image)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        h, w = original_image.shape[:2]

        # Draw S/R levels
        self._draw_sr_levels(draw, sr_results, w)

        # Draw pattern zones
        self._draw_patterns(draw, pattern_results, h)

        # Draw structure points
        self._draw_structure(draw, structure_results)

        # Draw SL/TP levels
        self._draw_sltp(draw, sltp_results, w)

        # Draw regime info panel
        self._draw_regime_panel(draw, regime_results, w, h)

        # Draw summary panel
        self._draw_summary_panel(draw, pattern_results, sr_results, sltp_results, w, h)

        self.annotated_image = np.array(pil_img)
        return self.annotated_image

    def _draw_sr_levels(self, draw: ImageDraw.Draw, sr_results: dict, width: int):
        """Draw support and resistance lines."""
        for level in sr_results.get("support", []):
            y = level.get("image_y", 0)
            strength = level.get("strength", 0.5)
            color = self.COLORS["support"]
            alpha = int(100 + strength * 155)

            # Dashed line effect
            for x in range(0, width, 8):
                draw.line([(x, y), (min(x + 4, width), y)],
                         fill=(*color, alpha), width=2)

            # Label
            draw.rectangle([(5, y - 10), (85, y + 10)], fill=(0, 0, 0, 160))
            draw.text((10, y - 8), f"S ({strength:.0%})", fill=(*color, 255))

        for level in sr_results.get("resistance", []):
            y = level.get("image_y", 0)
            strength = level.get("strength", 0.5)
            color = self.COLORS["resistance"]
            alpha = int(100 + strength * 155)

            for x in range(0, width, 8):
                draw.line([(x, y), (min(x + 4, width), y)],
                         fill=(*color, alpha), width=2)

            draw.rectangle([(5, y - 10), (85, y + 10)], fill=(0, 0, 0, 160))
            draw.text((10, y - 8), f"R ({strength:.0%})", fill=(*color, 255))

        for zone in sr_results.get("key_zones", []):
            y = int(zone.get("level", 0))
            # Can't directly use image_y for confluence, so approximate
            draw.rectangle([(0, y - 3), (width, y + 3)],
                          fill=(*self.COLORS["confluence"], 80))

    def _draw_patterns(self, draw: ImageDraw.Draw, patterns: list, img_height: int):
        """Draw pattern zones and labels."""
        for i, pattern in enumerate(patterns[:5]):  # Top 5 patterns
            name = pattern.get("name", "")
            ptype = pattern.get("type", "")
            confidence = pattern.get("confidence", 0.5)
            x_range = pattern.get("x_range", (0, 0))

            # Choose color
            if "BULLISH" in ptype:
                color = self.COLORS["bullish_pattern"]
            elif "BEARISH" in ptype:
                color = self.COLORS["bearish_pattern"]
            else:
                color = self.COLORS["neutral_pattern"]

            # Draw pattern zone
            x1, x2 = x_range
            if x1 > 0 or x2 > 0:
                draw.rectangle(
                    [(x1, 20 + i * 35), (x2, 50 + i * 35)],
                    outline=(*color, 150), width=2
                )

            # Draw pattern label
            label = f"📊 {name} ({confidence:.0%})"
            draw.rectangle([(width_offset := 100, 20 + i * 35),
                           (width_offset + 250, 50 + i * 35)],
                          fill=(0, 0, 0, 180))
            draw.text((width_offset + 5, 25 + i * 35), label, fill=(*color, 255))

    def _draw_structure(self, draw: ImageDraw.Draw, structure: dict):
        """Draw swing points and structure breaks."""
        # Swing highs
        for sh in structure.get("swing_highs", []):
            x = sh.get("x", 0)
            y = sh.get("price_y", 0)
            # Draw triangle marker
            draw.polygon([(x, y - 12), (x - 8, y), (x + 8, y)],
                        fill=(*self.COLORS["swing_high"], 200))
            draw.text((x + 10, y - 15), "SH", fill=(*self.COLORS["swing_high"], 200))

        # Swing lows
        for sl in structure.get("swing_lows", []):
            x = sl.get("x", 0)
            y = sl.get("price_y", 0)
            # Draw inverted triangle
            draw.polygon([(x, y + 12), (x - 8, y), (x + 8, y)],
                        fill=(*self.COLORS["swing_low"], 200))
            draw.text((x + 10, y + 5), "SL", fill=(*self.COLORS["swing_low"], 200))

        # Breaks of structure
        for bos in structure.get("structure_breaks", []):
            x = bos.get("x", 0)
            y = bos.get("price_y", 0)
            bos_type = bos.get("type", "")

            if "BULLISH" in bos_type:
                color = self.COLORS["bos_bullish"]
                label = "▲ BOS"
            else:
                color = self.COLORS["bos_bearish"]
                label = "▼ BOS"

            draw.ellipse([(x - 12, y - 12), (x + 12, y + 12)],
                        outline=(*color, 200), width=2)
            draw.text((x + 15, y - 8), label, fill=(*color, 255))

    def _draw_sltp(self, draw: ImageDraw.Draw, sltp_results: dict, width: int):
        """Draw SL/TP levels for the best scenario."""
        best = sltp_results.get("best_scenario")
        if not best:
            return

        direction = best.get("direction", "BUY")
        entry = best.get("entry", 0)
        sl = best.get("sl", 0)
        tp = best.get("tp", 1)
        rr = best.get("risk_reward", 0)

        # Note: These are in inverted price coordinates
        # We'll draw them as horizontal lines with labels
        sl_label = f"Stop Loss: {sl:.1f}"
        tp_label = f"Take Profit: {tp:.1f} (R:R = {rr:.2f})"
        direction_label = f"{'🟢' if direction == 'BUY' else '🔴'} {direction}"

        # Draw info box at bottom of image
        img_height = draw.im.size[1]
        box_y = img_height - 80

        draw.rectangle([(10, box_y), (350, img_height - 10)],
                      fill=(0, 0, 0, 200))

        draw.text((15, box_y + 5), direction_label,
                 fill=(0, 255, 0, 255) if direction == "BUY" else (255, 0, 0, 255))
        draw.text((15, box_y + 25), sl_label,
                 fill=(*self.COLORS["sl_line"], 255))
        draw.text((15, box_y + 45), tp_label,
                 fill=(*self.COLORS["tp_line"], 255))

    def _draw_regime_panel(self, draw: ImageDraw.Draw, regime: dict, w: int, h: int):
        """Draw market regime info panel."""
        regime_name = regime.get("regime", "UNKNOWN")
        sub_regime = regime.get("sub_regime", "UNKNOWN")
        confidence = regime.get("confidence", 0)
        trading_style = regime.get("trading_style", "")
        risk_level = regime.get("risk_level", {}).get("level", "")

        # Draw panel on top-right
        panel_w = 300
        panel_h = 120
        draw.rectangle([(w - panel_w - 10, 10), (w - 10, panel_h + 10)],
                      fill=(0, 0, 0, 200))

        x = w - panel_w - 5
        y = 15

        # Regime indicator
        regime_colors = {
            "TRENDING": (0, 255, 100),
            "RANGING": (255, 255, 0),
            "VOLATILE": (255, 50, 50),
            "TRANSITIONAL": (255, 165, 0)
        }
        color = regime_colors.get(regime_name, (200, 200, 200))

        draw.text((x, y), f"REGIME: {regime_name}", fill=(*color, 255))
        draw.text((x, y + 20), f"Sub: {sub_regime}", fill=(200, 200, 200, 255))
        draw.text((x, y + 40), f"Confidence: {confidence:.0%}", fill=(200, 200, 200, 255))
        draw.text((x, y + 60), f"Style: {trading_style[:30]}", fill=(200, 200, 200, 255))
        draw.text((x, y + 80), f"Risk: {risk_level}", fill=(200, 200, 200, 255))

    def _draw_summary_panel(self, draw: ImageDraw.Draw, patterns: list,
                            sr: dict, sltp: dict, w: int, h: int):
        """Draw analysis summary panel."""
        panel_w = 280
        panel_h = 100

        # Position on left side
        draw.rectangle([(10, h - panel_h - 90), (10 + panel_w, h - 90)],
                      fill=(0, 0, 0, 200))

        x = 15
        y = h - panel_h - 85

        n_patterns = len(patterns)
        n_supports = len(sr.get("support", []))
        n_resistances = len(sr.get("resistance", []))
        bias = sltp.get("bias", "NEUTRAL")

        draw.text((x, y), "━━━ ANALYSIS SUMMARY ━━━", fill=(255, 255, 255, 255))
        draw.text((x, y + 20), f"Patterns Found: {n_patterns}", fill=(255, 255, 0, 255))
        draw.text((x, y + 40), f"S/R Levels: {n_supports}S / {n_resistances}R", fill=(255, 255, 0, 255))

        bias_color = {
            "BULLISH": (0, 255, 0),
            "BEARISH": (255, 0, 0),
            "NEUTRAL": (255, 255, 0)
        }
        draw.text((x, y + 60), f"Bias: {bias}",
                 fill=(*bias_color.get(bias, (255, 255, 255)), 255))

        best = sltp.get("best_scenario")
        if best:
            draw.text((x, y + 80),
                     f"Best Setup: R:R {best.get('risk_reward', 0):.2f}",
                     fill=(0, 255, 255, 255))

    def create_pattern_detail_image(self, pattern: dict, img_height: int) -> np.ndarray:
        """Create a detailed visualization for a specific pattern."""
        # Create a blank image
        canvas = np.ones((300, 400, 4), dtype=np.uint8) * 30
        pil_canvas = Image.fromarray(canvas, "RGBA")
        draw = ImageDraw.Draw(pil_canvas, "RGBA")

        name = pattern.get("name", "Unknown Pattern")
        ptype = pattern.get("type", "")
        confidence = pattern.get("confidence", 0)
        desc = pattern.get("description", "")

        # Draw pattern info
        draw.text((10, 10), f"Pattern: {name}", fill=(255, 255, 255, 255))
        draw.text((10, 30), f"Type: {ptype}", fill=(200, 200, 200, 255))
        draw.text((10, 50), f"Confidence: {confidence:.0%}", fill=(200, 200, 200, 255))

        # Word-wrap description
        words = desc.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) > 50:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)

        for i, line in enumerate(lines):
            draw.text((10, 80 + i * 18), line, fill=(180, 180, 180, 255))

        return np.array(pil_canvas)
