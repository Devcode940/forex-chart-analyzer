"""
Liquidity Zone Detector Module
Identifies areas where stop-loss orders cluster (liquidity pools).
Smart money often targets these zones to fill large orders before reversing.

Key concept: When everyone puts their SL above a swing high, that's liquidity.
Price often sweeps that zone before reversing.
"""

import numpy as np

class LiquidityDetector:
    """
    Detects liquidity pools, equal highs/lows, and stop hunt zones.
    These are areas where retail traders cluster their stop losses.
    """

    def __init__(self):
        self.liquidity_zones = []

    def detect_all(self, price_series: dict, structure_results: dict,
                   sr_results: dict) -> dict:
        """
        Detect liquidity zones from price structure.
        Returns buy-side and sell-side liquidity pools.
        """
        smoothed = np.array(price_series.get("smoothed", []))
        x_positions = price_series.get("x", [])

        if len(smoothed) < 10:
            return {"buy_side_liquidity": [], "sell_side_liquidity": [],
                    "liquidity_sweeps": [], "stop_hunt_zones": []}

        # 1. Buy-side liquidity (above equal highs, above resistance)
        buy_side = self._detect_buy_side_liquidity(smoothed, x_positions, structure_results, sr_results)

        # 2. Sell-side liquidity (below equal lows, below support)
        sell_side = self._detect_sell_side_liquidity(smoothed, x_positions, structure_results, sr_results)

        # 3. Detect equal highs/lows (major liquidity magnets)
        equal_levels = self._detect_equal_levels(smoothed, x_positions, structure_results)

        # 4. Detect recent liquidity sweeps (stop hunts)
        sweeps = self._detect_liquidity_sweeps(smoothed, x_positions, structure_results)

        # 5. Identify stop hunt reversal zones
        stop_hunt_zones = self._identify_stop_hunt_zones(sweeps, buy_side, sell_side)

        return {
            "buy_side_liquidity": buy_side,
            "sell_side_liquidity": sell_side,
            "equal_levels": equal_levels,
            "liquidity_sweeps": sweeps,
            "stop_hunt_zones": stop_hunt_zones,
            "summary": self._generate_summary(buy_side, sell_side, sweeps, stop_hunt_zones),
        }

    def _detect_buy_side_liquidity(self, smoothed, x_pos, structure, sr) -> list:
        """
        Buy-side liquidity = where BUY stop losses cluster.
        These are above swing highs, above resistance, above equal highs.
        When swept, price typically reverses DOWN after.
        """
        zones = []
        swing_highs = structure.get("swing_highs", [])

        for sh in swing_highs:
            zones.append({
                "level": sh["value"],
                "x": sh.get("x", 0),
                "type": "BUY_SIDE",
                "source": "Swing High",
                "description": (
                    f"Liquidity above swing high at {sh['value']:.1f}. "
                    f"Buy stops cluster here. If swept, expect bearish reversal."
                ),
                "strategy": (
                    "If price sweeps above this level and rejects, "
                    "enter SHORT with SL above the sweep high. "
                    "This is a 'stop hunt' — smart money took out buy stops."
                ),
            })

        # Add resistance levels as liquidity
        for r in sr.get("resistance", []):
            zones.append({
                "level": r["price_level"],
                "x": 0,
                "type": "BUY_SIDE",
                "source": "Resistance Level",
                "description": (
                    f"Liquidity above resistance at {r['price_level']:.1f}. "
                    f"Buy stops above resistance = prime target for stop hunts."
                ),
                "strategy": "Watch for a false breakout above this resistance before shorting.",
            })

        return zones[:5]

    def _detect_sell_side_liquidity(self, smoothed, x_pos, structure, sr) -> list:
        """
        Sell-side liquidity = where SELL stop losses cluster.
        These are below swing lows, below support, below equal lows.
        When swept, price typically reverses UP after.
        """
        zones = []
        swing_lows = structure.get("swing_lows", [])

        for sl in swing_lows:
            zones.append({
                "level": sl["value"],
                "x": sl.get("x", 0),
                "type": "SELL_SIDE",
                "source": "Swing Low",
                "description": (
                    f"Liquidity below swing low at {sl['value']:.1f}. "
                    f"Sell stops cluster here. If swept, expect bullish reversal."
                ),
                "strategy": (
                    "If price sweeps below this level and rejects, "
                    "enter LONG with SL below the sweep low. "
                    "This is a 'stop hunt' — smart money took out sell stops."
                ),
            })

        for s in sr.get("support", []):
            zones.append({
                "level": s["price_level"],
                "x": 0,
                "type": "SELL_SIDE",
                "source": "Support Level",
                "description": (
                    f"Liquidity below support at {s['price_level']:.1f}. "
                    f"Sell stops below support = prime target for stop hunts."
                ),
                "strategy": "Watch for a false breakout below this support before buying.",
            })

        return zones[:5]

    def _detect_equal_levels(self, smoothed, x_pos, structure) -> list:
        """
        Equal highs/lows are the strongest liquidity magnets.
        When multiple swing points are at the same level, stops cluster there.
        """
        equal_levels = []

        swing_highs = structure.get("swing_highs", [])
        swing_lows = structure.get("swing_lows", [])

        # Check for equal highs
        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                h1 = swing_highs[i]["value"]
                h2 = swing_highs[j]["value"]
                tolerance = abs(h1) * 0.02  # 2% tolerance

                if abs(h1 - h2) < tolerance:
                    avg = (h1 + h2) / 2
                    equal_levels.append({
                        "level": avg,
                        "type": "EQUAL_HIGHS",
                        "count": 2,
                        "description": (
                            f"Equal highs at ~{avg:.1f}. "
                            f"Multiple touches = strong liquidity magnet. "
                            f"High probability of stop hunt above this level."
                        ),
                        "significance": "HIGH",
                    })

        # Check for equal lows
        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                l1 = swing_lows[i]["value"]
                l2 = swing_lows[j]["value"]
                tolerance = abs(l1) * 0.02

                if abs(l1 - l2) < tolerance:
                    avg = (l1 + l2) / 2
                    equal_levels.append({
                        "level": avg,
                        "type": "EQUAL_LOWS",
                        "count": 2,
                        "description": (
                            f"Equal lows at ~{avg:.1f}. "
                            f"Multiple touches = strong liquidity magnet. "
                            f"High probability of stop hunt below this level."
                        ),
                        "significance": "HIGH",
                    })

        return equal_levels

    def _detect_liquidity_sweeps(self, smoothed, x_pos, structure) -> list:
        """
        Detect recent liquidity sweeps (when price just took out a level).
        These are the MOST actionable signals — sweep + reversal = high-probability trade.
        """
        sweeps = []
        n = len(smoothed)

        if n < 5:
            return sweeps

        # Check the last 20% of the chart for recent sweeps
        recent_start = int(n * 0.8)
        recent = smoothed[recent_start:]

        swing_highs = structure.get("swing_highs", [])
        swing_lows = structure.get("swing_lows", [])

        for sh in swing_highs:
            sh_level = sh["value"]
            # Did recent price go above this level?
            for i, price in enumerate(recent):
                idx = i + recent_start
                if price > sh_level:

                    if i + 3 < len(recent) and recent[i + 3] < sh_level:
                        sweeps.append({
                            "level": sh_level,
                            "type": "BUY_SIDE_SWEEP",
                            "sweep_index": idx,
                            "description": (
                                f"Price swept above {sh_level:.1f} (took out buy stops) "
                                f"then reversed. Bull trap / stop hunt confirmed."
                            ),
                            "action": "SHORT — enter below the sweep level",
                            "sl_placement": "Above the sweep high",
                        })
                        break

        for sl in swing_lows:
            sl_level = sl["value"]
            for i, price in enumerate(recent):
                idx = i + recent_start
                if price < sl_level:
                    if i + 3 < len(recent) and recent[i + 3] > sl_level:
                        sweeps.append({
                            "level": sl_level,
                            "type": "SELL_SIDE_SWEEP",
                            "sweep_index": idx,
                            "description": (
                                f"Price swept below {sl_level:.1f} (took out sell stops) "
                                f"then reversed. Bear trap / stop hunt confirmed."
                            ),
                            "action": "LONG — enter above the sweep level",
                            "sl_placement": "Below the sweep low",
                        })
                        break

        return sweeps

    def _identify_stop_hunt_zones(self, sweeps, buy_side, sell_side) -> list:
        """
        Identify zones where a stop hunt is likely to occur next.
        Combines liquidity presence + sweep history to predict.
        """
        zones = []

        # If a liquidity level hasn't been swept yet, it's a target
        swept_levels = set(round(s["level"], 1) for s in sweeps)

        for bs in buy_side:
            level = round(bs["level"], 1)
            if level not in swept_levels:
                zones.append({
                    "level": bs["level"],
                    "type": "PENDING_STOP_HUNT_BUY",
                    "description": (
                        f"Buy-side liquidity at {bs['level']:.1f} has NOT been swept yet. "
                        f"This is a TARGET for smart money to hunt buy stops."
                    ),
                    "watch_for": (
                        "Watch for price to spike above this level then reject. "
                        "That rejection is your short entry signal."
                    ),
                })

        for ss in sell_side:
            level = round(ss["level"], 1)
            if level not in swept_levels:
                zones.append({
                    "level": ss["level"],
                    "type": "PENDING_STOP_HUNT_SELL",
                    "description": (
                        f"Sell-side liquidity at {ss['level']:.1f} has NOT been swept yet. "
                        f"This is a TARGET for smart money to hunt sell stops."
                    ),
                    "watch_for": (
                        "Watch for price to spike below this level then reject. "
                        "That rejection is your long entry signal."
                    ),
                })

        return zones[:4]

    def _generate_summary(self, buy_side, sell_side, sweeps, stop_hunt_zones) -> dict:
        """Generate a human-readable summary of the liquidity landscape."""
        return {
            "buy_side_pools": len(buy_side),
            "sell_side_pools": len(sell_side),
            "recent_sweeps": len(sweeps),
            "pending_hunt_zones": len(stop_hunt_zones),
            "interpretation": self._interpret_liquidity(buy_side, sell_side, sweeps),
        }

    def _interpret_liquidity(self, buy_side, sell_side, sweeps) -> str:
        """Generate natural language interpretation."""
        if len(sweeps) > 0:
            sweep_types = [s["type"] for s in sweeps]
            if any("BUY_SIDE" in st for st in sweep_types):
                return ("⚠️ Recent buy-side liquidity sweep detected. Smart money may have "
                        "triggered buy stops above a high. Watch for bearish reversal. "
                        "This is a high-probability short setup.")
            if any("SELL_SIDE" in st for st in sweep_types):
                return ("⚠️ Recent sell-side liquidity sweep detected. Smart money may have "
                        "triggered sell stops below a low. Watch for bullish reversal. "
                        "This is a high-probability long setup.")

        if len(buy_side) > len(sell_side):
            return ("More buy-side liquidity (above highs) than sell-side. "
                    "Smart money tends to target the side with more liquidity. "
                    "Beware of false breakouts to the upside.")
        elif len(sell_side) > len(buy_side):
            return ("More sell-side liquidity (below lows) than buy-side. "
                    "Watch for false breakdowns — price may dip below support then rally.")
        else:
            return ("Balanced liquidity on both sides. Wait for a sweep on one side "
                    "then trade the reversal.")

