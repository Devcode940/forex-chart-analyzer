"""
Real Kelly Criterion Calculator
================================
Uses MEASURED (not estimated) win rates from the trade database.
This is the only mathematically optimal position sizing method.

Kelly Formula: f* = (p × b - q) / b
  where p = measured win rate, q = 1-p, b = avg_win / avg_loss

We use HALF-KELLY for safety (full Kelly is too aggressive).
"""

import numpy as np

from analyzers.trade_database import TradeDatabase


class RealKellyCalculator:
    """Calculate Kelly-optimal position sizing from MEASURED statistics."""

    def __init__(self, db: TradeDatabase = None):
        self.db = db or TradeDatabase()

    def calculate(
        self,
        setup_type: str,
        account_balance: float,
        sl_pips: float,
        pair: str = "EURUSD",
    ) -> dict:
        """
        Calculate Kelly-optimal position size from MEASURED data."""
        kelly_data = self.db.get_kelly_params(setup_type)

        if not kelly_data:
            return {
                "error": f"No measured data for setup type '{setup_type}'",
                "fallback_risk_pct": 0.5,
                "fallback_lots": self._calc_lots(account_balance, 0.005, sl_pips, pair),
            }

        # Measured parameters
        measured_wr = kelly_data["win_rate"]
        measured_avg_win = kelly_data["avg_win"]
        measured_avg_loss = kelly_data["avg_loss"]
        win_loss_ratio = kelly_data["win_loss_ratio"]
        sample_size = kelly_data["total_trades"]

        # ── Full Kelly ──
        # f* = (p*b - q) / b  where p=win_rate, q=1-p, b=win_loss_ratio
        p = measured_wr
        q = 1 - p
        b = win_loss_ratio

        full_kelly = (p * b - q) / b if b > 0 else 0

        # ── Half Kelly (recommended) ──
        half_kelly = full_kelly / 2

        # ── Quarter Kelly (conservative) ──
        quarter_kelly = full_kelly / 4

        # ── Confidence adjustment based on sample size ──
        # Small samples → reduce Kelly further
        if sample_size < 50:
            confidence_factor = sample_size / 100
        elif sample_size < 200:
            confidence_factor = 0.7 + (sample_size - 50) / 500
        else:
            confidence_factor = 1.0

        adjusted_half_kelly = half_kelly * confidence_factor

        # ── Cap at maximum risk ──
        max_risk = 0.02  # Never risk more than 2%
        final_risk_pct = min(adjusted_half_kelly, max_risk)

        # If Kelly is negative → no edge → don't trade
        if full_kelly <= 0:
            final_risk_pct = 0

        # ── Calculate lot size ──
        lots = self._calc_lots(account_balance, final_risk_pct, sl_pips, pair)

        # ── Expected value per trade ──
        expected_value = p * measured_avg_win - q * measured_avg_loss

        # ── Risk of ruin with this sizing ──
        ror = self._risk_of_ruin(p, final_risk_pct, b)

        # ── Time to double account ──
        if expected_value > 0 and final_risk_pct > 0:
            trades_to_double = int(
                np.log(2)
                / np.log(1 + expected_value * final_risk_pct / measured_avg_loss)
            )
        else:
            trades_to_double = float("inf")

        return {
            "setup_type": setup_type,
            "measured_win_rate": measured_wr,
            "measured_avg_win_pips": measured_avg_win,
            "measured_avg_loss_pips": measured_avg_loss,
            "win_loss_ratio": round(b, 2),
            "sample_size": sample_size,
            "full_kelly": round(full_kelly, 4),
            "half_kelly": round(half_kelly, 4),
            "quarter_kelly": round(quarter_kelly, 4),
            "confidence_factor": round(confidence_factor, 3),
            "adjusted_half_kelly": round(adjusted_half_kelly, 4),
            "final_risk_pct": round(final_risk_pct, 4),
            "recommended_lots": round(lots, 2),
            "risk_amount_usd": round(account_balance * final_risk_pct, 2),
            "expected_value_pips": round(expected_value, 2),
            "risk_of_ruin": round(ror, 4),
            "trades_to_double": (
                trades_to_double if trades_to_double != float("inf") else "N/A"
            ),
            "has_edge": full_kelly > 0,
            "recommendation": self._recommendation(
                full_kelly, measured_wr, sample_size, final_risk_pct
            ),
            "pip_value": self._pip_value(pair),
        }

    def _calc_lots(
        self, balance: float, risk_pct: float, sl_pips: float, pair: str
    ) -> float:
        """Calculate lot size from risk parameters."""
        pip_val = self._pip_value(pair)
        risk_amount = balance * risk_pct
        if sl_pips > 0 and pip_val > 0:
            lots = risk_amount / (sl_pips * pip_val)
            return min(lots, 100)  # Cap at 100 lots
        return 0

    def _pip_value(self, pair: str) -> float:
        """Get pip value per standard lot for position sizing.

        Values are approximate and assume standard lot sizes.
        XAUUSD uses $1/pip (1 oz lot); for 100-oz lots use $10/pip.
        Adjust based on your broker's contract specifications.
        """
        values = {
            "EURUSD": 10.0,
            "GBPUSD": 10.0,
            "AUDUSD": 10.0,
            "NZDUSD": 10.0,
            "USDJPY": 6.5,
            "EURJPY": 6.5,
            "GBPJPY": 6.5,
            "USDCHF": 10.5,
            "USDCAD": 7.5,
            "AUDJPY": 6.5,
            "XAUUSD": 1.0,
            "US30": 1.0,
            "NAS100": 1.0,
        }
        return values.get(pair, 10.0)

    def _risk_of_ruin(self, win_rate: float, risk_pct: float, wlr: float) -> float:
        """Simplified risk of ruin calculation."""
        if win_rate <= 0 or risk_pct <= 0:
            return 1.0

        q = 1 - win_rate
        if win_rate > q:
            ror = (q / win_rate) ** (1 / (risk_pct * wlr))
        else:
            ror = 1.0

        return float(np.clip(ror, 0, 1))

    def _recommendation(self, full_kelly: float, wr: float, n: int, risk: float) -> str:
        """Generate Kelly-based recommendation."""
        if full_kelly <= 0:
            return "🔴 No edge: Kelly is negative for this setup. Do not trade."

        if n < 30:
            return f"⚠️ INSUFFICIENT DATA: Only {n} trades. Kelly is unreliable with < 30 samples. Use minimum position size."

        if full_kelly < 0.05:
            return f"🟠 VERY THIN EDGE: Full Kelly = {full_kelly:.1%}. Edge exists but is small. Trade only with quarter-Kelly ({risk:.2%} risk)."

        if full_kelly < 0.15:
            return f"🟡 MODERATE EDGE: Full Kelly = {full_kelly:.1%}. Half-Kelly recommended ({risk:.2%} risk). {wr:.0%} measured win rate."

        if full_kelly < 0.30:
            return f"✅ GOOD EDGE: Full Kelly = {full_kelly:.1%}. Half-Kelly ({risk:.2%} risk) gives good growth with manageable drawdowns."

        return f"🟢 STRONG EDGE: Full Kelly = {full_kelly:.1%}. But DO NOT use full Kelly! Half-Kelly ({risk:.2%} risk) is the maximum recommended."
