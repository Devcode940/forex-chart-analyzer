"""
Risk Management & Position Size Calculator
Calculates proper position sizing based on account size, risk %, and SL distance.
Converts abstract price units to pip values for major forex pairs.
"""

import numpy as np


class RiskManager:
    """
    Professional position sizing calculator.
    Handles: account risk, lot sizing, pip value, max drawdown, risk of ruin.
    """

    # Standard pip values for major pairs (approximate)
    PIP_VALUES = {
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

    # Pip sizes
    PIP_SIZES = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "AUDUSD": 0.0001,
        "USDJPY": 0.01,
        "EURJPY": 0.01,
        "GBPJPY": 0.01,
        "XAUUSD": 0.10,  # Gold
    }

    def __init__(self):
        self.results = {}

    def calculate(
        self,
        sltp_results: dict,
        sr_results: dict,
        structure_results: dict,
        image_height: int,
        account_balance: float = 10000,
        risk_percent: float = 1.0,
        pair: str = "EURUSD",
    ) -> dict:
        """
        Calculate full risk management profile."""
        best = sltp_results.get("best_scenario", {})
        scenarios = sltp_results.get("scenarios", [])

        if not best and not scenarios:
            return {"error": "No SL/TP data available for risk calculation"}

        valid_scenarios = [
            s
            for s in scenarios
            if s.get("sl") is not None
            and s.get("entry") is not None
            and abs(s.get("entry", 0) - s.get("sl", 0)) > 0.001
        ]
        if best and (
            best.get("sl") is None
            or best.get("entry") is None
            or abs(best.get("entry", 0) - best.get("sl", 0)) < 0.001
        ):
            best = valid_scenarios[0] if valid_scenarios else None

        # Get pip value for this pair
        pip_value = self.PIP_VALUES.get(pair, 10.0)
        pip_size = self.PIP_SIZES.get(pair, 0.0001)

        # Calculate for each scenario
        scenario_risks = []
        for scenario in scenarios:
            risk_data = self._calc_scenario_risk(
                scenario,
                account_balance,
                risk_percent,
                pip_value,
                pip_size,
                image_height,
                pair,
            )
            scenario_risks.append(risk_data)

        # Best scenario detailed calculation
        best_risk = (
            self._calc_scenario_risk(
                best,
                account_balance,
                risk_percent,
                pip_value,
                pip_size,
                image_height,
                pair,
            )
            if best
            else None
        )

        # Risk of ruin calculation
        win_rate = self._estimate_win_rate(sltp_results, structure_results)
        risk_of_ruin = self._calc_risk_of_ruin(risk_percent, win_rate)

        # Max drawdown estimate
        max_dd = self._calc_max_drawdown(risk_percent, win_rate)

        # Portfolio heat (total exposure across scenarios)
        portfolio_heat = risk_percent  # Single trade, so = risk %

        self.results = {
            "account_balance": account_balance,
            "risk_percent": risk_percent,
            "risk_amount": account_balance * risk_percent / 100,
            "pair": pair,
            "pip_value": pip_value,
            "pip_size": pip_size,
            "best_scenario_risk": best_risk,
            "all_scenario_risks": scenario_risks,
            "estimated_win_rate": round(win_rate, 3),
            "risk_of_ruin": risk_of_ruin,
            "max_drawdown_estimate": max_dd,
            "portfolio_heat": portfolio_heat,
            "position_sizing_table": self._generate_sizing_table(
                account_balance, pip_value, pip_size, image_height
            ),
            "risk_rules": self._get_risk_rules(),
        }

        return self.results

    def _calc_scenario_risk(
        self,
        scenario: dict,
        balance: float,
        risk_pct: float,
        pip_val: float,
        pip_size: float,
        img_h: int,
        pair: str,
    ) -> dict:
        """Calculate risk metrics for a single scenario."""
        if not scenario:
            return {}

        entry = scenario.get("entry", 0)
        sl = scenario.get("sl", 0)
        tp = scenario.get("tp", 0)
        rr = scenario.get("risk_reward", 0)

        # Calculate SL distance in pips
        sl_distance_price = abs(entry - sl)
        sl_distance_pips = sl_distance_price / pip_size if pip_size > 0 else 0

        # Calculate TP distance in pips
        tp_distance_price = abs(tp - entry)
        tp_distance_pips = tp_distance_price / pip_size if pip_size > 0 else 0

        # Position sizing
        risk_amount = balance * risk_pct / 100
        if sl_distance_pips > 0 and pip_val > 0:
            lot_size = risk_amount / (sl_distance_pips * pip_val)
        else:
            lot_size = 0

        # Cap at reasonable lot sizes
        lot_size = min(lot_size, 100.0)  # Max 100 lots

        # Round to standard lot increments
        if lot_size >= 1.0:
            lots = round(lot_size * 10) / 10  # 0.1 lot precision
        elif lot_size >= 0.1:
            lots = round(lot_size * 100) / 100  # 0.01 lot precision
        else:
            lots = round(lot_size * 1000) / 1000  # 0.001 lot precision (micro)

        # Actual risk with this lot size
        actual_risk = lots * sl_distance_pips * pip_val

        # Profit potential
        profit_potential = lots * tp_distance_pips * pip_val

        # Risk as % of account
        actual_risk_pct = (actual_risk / balance * 100) if balance > 0 else 0

        return {
            "scenario_name": scenario.get("name", "Unknown"),
            "direction": scenario.get("direction", "BUY"),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk_reward": rr,
            "sl_distance_pips": round(sl_distance_pips, 1),
            "tp_distance_pips": round(tp_distance_pips, 1),
            "recommended_lots": lots,
            "risk_amount_usd": round(actual_risk, 2),
            "profit_potential_usd": round(profit_potential, 2),
            "actual_risk_pct": round(actual_risk_pct, 2),
            "break_even_price": entry,
            "risk_per_pip": round(lots * pip_val, 2),
        }

    def _estimate_win_rate(self, sltp: dict, structure: dict) -> float:
        """Estimate expected win rate based on signal quality."""
        # Base win rate
        base_win_rate = 0.45  # Random entry baseline

        # Adjust based on confluence
        bias = sltp.get("bias", "NEUTRAL")
        if bias != "NEUTRAL":
            base_win_rate += 0.05

        # Adjust based on trend alignment
        trend = structure.get("trend_direction", "RANGING")
        if trend != "RANGING":
            base_win_rate += 0.05

        # Adjust based on structure strength
        strength = structure.get("trend_strength", 0.5)
        base_win_rate += strength * 0.05

        return min(base_win_rate, 0.70)  # Cap at 70%

    def _calc_risk_of_ruin(self, risk_pct: float, win_rate: float) -> dict:
        """
        Calculate risk of ruin (probability of losing entire account).
        Simplified model based on Kelly Criterion.
        """
        avg_win = 2.0  # Assume 1:2 R:R
        avg_loss = 1.0

        # Kelly percentage
        kelly = win_rate - (1 - win_rate) / avg_win

        # Risk of ruin approximation
        if risk_pct <= 0:
            return {"ror": 0, "level": "NONE", "advice": ""}

        # Simplified ROR formula
        q = 1 - win_rate
        ror = (
            (q / win_rate) ** (1 / (risk_pct / 100 * avg_win / avg_loss))
            if win_rate > q
            else 0.9
        )

        ror = max(0, min(ror, 1.0))

        if ror < 0.01:
            level = "VERY_LOW"
            advice = "Excellent risk management. Keep it up!"
        elif ror < 0.05:
            level = "LOW"
            advice = "Good risk control. Continue being disciplined."
        elif ror < 0.15:
            level = "MODERATE"
            advice = "Consider reducing position sizes slightly."
        elif ror < 0.30:
            level = "HIGH"
            advice = "⚠️ Your risk of ruin is elevated. Reduce position sizes to 0.5% or less."
        else:
            level = "VERY_HIGH"
            advice = "🚨 DANGER: High risk of account blow-up. Stop trading and reassess your strategy."

        return {
            "ror": round(ror, 4),
            "level": level,
            "advice": advice,
            "kelly_pct": round(kelly * 100, 2),
            "recommended_max_risk": round(max(kelly * 50, 0.25), 2),  # Half-Kelly
        }

    def _calc_max_drawdown(self, risk_pct: float, win_rate: float) -> dict:
        """Estimate maximum drawdown based on risk per trade and win rate."""
        # Expected max drawdown (simplified)
        # Using formula: E[MaxDD] ≈ risk_pct * (1 / (1 - 2*p + 2*p^2)) for series of trades
        if win_rate > 0 and win_rate < 1:
            # Expected consecutive losses
            avg_consecutive_losses = int(1 / (1 - win_rate + 0.01))
            max_consecutive_losses = min(avg_consecutive_losses * 3, 20)

            expected_dd = max_consecutive_losses * risk_pct
            worst_case_dd = min(expected_dd * 1.5, 100)
        else:
            expected_dd = risk_pct * 5
            worst_case_dd = risk_pct * 15

        return {
            "expected_max_dd_pct": round(min(expected_dd, 100), 1),
            "worst_case_dd_pct": round(min(worst_case_dd, 100), 1),
            "avg_consecutive_losses": (
                min(int(1 / (1 - win_rate + 0.01)), 10) if win_rate < 0.99 else 1
            ),
            "recovery_needed": (
                round(1 / (1 - expected_dd / 100) - 1, 2) * 100
                if expected_dd < 100
                else 999
            ),
        }

    def _generate_sizing_table(
        self, balance: float, pip_val: float, pip_size: float, img_h: int
    ) -> list:
        """Generate a position sizing reference table."""
        table = []
        for risk_pct in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            risk_amount = balance * risk_pct / 100
            for sl_pips in [20, 30, 50, 75, 100, 150, 200]:
                if sl_pips * pip_val > 0:
                    lots = risk_amount / (sl_pips * pip_val)
                    lots = round(lots * 100) / 100
                    table.append(
                        {
                            "risk_pct": risk_pct,
                            "risk_amount": round(risk_amount, 2),
                            "sl_pips": sl_pips,
                            "lots": lots,
                        }
                    )
        return table

    def _get_risk_rules(self) -> list:
        """Professional risk management rules."""
        return [
            "📏 Never risk more than 2% of your account on a single trade",
            "🔗 Never have more than 6% total exposure across all open trades",
            "📊 Always have a minimum 1:2 risk-to-reward ratio before entering",
            "🛑 Never move your stop loss against your position",
            "📉 After 3 consecutive losses, reduce position size by 50%",
            "📈 Increase position size only after consistent wins (not after a big win)",
            "⏰ Avoid trading during high-impact news events unless you're a news trader",
            "🎯 If you can't see a clear setup, DON'T TRADE — cash is a position",
            "🔄 Risk the same % on every trade — don't vary based on 'confidence'",
            "📋 Journal every trade: entry reason, SL, TP, outcome, lessons learned",
        ]
