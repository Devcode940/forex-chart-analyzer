## 2025-05-18 - Vectorize Divergence Detector Momentum Oscillators
**Learning:** In momentum/oscillator calculations like RSI or Rate of Change, replacing explicit Python loops with NumPy vector slice operations and `scipy.signal.lfilter` with initial conditions (`zi`) achieves a ~15x speedup (~1.2s down to ~0.08s for 500 executions) without numerical drift.
**Action:** Use `scipy.signal.lfilter` with initialized filter state `zi` when vectorizing recursive exponential moving averages across sequence series in technical indicator calculations.
