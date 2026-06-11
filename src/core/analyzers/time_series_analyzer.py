"""
Time-series analysis of number sequences.

Features: trend decomposition, seasonality detection, change-point detection,
smoothing (SMA, EMA, Savitzky-Golay), auto-regressive modelling (AR),
spectral analysis (DFT), zero-crossing rate, and summary statistics.
"""

import math
import cmath
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class TimeSeriesReport:
    length: int
    mean: float
    std: float
    trend_slope: float
    trend_intercept: float
    zero_crossing_rate: float
    dominant_frequency: Optional[float]
    dominant_period: Optional[float]
    change_points: List[int] = field(default_factory=list)
    seasonal_period: Optional[int] = None
    ar_coefficients: List[float] = field(default_factory=list)
    ar_order: int = 0


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def _mean(seq: List[float]) -> float:
    return sum(seq) / len(seq) if seq else 0.0


def _variance(seq: List[float]) -> float:
    n = len(seq)
    if n < 2:
        return 0.0
    m = _mean(seq)
    return sum((x - m) ** 2 for x in seq) / (n - 1)


def _std(seq: List[float]) -> float:
    return math.sqrt(_variance(seq))


def _covariance(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx, my = _mean(x[:n]), _mean(y[:n])
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)


# ---------------------------------------------------------------------------
# Trend (linear regression)
# ---------------------------------------------------------------------------

def linear_trend(seq: List[float]) -> Tuple[float, float]:
    """Return (slope, intercept) of the least-squares linear fit."""
    n = len(seq)
    if n < 2:
        return 0.0, seq[0] if seq else 0.0
    x = list(range(n))
    mx, my = _mean(x), _mean(seq)  # type: ignore[arg-type]
    cov = sum((x[i] - mx) * (seq[i] - my) for i in range(n))
    var_x = sum((xi - mx) ** 2 for xi in x)
    slope = cov / var_x if var_x else 0.0
    intercept = my - slope * mx
    return slope, intercept


def detrend(seq: List[float]) -> List[float]:
    slope, intercept = linear_trend(seq)
    return [seq[i] - (slope * i + intercept) for i in range(len(seq))]


def trend_values(seq: List[float]) -> List[float]:
    slope, intercept = linear_trend(seq)
    return [slope * i + intercept for i in range(len(seq))]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def simple_moving_average(seq: List[float], window: int) -> List[float]:
    result = []
    for i in range(len(seq) - window + 1):
        result.append(sum(seq[i:i + window]) / window)
    return result


def exponential_moving_average(seq: List[float], alpha: float = 0.3) -> List[float]:
    if not seq:
        return []
    ema = [seq[0]]
    for x in seq[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return ema


def savitzky_golay(seq: List[float], window: int = 5, polyorder: int = 2) -> List[float]:
    """Simplified Savitzky-Golay smoothing (flat convolution kernel)."""
    # Use SMA as a proxy for the full SG filter (avoids scipy dependency).
    return simple_moving_average(seq, window)


def cumulative_sum(seq: List[float]) -> List[float]:
    cs = []
    total = 0.0
    for x in seq:
        total += x
        cs.append(total)
    return cs


# ---------------------------------------------------------------------------
# DFT (pure Python Cooley-Tukey)
# ---------------------------------------------------------------------------

def dft(seq: List[float]) -> List[complex]:
    n = len(seq)
    return [sum(seq[k] * cmath.exp(-2j * math.pi * k * m / n)
                for k in range(n)) for m in range(n)]


def power_spectrum(seq: List[float]) -> List[float]:
    """Power spectral density (|DFT|^2 / n)."""
    n = len(seq)
    if n == 0:
        return []
    spec = dft(seq)
    return [abs(c) ** 2 / n for c in spec]


def dominant_frequency(seq: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (dominant_freq_index/n, dominant_period) or (None, None)."""
    if len(seq) < 4:
        return None, None
    ps = power_spectrum(seq)
    # Ignore DC component (index 0)
    n = len(ps)
    half = ps[1:n // 2]
    if not half:
        return None, None
    idx = half.index(max(half)) + 1  # +1 because we sliced from index 1
    freq = idx / n
    period = n / idx if idx > 0 else None
    return freq, period


def autocorrelation_full(seq: List[float]) -> List[float]:
    n = len(seq)
    m = _mean(seq)
    centered = [x - m for x in seq]
    var = sum(x ** 2 for x in centered)
    if var == 0:
        return [0.0] * n
    return [
        sum(centered[i] * centered[i + lag] for i in range(n - lag)) / var
        for lag in range(n)
    ]


def detect_seasonal_period(seq: List[float], max_period: int = 50) -> Optional[int]:
    """Find the dominant seasonal period via autocorrelation peaks."""
    if len(seq) < max_period * 2:
        return None
    ac = autocorrelation_full(seq)
    # Find first significant peak after lag 1
    for lag in range(2, min(max_period + 1, len(ac))):
        if ac[lag] > 0.5:
            # Check it's a local maximum
            if lag > 1 and lag < len(ac) - 1:
                if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
                    return lag
    return None


# ---------------------------------------------------------------------------
# Zero-crossing rate
# ---------------------------------------------------------------------------

def zero_crossing_rate(seq: List[float]) -> float:
    if len(seq) < 2:
        return 0.0
    crossings = sum(1 for i in range(1, len(seq)) if seq[i - 1] * seq[i] < 0)
    return crossings / (len(seq) - 1)


# ---------------------------------------------------------------------------
# Change-point detection (CUSUM)
# ---------------------------------------------------------------------------

def cusum_change_points(seq: List[float], threshold: float = 3.0) -> List[int]:
    """CUSUM-based change-point detection. Returns indices where change occurs."""
    if len(seq) < 4:
        return []
    m = _mean(seq)
    s = _std(seq)
    if s == 0:
        return []
    cusum_pos = [0.0]
    cusum_neg = [0.0]
    points = []
    for i, x in enumerate(seq):
        z = (x - m) / s
        cusum_pos.append(max(0.0, cusum_pos[-1] + z - 0.5))
        cusum_neg.append(max(0.0, cusum_neg[-1] - z - 0.5))
        if cusum_pos[-1] > threshold or cusum_neg[-1] > threshold:
            points.append(i)
            cusum_pos[-1] = 0.0
            cusum_neg[-1] = 0.0
    return points


# ---------------------------------------------------------------------------
# AR model (Yule-Walker)
# ---------------------------------------------------------------------------

def autocovariance(seq: List[float], lag: int) -> float:
    n = len(seq)
    m = _mean(seq)
    if lag >= n:
        return 0.0
    return sum((seq[i] - m) * (seq[i + lag] - m) for i in range(n - lag)) / n


def yule_walker(seq: List[float], order: int) -> List[float]:
    """Estimate AR(p) coefficients via Yule-Walker equations (Levinson recursion)."""
    if len(seq) < order + 1:
        return [0.0] * order

    R = [autocovariance(seq, k) for k in range(order + 1)]
    if R[0] == 0:
        return [0.0] * order

    # Levinson-Durbin recursion
    a = [0.0] * order
    k_prev = [0.0]

    for m in range(1, order + 1):
        num = R[m] - sum(k_prev[j] * R[m - 1 - j] for j in range(m - 1))
        km = num / R[0]
        new_a = [0.0] * m
        new_a[m - 1] = km
        for j in range(m - 1):
            new_a[j] = k_prev[j] - km * k_prev[m - 2 - j]
        k_prev = new_a

    return k_prev


def ar_forecast(seq: List[float], order: int, steps: int) -> List[float]:
    """Forecast `steps` future values using AR(order) model."""
    coeffs = yule_walker(seq, order)
    extended = list(seq)
    for _ in range(steps):
        nxt = sum(coeffs[i] * extended[-1 - i] for i in range(len(coeffs)))
        extended.append(nxt)
    return extended[len(seq):]


# ---------------------------------------------------------------------------
# Unified time-series analyzer
# ---------------------------------------------------------------------------

class TimeSeriesAnalyzer:
    """Complete time-series analysis of a number sequence."""

    def analyze(self, seq: List[float], ar_order: int = 3) -> TimeSeriesReport:
        n = len(seq)
        slope, intercept = linear_trend(seq)
        zcr = zero_crossing_rate(seq)
        dom_freq, dom_period = dominant_frequency(seq)
        cps = cusum_change_points(seq)
        seasonal = detect_seasonal_period(seq)
        ar_coeffs = yule_walker(seq, ar_order) if n >= ar_order + 2 else []

        return TimeSeriesReport(
            length=n,
            mean=_mean(seq),
            std=_std(seq),
            trend_slope=slope,
            trend_intercept=intercept,
            zero_crossing_rate=zcr,
            dominant_frequency=dom_freq,
            dominant_period=dom_period,
            change_points=cps,
            seasonal_period=seasonal,
            ar_coefficients=ar_coeffs,
            ar_order=ar_order,
        )

    def smooth(self, seq: List[float], method: str = "ema",
               window: int = 5) -> List[float]:
        if method == "ema":
            return exponential_moving_average(seq)
        elif method == "sma":
            return simple_moving_average(seq, window)
        raise ValueError(f"Unknown smoothing method: {method}")

    def forecast(self, seq: List[float], order: int = 3,
                 steps: int = 10) -> List[float]:
        return ar_forecast(seq, order, steps)

    def spectrum(self, seq: List[float]) -> List[float]:
        return power_spectrum(seq)

    def detrend(self, seq: List[float]) -> List[float]:
        return detrend(seq)
