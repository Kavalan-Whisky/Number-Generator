"""
Pattern Detection in Number Sequences.
Detects arithmetic, geometric, Fibonacci-like, polynomial, and periodic patterns.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PatternResult:
    """Result of a pattern detection analysis."""
    pattern_type: str
    detected: bool
    confidence: float  # 0 to 1
    parameters: Dict = field(default_factory=dict)
    description: str = ""

    def __str__(self) -> str:
        if self.detected:
            return f"{self.pattern_type}: DETECTED (confidence={self.confidence:.2f}) - {self.description}"
        return f"{self.pattern_type}: NOT DETECTED (confidence={self.confidence:.2f})"


@dataclass
class PatternReport:
    """Comprehensive pattern detection report."""
    patterns: List[PatternResult] = field(default_factory=list)

    def add(self, result: PatternResult) -> None:
        self.patterns.append(result)

    def dominant_pattern(self) -> Optional[PatternResult]:
        detected = [p for p in self.patterns if p.detected]
        if not detected:
            return None
        return max(detected, key=lambda p: p.confidence)

    def summary(self) -> str:
        lines = ["=== Pattern Detection Report ==="]
        for p in self.patterns:
            lines.append(str(p))
        dominant = self.dominant_pattern()
        if dominant:
            lines.append(f"\nDominant pattern: {dominant.pattern_type}")
        else:
            lines.append("\nNo clear pattern detected.")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "patterns": [
                {
                    "type": p.pattern_type,
                    "detected": p.detected,
                    "confidence": p.confidence,
                    "parameters": p.parameters,
                    "description": p.description,
                }
                for p in self.patterns
            ]
        }


def detect_arithmetic_pattern(data: List[float], tolerance: float = 1e-6) -> PatternResult:
    """
    Detect if sequence is arithmetic (constant difference).
    Returns confidence based on consistency of differences.
    """
    if len(data) < 3:
        return PatternResult("arithmetic", False, 0.0)

    diffs = [data[i + 1] - data[i] for i in range(len(data) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    max_dev = max(abs(d - mean_diff) for d in diffs)

    if max_dev <= tolerance:
        return PatternResult(
            "arithmetic",
            True,
            1.0,
            {"common_difference": mean_diff, "first_term": data[0]},
            f"Arithmetic sequence with d={mean_diff:.4g}",
        )

    # Partial confidence
    scale = max(abs(d) for d in diffs) or 1
    relative_dev = max_dev / scale
    confidence = max(0.0, 1.0 - relative_dev)

    return PatternResult(
        "arithmetic",
        confidence > 0.9,
        confidence,
        {"approx_difference": mean_diff, "max_deviation": max_dev},
        f"Approximate arithmetic with d≈{mean_diff:.4g}",
    )


def detect_geometric_pattern(data: List[float], tolerance: float = 1e-6) -> PatternResult:
    """
    Detect if sequence is geometric (constant ratio).
    """
    if len(data) < 3:
        return PatternResult("geometric", False, 0.0)

    # Check for zeros
    if any(x == 0 for x in data[:-1]):
        return PatternResult("geometric", False, 0.0, description="Contains zero values")

    ratios = [data[i + 1] / data[i] for i in range(len(data) - 1) if data[i] != 0]
    if not ratios:
        return PatternResult("geometric", False, 0.0)

    mean_ratio = sum(ratios) / len(ratios)
    max_dev = max(abs(r - mean_ratio) for r in ratios)

    if max_dev <= tolerance:
        return PatternResult(
            "geometric",
            True,
            1.0,
            {"common_ratio": mean_ratio, "first_term": data[0]},
            f"Geometric sequence with r={mean_ratio:.4g}",
        )

    scale = abs(mean_ratio) or 1
    relative_dev = max_dev / scale
    confidence = max(0.0, 1.0 - relative_dev)

    return PatternResult(
        "geometric",
        confidence > 0.9,
        confidence,
        {"approx_ratio": mean_ratio, "max_deviation": max_dev},
        f"Approximate geometric with r≈{mean_ratio:.4g}",
    )


def detect_fibonacci_like(data: List[float], tolerance: float = 1e-6) -> PatternResult:
    """
    Detect if sequence follows a Fibonacci-like recurrence a[n] = a[n-1] + a[n-2].
    """
    if len(data) < 4:
        return PatternResult("fibonacci_like", False, 0.0)

    errors = []
    for i in range(2, len(data)):
        expected = data[i - 1] + data[i - 2]
        if expected != 0:
            errors.append(abs(data[i] - expected) / max(abs(expected), 1))
        else:
            errors.append(abs(data[i]))

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)

    if max_error <= tolerance:
        return PatternResult(
            "fibonacci_like",
            True,
            1.0,
            {"initial": [data[0], data[1]]},
            f"Fibonacci-like with a[0]={data[0]}, a[1]={data[1]}",
        )

    confidence = max(0.0, 1.0 - mean_error)
    return PatternResult(
        "fibonacci_like",
        confidence > 0.95,
        confidence,
        {"mean_error": mean_error},
        f"Approximate Fibonacci-like (mean error={mean_error:.4g})",
    )


def detect_polynomial_pattern(data: List[float], max_degree: int = 4) -> PatternResult:
    """
    Detect if sequence follows a polynomial pattern using finite differences.
    A polynomial of degree d has (d+1)th differences equal to zero.
    """
    if len(data) < 3:
        return PatternResult("polynomial", False, 0.0)

    def finite_differences(seq: List[float]) -> List[float]:
        return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]

    diffs = list(data)
    for degree in range(1, min(max_degree + 1, len(data))):
        diffs = finite_differences(diffs)
        if not diffs:
            break
        max_val = max(abs(x) for x in diffs)
        mean_val = abs(sum(diffs) / len(diffs)) if diffs else 0

        # If differences are nearly constant
        if len(diffs) >= 1:
            diff_of_diffs_max = max(abs(diffs[i+1] - diffs[i]) for i in range(len(diffs)-1)) if len(diffs) > 1 else 0
            if diff_of_diffs_max < 1e-6 and degree <= max_degree:
                return PatternResult(
                    "polynomial",
                    True,
                    1.0 - degree / (max_degree + 1),
                    {"degree": degree, "leading_difference": diffs[0] if diffs else 0},
                    f"Polynomial of degree {degree}",
                )

    return PatternResult("polynomial", False, 0.0, description="No polynomial pattern detected")


def detect_periodicity(data: List[float]) -> PatternResult:
    """
    Detect periodic patterns using autocorrelation.
    Returns period estimate and confidence.
    """
    n = len(data)
    if n < 6:
        return PatternResult("periodic", False, 0.0)

    # Compute mean and variance
    mean_v = sum(data) / n
    var_v = sum((x - mean_v) ** 2 for x in data) / n
    if var_v < 1e-12:
        return PatternResult("periodic", False, 0.0, description="Constant sequence")

    # Autocorrelation at various lags
    max_lag = n // 2
    correlations = []
    for lag in range(1, max_lag):
        cov = sum((data[i] - mean_v) * (data[i + lag] - mean_v) for i in range(n - lag)) / (n - lag)
        corr = cov / var_v
        correlations.append((lag, corr))

    # Find peaks in autocorrelation
    best_lag = 1
    best_corr = 0.0
    for lag, corr in correlations:
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    confidence = max(0.0, best_corr)

    if best_corr > 0.8:
        return PatternResult(
            "periodic",
            True,
            confidence,
            {"period": best_lag, "correlation": best_corr},
            f"Periodic with period {best_lag} (correlation={best_corr:.3f})",
        )

    return PatternResult(
        "periodic",
        False,
        confidence,
        {"best_lag": best_lag, "best_correlation": best_corr},
        f"No clear periodicity (best correlation={best_corr:.3f} at lag={best_lag})",
    )


def detect_power_law(data: List[float]) -> PatternResult:
    """
    Detect if data follows a power law distribution.
    Uses log-log regression.
    """
    if len(data) < 5:
        return PatternResult("power_law", False, 0.0)

    positive = [x for x in data if x > 0]
    if len(positive) < 5:
        return PatternResult("power_law", False, 0.0)

    # Rank-frequency analysis
    sorted_data = sorted(positive, reverse=True)
    n = len(sorted_data)

    log_ranks = [math.log(i + 1) for i in range(n)]
    log_vals = [math.log(v) for v in sorted_data]

    # Linear regression on log-log
    n_pts = len(log_ranks)
    mean_x = sum(log_ranks) / n_pts
    mean_y = sum(log_vals) / n_pts

    ss_xx = sum((x - mean_x) ** 2 for x in log_ranks)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_vals))

    if ss_xx == 0:
        return PatternResult("power_law", False, 0.0)

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    # R-squared
    y_pred = [slope * x + intercept for x in log_ranks]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(log_vals, y_pred))
    ss_tot = sum((y - mean_y) ** 2 for y in log_vals)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return PatternResult(
        "power_law",
        r_squared > 0.9,
        r_squared,
        {"exponent": -slope, "intercept": intercept, "r_squared": r_squared},
        f"Power law with exponent={-slope:.3f} (R²={r_squared:.3f})",
    )


class PatternDetector:
    """Comprehensive pattern detector for number sequences."""

    def analyze(self, data: List[float]) -> PatternReport:
        """Run all pattern detectors and return a comprehensive report."""
        report = PatternReport()

        report.add(detect_arithmetic_pattern(data))
        report.add(detect_geometric_pattern(data))
        report.add(detect_fibonacci_like(data))
        report.add(detect_polynomial_pattern(data))
        report.add(detect_periodicity(data))
        report.add(detect_power_law(data))

        return report

    def detect_arithmetic(self, data: List[float], tolerance: float = 1e-6) -> PatternResult:
        return detect_arithmetic_pattern(data, tolerance)

    def detect_geometric(self, data: List[float], tolerance: float = 1e-6) -> PatternResult:
        return detect_geometric_pattern(data, tolerance)

    def detect_fibonacci_like(self, data: List[float]) -> PatternResult:
        return detect_fibonacci_like(data)

    def detect_polynomial(self, data: List[float], max_degree: int = 4) -> PatternResult:
        return detect_polynomial_pattern(data, max_degree)

    def detect_periodicity(self, data: List[float]) -> PatternResult:
        return detect_periodicity(data)

    def detect_power_law(self, data: List[float]) -> PatternResult:
        return detect_power_law(data)
