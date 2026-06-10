"""
Statistical Analyzer - Comprehensive statistical analysis of number sequences.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


def mean(data: List[float]) -> float:
    """Arithmetic mean."""
    if not data:
        raise ValueError("Empty sequence")
    return sum(data) / len(data)


def median(data: List[float]) -> float:
    """Median value."""
    if not data:
        raise ValueError("Empty sequence")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def mode(data: List[float]) -> List[float]:
    """Mode(s) - most frequent value(s)."""
    if not data:
        raise ValueError("Empty sequence")
    counts = Counter(data)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]


def variance(data: List[float], ddof: int = 0) -> float:
    """Variance with optional degrees-of-freedom correction."""
    n = len(data)
    if n <= ddof:
        raise ValueError(f"Need at least {ddof + 1} data points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (n - ddof)


def std_dev(data: List[float], ddof: int = 0) -> float:
    """Standard deviation."""
    return math.sqrt(variance(data, ddof))


def skewness(data: List[float]) -> float:
    """
    Sample skewness (Fisher's definition).
    Positive: right-skewed, Negative: left-skewed.
    """
    n = len(data)
    if n < 3:
        raise ValueError("Need at least 3 data points for skewness")
    m = mean(data)
    s = std_dev(data)
    if s == 0:
        return 0.0
    return sum(((x - m) / s) ** 3 for x in data) * n / ((n - 1) * (n - 2))


def kurtosis(data: List[float], excess: bool = True) -> float:
    """
    Sample kurtosis.
    excess=True: Fisher's excess kurtosis (normal = 0)
    excess=False: Pearson's kurtosis (normal = 3)
    """
    n = len(data)
    if n < 4:
        raise ValueError("Need at least 4 data points for kurtosis")
    m = mean(data)
    s = std_dev(data)
    if s == 0:
        return 0.0
    # Biased kurtosis
    k4 = sum(((x - m) / s) ** 4 for x in data) / n
    if excess:
        return k4 - 3.0
    return k4


def entropy(data: List[float], bins: int = 10) -> float:
    """Shannon entropy of the distribution."""
    if not data:
        return 0.0
    min_v, max_v = min(data), max(data)
    if min_v == max_v:
        return 0.0
    # Bin the data
    bin_width = (max_v - min_v) / bins
    counts = [0] * bins
    for x in data:
        idx = int((x - min_v) / bin_width)
        idx = min(idx, bins - 1)
        counts[idx] += 1
    total = len(data)
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile (0 <= p <= 100)."""
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(data)
    n = len(s)
    if n == 0:
        raise ValueError("Empty sequence")
    if n == 1:
        return float(s[0])
    k = (n - 1) * p / 100.0
    f = int(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return s[f] * (c - k) + s[c] * (k - f)


def iqr(data: List[float]) -> float:
    """Interquartile range (Q3 - Q1)."""
    return percentile(data, 75) - percentile(data, 25)


def z_scores(data: List[float]) -> List[float]:
    """Standardize data to z-scores."""
    m = mean(data)
    s = std_dev(data)
    if s == 0:
        return [0.0] * len(data)
    return [(x - m) / s for x in data]


def range_(data: List[float]) -> float:
    """Range of the data."""
    return max(data) - min(data)


def geometric_mean(data: List[float]) -> float:
    """Geometric mean (all values must be positive)."""
    if any(x <= 0 for x in data):
        raise ValueError("All values must be positive for geometric mean")
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))


def harmonic_mean(data: List[float]) -> float:
    """Harmonic mean."""
    if any(x == 0 for x in data):
        raise ValueError("No zero values allowed for harmonic mean")
    return len(data) / sum(1.0 / x for x in data)


def chi_square_test(observed: List[float], expected: Optional[List[float]] = None) -> Tuple[float, float]:
    """
    Chi-square goodness-of-fit test.
    Returns (chi2_statistic, p_value).
    If expected is None, assumes uniform distribution.
    """
    n = len(observed)
    if expected is None:
        total = sum(observed)
        expected = [total / n] * n
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have same length")

    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e != 0)
    dof = n - 1

    # Approximate p-value using regularized incomplete gamma function
    p_value = _chi2_p_value(chi2, dof)
    return chi2, p_value


def _chi2_p_value(chi2: float, dof: int) -> float:
    """Approximate p-value for chi-square test using scipy if available, else approximation."""
    try:
        from scipy.stats import chi2 as chi2_dist
        return float(1 - chi2_dist.cdf(chi2, dof))
    except ImportError:
        # Wilson-Hilferty approximation
        if dof <= 0:
            return 0.0
        z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        return 0.5 * math.erfc(z / math.sqrt(2))


def t_test_one_sample(data: List[float], mu0: float = 0.0) -> Tuple[float, float]:
    """
    One-sample t-test.
    Tests whether mean equals mu0.
    Returns (t_statistic, p_value).
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")
    m = mean(data)
    s = std_dev(data, ddof=1)
    if s == 0:
        return float("inf"), 0.0
    t_stat = (m - mu0) / (s / math.sqrt(n))
    dof = n - 1
    p_value = _t_p_value(abs(t_stat), dof)
    return t_stat, p_value


def _t_p_value(t: float, dof: int) -> float:
    """Approximate two-tailed p-value for t-distribution."""
    try:
        from scipy.stats import t as t_dist
        return float(2 * (1 - t_dist.cdf(abs(t), dof)))
    except ImportError:
        # Abramowitz and Stegun approximation
        x = dof / (dof + t * t)
        return _regularized_incomplete_beta(dof / 2, 0.5, x)


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Approximation of regularized incomplete beta function."""
    # Using continued fraction approximation
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    try:
        import math
        # Lentz continued fraction
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
        # Simple approximation
        return min(1.0, front * 2)
    except Exception:
        return 0.5


def f_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """
    F-test for equality of variances.
    Returns (F_statistic, p_value).
    """
    var1 = variance(data1, ddof=1)
    var2 = variance(data2, ddof=1)
    if var2 == 0:
        return float("inf"), 0.0
    f_stat = var1 / var2
    dof1 = len(data1) - 1
    dof2 = len(data2) - 1
    try:
        from scipy.stats import f as f_dist
        p_value = float(2 * min(f_dist.cdf(f_stat, dof1, dof2), 1 - f_dist.cdf(f_stat, dof1, dof2)))
    except ImportError:
        p_value = 0.5  # fallback
    return f_stat, p_value


def autocorrelation(data: List[float], lag: int = 1) -> float:
    """Autocorrelation at given lag."""
    n = len(data)
    if lag >= n:
        raise ValueError("lag must be less than data length")
    m = mean(data)
    var = variance(data)
    if var == 0:
        return 0.0
    cov = sum((data[i] - m) * (data[i + lag] - m) for i in range(n - lag)) / (n - lag)
    return cov / var


def cross_correlation(x: List[float], y: List[float], lag: int = 0) -> float:
    """Cross-correlation between two sequences at given lag."""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    if lag >= n:
        raise ValueError("lag must be less than data length")
    mx, my = mean(x), mean(y)
    sx = std_dev(x) or 1.0
    sy = std_dev(y) or 1.0
    if lag >= 0:
        cov = sum((x[i] - mx) * (y[i + lag] - my) for i in range(n - lag)) / (n - lag)
    else:
        lag = -lag
        cov = sum((x[i + lag] - mx) * (y[i] - my) for i in range(n - lag)) / (n - lag)
    return cov / (sx * sy)


def describe(data: List[float]) -> Dict:
    """Full descriptive statistics."""
    n = len(data)
    if n == 0:
        return {"count": 0}
    sorted_data = sorted(data)
    return {
        "count": n,
        "mean": mean(data),
        "median": median(data),
        "mode": mode(data),
        "std_dev": std_dev(data, ddof=1) if n > 1 else 0.0,
        "variance": variance(data, ddof=1) if n > 1 else 0.0,
        "min": sorted_data[0],
        "max": sorted_data[-1],
        "range": sorted_data[-1] - sorted_data[0],
        "q1": percentile(data, 25),
        "q3": percentile(data, 75),
        "iqr": iqr(data),
        "skewness": skewness(data) if n >= 3 else None,
        "kurtosis": kurtosis(data) if n >= 4 else None,
        "entropy": entropy(data),
    }


class StatisticalAnalyzer:
    """Class-based interface for statistical analysis."""

    def __init__(self, data: List[float]):
        if not data:
            raise ValueError("Data cannot be empty")
        self.data = list(data)

    def describe(self) -> Dict:
        return describe(self.data)

    def mean(self) -> float:
        return mean(self.data)

    def median(self) -> float:
        return median(self.data)

    def mode(self) -> List[float]:
        return mode(self.data)

    def variance(self, ddof: int = 1) -> float:
        return variance(self.data, ddof)

    def std_dev(self, ddof: int = 1) -> float:
        return std_dev(self.data, ddof)

    def skewness(self) -> float:
        return skewness(self.data)

    def kurtosis(self, excess: bool = True) -> float:
        return kurtosis(self.data, excess)

    def entropy(self) -> float:
        return entropy(self.data)

    def percentile(self, p: float) -> float:
        return percentile(self.data, p)

    def iqr(self) -> float:
        return iqr(self.data)

    def z_scores(self) -> List[float]:
        return z_scores(self.data)

    def chi_square_test(self, expected: Optional[List[float]] = None) -> Tuple[float, float]:
        return chi_square_test(self.data, expected)

    def t_test(self, mu0: float = 0.0) -> Tuple[float, float]:
        return t_test_one_sample(self.data, mu0)

    def autocorrelation(self, lag: int = 1) -> float:
        return autocorrelation(self.data, lag)

    def autocorrelation_function(self, max_lag: int = 20) -> List[float]:
        return [autocorrelation(self.data, lag) for lag in range(1, min(max_lag + 1, len(self.data)))]

    def outliers_iqr(self, factor: float = 1.5) -> List[float]:
        """Find outliers using IQR method."""
        q1 = percentile(self.data, 25)
        q3 = percentile(self.data, 75)
        iq = q3 - q1
        lower = q1 - factor * iq
        upper = q3 + factor * iq
        return [x for x in self.data if x < lower or x > upper]

    def outliers_zscore(self, threshold: float = 3.0) -> List[float]:
        """Find outliers using z-score method."""
        zs = z_scores(self.data)
        return [self.data[i] for i, z in enumerate(zs) if abs(z) > threshold]
