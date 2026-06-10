"""
Distribution Fitting and Goodness-of-Fit Tests.
Fits data to common distributions and tests fit quality.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FitResult:
    """Result of fitting data to a distribution."""
    distribution: str
    parameters: Dict
    ks_statistic: float
    ks_p_value: float
    aic: float
    bic: float
    is_good_fit: bool
    description: str = ""

    def __str__(self) -> str:
        status = "GOOD FIT" if self.is_good_fit else "POOR FIT"
        params = ", ".join(f"{k}={v:.4g}" for k, v in self.parameters.items())
        return f"{self.distribution}({params}): {status} KS={self.ks_statistic:.4f} p={self.ks_p_value:.4f}"


def _ks_test(data: List[float], cdf_func) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov one-sample test.
    Returns (statistic, p_value).
    """
    n = len(data)
    sorted_data = sorted(data)
    d_max = 0.0
    for i, x in enumerate(sorted_data):
        empirical = (i + 1) / n
        theoretical = cdf_func(x)
        d = max(abs(empirical - theoretical), abs(i / n - theoretical))
        d_max = max(d_max, d)

    # Approximate p-value
    sqrt_n = math.sqrt(n)
    lambda_val = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_max
    p_value = 2 * sum(
        (-1) ** (k + 1) * math.exp(-2 * k * k * lambda_val * lambda_val)
        for k in range(1, 100)
    )
    p_value = max(0.0, min(1.0, p_value))
    return d_max, p_value


def _normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def _exponential_cdf(x: float, rate: float) -> float:
    if x < 0:
        return 0.0
    return 1 - math.exp(-rate * x)


def _uniform_cdf(x: float, low: float, high: float) -> float:
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def _log_normal_cdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    return 0.5 * (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2))))


def _compute_aic(n: int, k: int, log_likelihood: float) -> float:
    """Akaike Information Criterion: AIC = 2k - 2*ln(L)"""
    return 2 * k - 2 * log_likelihood


def _compute_bic(n: int, k: int, log_likelihood: float) -> float:
    """Bayesian Information Criterion: BIC = k*ln(n) - 2*ln(L)"""
    return k * math.log(n) - 2 * log_likelihood


def fit_normal(data: List[float]) -> FitResult:
    """Fit normal distribution to data."""
    n = len(data)
    if n < 3:
        return FitResult("normal", {}, 1.0, 0.0, float("inf"), float("inf"), False)

    mu = sum(data) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in data) / (n - 1))
    if sigma == 0:
        sigma = 1.0

    cdf = lambda x: _normal_cdf(x, mu, sigma)
    ks_stat, ks_p = _ks_test(data, cdf)

    # Log-likelihood
    log_lik = sum(
        -0.5 * math.log(2 * math.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)
        for x in data
    )
    aic = _compute_aic(n, 2, log_lik)
    bic = _compute_bic(n, 2, log_lik)

    return FitResult(
        "normal",
        {"mu": mu, "sigma": sigma},
        ks_stat, ks_p, aic, bic,
        ks_p >= 0.05,
        f"Normal distribution N(mu={mu:.3f}, sigma={sigma:.3f})",
    )


def fit_exponential(data: List[float]) -> FitResult:
    """Fit exponential distribution to data."""
    n = len(data)
    if n < 3 or any(x <= 0 for x in data):
        return FitResult("exponential", {}, 1.0, 0.0, float("inf"), float("inf"), False,
                         "Data must be positive for exponential fit")

    rate = n / sum(data)  # MLE: 1/mean

    cdf = lambda x: _exponential_cdf(x, rate)
    ks_stat, ks_p = _ks_test(data, cdf)

    log_lik = sum(math.log(rate) - rate * x for x in data)
    aic = _compute_aic(n, 1, log_lik)
    bic = _compute_bic(n, 1, log_lik)

    return FitResult(
        "exponential",
        {"rate": rate, "mean": 1 / rate},
        ks_stat, ks_p, aic, bic,
        ks_p >= 0.05,
        f"Exponential distribution Exp(rate={rate:.3f})",
    )


def fit_uniform(data: List[float]) -> FitResult:
    """Fit uniform distribution to data."""
    n = len(data)
    if n < 2:
        return FitResult("uniform", {}, 1.0, 0.0, float("inf"), float("inf"), False)

    low = min(data)
    high = max(data)
    if low == high:
        return FitResult("uniform", {}, 1.0, 0.0, float("inf"), float("inf"), False)

    cdf = lambda x: _uniform_cdf(x, low, high)
    ks_stat, ks_p = _ks_test(data, cdf)

    span = high - low
    log_lik = n * (-math.log(span))
    aic = _compute_aic(n, 2, log_lik)
    bic = _compute_bic(n, 2, log_lik)

    return FitResult(
        "uniform",
        {"low": low, "high": high},
        ks_stat, ks_p, aic, bic,
        ks_p >= 0.05,
        f"Uniform distribution U({low:.3f}, {high:.3f})",
    )


def fit_log_normal(data: List[float]) -> FitResult:
    """Fit log-normal distribution to data."""
    n = len(data)
    if n < 3 or any(x <= 0 for x in data):
        return FitResult("log_normal", {}, 1.0, 0.0, float("inf"), float("inf"), False,
                         "Data must be positive for log-normal fit")

    log_data = [math.log(x) for x in data]
    mu = sum(log_data) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in log_data) / (n - 1))
    if sigma == 0:
        sigma = 1.0

    cdf = lambda x: _log_normal_cdf(x, mu, sigma)
    ks_stat, ks_p = _ks_test(data, cdf)

    log_lik = sum(
        -math.log(x * sigma * math.sqrt(2 * math.pi)) - (math.log(x) - mu)**2 / (2 * sigma**2)
        for x in data
    )
    aic = _compute_aic(n, 2, log_lik)
    bic = _compute_bic(n, 2, log_lik)

    return FitResult(
        "log_normal",
        {"mu": mu, "sigma": sigma},
        ks_stat, ks_p, aic, bic,
        ks_p >= 0.05,
        f"Log-Normal distribution LN(mu={mu:.3f}, sigma={sigma:.3f})",
    )


def anderson_darling_test(data: List[float], distribution: str = "normal") -> Tuple[float, float]:
    """
    Anderson-Darling test for goodness of fit.
    Returns (A_statistic, significance_level_passed).
    """
    n = len(data)
    if n < 4:
        return 0.0, 0.0

    sorted_data = sorted(data)

    if distribution == "normal":
        mu = sum(data) / n
        sigma = math.sqrt(sum((x - mu) ** 2 for x in data) / (n - 1)) or 1.0
        cdf_vals = [_normal_cdf(x, mu, sigma) for x in sorted_data]
    elif distribution == "exponential":
        if any(x <= 0 for x in data):
            return 0.0, 0.0
        rate = n / sum(data)
        cdf_vals = [_exponential_cdf(x, rate) for x in sorted_data]
    elif distribution == "uniform":
        low, high = min(data), max(data)
        if low == high:
            return 0.0, 0.0
        cdf_vals = [_uniform_cdf(x, low, high) for x in sorted_data]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Clamp to avoid log(0)
    eps = 1e-10
    cdf_vals = [max(eps, min(1 - eps, c)) for c in cdf_vals]

    A_sq = -n - sum(
        (2 * i + 1) * (math.log(cdf_vals[i]) + math.log(1 - cdf_vals[n - 1 - i]))
        for i in range(n)
    ) / n

    # Critical values for normal distribution
    # p_value approximation
    if A_sq < 0.2:
        p = 1 - math.exp(-13.436 + 101.14 * A_sq - 223.73 * A_sq**2)
    elif A_sq < 0.34:
        p = 1 - math.exp(-8.318 + 42.796 * A_sq - 59.938 * A_sq**2)
    elif A_sq < 0.6:
        p = math.exp(0.9177 - 4.279 * A_sq - 1.38 * A_sq**2)
    else:
        p = math.exp(1.2937 - 5.709 * A_sq + 0.0186 * A_sq**2)

    return A_sq, max(0.0, min(1.0, p))


class DistributionFitter:
    """Fits data to multiple distributions and ranks them by goodness-of-fit."""

    def fit_all(self, data: List[float]) -> List[FitResult]:
        """Fit all available distributions and return sorted results."""
        results = []

        # Normal
        results.append(fit_normal(data))

        # Exponential (only for positive data)
        if all(x > 0 for x in data):
            results.append(fit_exponential(data))
            results.append(fit_log_normal(data))

        # Uniform
        results.append(fit_uniform(data))

        # Sort by KS p-value (higher is better fit)
        results.sort(key=lambda r: r.ks_p_value, reverse=True)
        return results

    def best_fit(self, data: List[float]) -> FitResult:
        """Return the best-fitting distribution."""
        results = self.fit_all(data)
        return results[0]

    def fit(self, data: List[float], distribution: str) -> FitResult:
        """Fit a specific distribution."""
        dist = distribution.lower()
        if dist == "normal":
            return fit_normal(data)
        elif dist == "exponential":
            return fit_exponential(data)
        elif dist == "uniform":
            return fit_uniform(data)
        elif dist in ("log_normal", "lognormal"):
            return fit_log_normal(data)
        raise ValueError(f"Unknown distribution: {distribution}")

    def anderson_darling(self, data: List[float], distribution: str = "normal") -> Tuple[float, float]:
        """Run Anderson-Darling test."""
        return anderson_darling_test(data, distribution)
