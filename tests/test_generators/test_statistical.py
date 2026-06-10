"""Tests for statistical distribution generators."""

import math
import pytest
from src.core.generators.statistical_generator import (
    NormalDistribution, PoissonDistribution, ExponentialDistribution,
    GammaDistribution, BetaDistribution, UniformDistribution,
    BinomialDistribution, ChiSquaredDistribution,
    StatisticalGeneratorFactory,
)


def mean(data):
    return sum(data) / len(data)

def std(data):
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))


class TestNormalDistribution:
    def test_sample_type(self):
        dist = NormalDistribution(seed=42)
        s = dist.sample()
        assert isinstance(s, float)

    def test_generate_count(self):
        dist = NormalDistribution(seed=42)
        samples = dist.generate(100)
        assert len(samples) == 100

    def test_mean_approx(self):
        dist = NormalDistribution(mu=5.0, sigma=1.0, seed=42)
        samples = dist.generate(1000)
        assert abs(mean(samples) - 5.0) < 0.5

    def test_std_approx(self):
        dist = NormalDistribution(mu=0.0, sigma=2.0, seed=42)
        samples = dist.generate(1000)
        assert abs(std(samples) - 2.0) < 0.5

    def test_pdf(self):
        dist = NormalDistribution()
        assert dist.pdf(0) == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-5)

    def test_cdf_at_mean(self):
        dist = NormalDistribution(mu=5, sigma=1)
        assert dist.cdf(5) == pytest.approx(0.5, abs=1e-5)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            NormalDistribution(sigma=-1)


class TestPoissonDistribution:
    def test_non_negative_integers(self):
        dist = PoissonDistribution(lam=3.0, seed=42)
        samples = dist.generate(100)
        assert all(isinstance(s, int) and s >= 0 for s in samples)

    def test_mean_approx(self):
        dist = PoissonDistribution(lam=5.0, seed=42)
        samples = dist.generate(2000)
        assert abs(mean(samples) - 5.0) < 0.5

    def test_pmf(self):
        dist = PoissonDistribution(lam=1.0)
        assert dist.pmf(0) == pytest.approx(math.exp(-1), rel=1e-5)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            PoissonDistribution(lam=0)


class TestExponentialDistribution:
    def test_positive_samples(self):
        dist = ExponentialDistribution(rate=2.0, seed=42)
        samples = dist.generate(100)
        assert all(s > 0 for s in samples)

    def test_mean_approx(self):
        dist = ExponentialDistribution(rate=1.0, seed=42)
        samples = dist.generate(2000)
        assert abs(mean(samples) - 1.0) < 0.2

    def test_cdf(self):
        dist = ExponentialDistribution(rate=1.0)
        assert dist.cdf(0) == 0
        assert abs(dist.cdf(1) - (1 - math.exp(-1))) < 1e-9

    def test_pdf_at_zero(self):
        dist = ExponentialDistribution(rate=2.0)
        assert dist.pdf(0) == 2.0


class TestGammaDistribution:
    def test_positive(self):
        dist = GammaDistribution(shape=2.0, rate=1.0, seed=42)
        samples = dist.generate(100)
        assert all(s > 0 for s in samples)

    def test_mean_approx(self):
        dist = GammaDistribution(shape=3.0, rate=1.0, seed=42)
        samples = dist.generate(2000)
        assert abs(mean(samples) - 3.0) < 0.5

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            GammaDistribution(shape=-1)


class TestBetaDistribution:
    def test_unit_interval(self):
        dist = BetaDistribution(alpha=2, beta=2, seed=42)
        samples = dist.generate(200)
        assert all(0 <= s <= 1 for s in samples)

    def test_mean(self):
        dist = BetaDistribution(alpha=2, beta=3)
        assert dist.mean() == pytest.approx(0.4, rel=1e-5)

    def test_variance(self):
        dist = BetaDistribution(alpha=2, beta=3)
        assert dist.variance() == pytest.approx(2*3 / (25 * 6), rel=1e-3)


class TestUniformDistribution:
    def test_range(self):
        dist = UniformDistribution(low=5, high=10, seed=42)
        samples = dist.generate(500)
        assert all(5 <= s <= 10 for s in samples)

    def test_mean(self):
        dist = UniformDistribution(low=0, high=10, seed=42)
        samples = dist.generate(2000)
        assert abs(mean(samples) - 5.0) < 0.5

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            UniformDistribution(low=10, high=5)


class TestBinomialDistribution:
    def test_range(self):
        dist = BinomialDistribution(n=10, p=0.5, seed=42)
        samples = dist.generate(200)
        assert all(0 <= s <= 10 for s in samples)

    def test_mean_approx(self):
        dist = BinomialDistribution(n=10, p=0.5, seed=42)
        samples = dist.generate(2000)
        assert abs(mean(samples) - 5.0) < 0.5

    def test_pmf_sum(self):
        dist = BinomialDistribution(n=5, p=0.5)
        total = sum(dist.pmf(k) for k in range(6))
        assert abs(total - 1.0) < 1e-9


class TestStatisticalGeneratorFactory:
    def test_create_normal(self):
        gen = StatisticalGeneratorFactory.create("normal", mu=0, sigma=1, seed=42)
        samples = gen.generate(10)
        assert len(samples) == 10

    def test_create_all_distributions(self):
        params_map = {
            "normal": {"mu": 0, "sigma": 1},
            "exponential": {"rate": 1},
            "uniform": {"low": 0, "high": 1},
            "binomial": {"n": 5, "p": 0.5},
        }
        for dist_name, params in params_map.items():
            gen = StatisticalGeneratorFactory.create(dist_name, seed=42, **params)
            samples = gen.generate(10)
            assert len(samples) == 10

    def test_unknown_distribution(self):
        with pytest.raises(ValueError):
            StatisticalGeneratorFactory.create("unknown_dist")
