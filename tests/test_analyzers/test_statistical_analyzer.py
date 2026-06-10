"""Tests for the statistical analyzer module."""

import math
import pytest
from src.core.analyzers.statistical_analyzer import (
    mean, median, mode, variance, std_dev, skewness, kurtosis,
    entropy, percentile, iqr, z_scores, chi_square_test,
    t_test_one_sample, autocorrelation, cross_correlation,
    describe, StatisticalAnalyzer, geometric_mean, harmonic_mean
)

SAMPLE_DATA = [2, 4, 4, 4, 5, 5, 7, 9]  # Classic textbook example


class TestMean:
    def test_basic(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_single(self):
        assert mean([7]) == 7.0

    def test_floats(self):
        assert abs(mean([1.5, 2.5, 3.5]) - 2.5) < 1e-10

    def test_negative(self):
        assert mean([-1, 0, 1]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean([])


class TestMedian:
    def test_odd_length(self):
        assert median([1, 2, 3]) == 2.0

    def test_even_length(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_unsorted_input(self):
        assert median([3, 1, 2]) == 2.0

    def test_single(self):
        assert median([5]) == 5.0


class TestMode:
    def test_single_mode(self):
        assert mode([1, 2, 2, 3]) == [2]

    def test_multiple_modes(self):
        modes = mode([1, 2, 1, 2, 3])
        assert set(modes) == {1, 2}

    def test_all_different(self):
        modes = mode([1, 2, 3])
        assert len(modes) == 3


class TestVarianceStd:
    def test_variance(self):
        assert variance(SAMPLE_DATA) == pytest.approx(4.0, abs=0.01)

    def test_std_dev(self):
        assert std_dev(SAMPLE_DATA) == pytest.approx(2.0, abs=0.01)

    def test_variance_ddof1(self):
        v = variance([2, 4, 6], ddof=1)
        assert v == pytest.approx(4.0, abs=0.01)

    def test_constant_data(self):
        assert variance([5, 5, 5]) == 0.0


class TestSkewnessKurtosis:
    def test_symmetric_low_skew(self):
        # Normal-ish data should have near-zero skewness
        data = list(range(1, 100))
        s = skewness(data)
        assert abs(s) < 0.5

    def test_positive_skew(self):
        # Right-skewed: more small values
        data = [1, 1, 1, 2, 2, 3, 4, 10, 20]
        s = skewness(data)
        assert s > 0

    def test_kurtosis_normal(self):
        # For uniform data, excess kurtosis should be negative
        data = list(range(1, 1001))
        k = kurtosis(data)
        assert k < 0  # Uniform is platykurtic


class TestEntropy:
    def test_uniform_high_entropy(self):
        data = list(range(1, 101))
        ent = entropy(data)
        assert ent > 3.0  # Should be near log2(10) ≈ 3.32

    def test_constant_zero_entropy(self):
        data = [5] * 100
        ent = entropy(data)
        assert ent == 0.0


class TestPercentileIQR:
    def test_p50(self):
        data = list(range(1, 101))
        p50 = percentile(data, 50)
        assert 49 <= p50 <= 51

    def test_p0_p100(self):
        data = [1, 2, 3, 4, 5]
        assert percentile(data, 0) == 1
        assert percentile(data, 100) == 5

    def test_iqr(self):
        data = list(range(1, 101))
        r = iqr(data)
        assert 40 <= r <= 60


class TestZScores:
    def test_mean_zero(self):
        data = [1, 2, 3, 4, 5]
        zs = z_scores(data)
        assert abs(sum(zs) / len(zs)) < 1e-10

    def test_std_approx_one(self):
        data = [1, 2, 3, 4, 5]
        zs = z_scores(data)
        std_z = math.sqrt(sum(z**2 for z in zs) / len(zs))
        assert abs(std_z - 1.0) < 0.01


class TestChiSquare:
    def test_uniform_distribution(self):
        # Uniform data should not reject
        counts = [100] * 10
        chi2, p = chi_square_test(counts)
        assert chi2 == 0.0
        assert p == pytest.approx(1.0, abs=0.1)

    def test_non_uniform(self):
        counts = [10, 10, 10, 10, 1000]
        chi2, p = chi_square_test(counts)
        assert chi2 > 10  # Large statistic for very unequal distribution


class TestTTest:
    def test_zero_mean(self):
        data = list(range(-50, 51))  # Mean = 0
        t, p = t_test_one_sample(data, 0)
        assert p > 0.5  # Should not reject H0

    def test_reject_wrong_mean(self):
        data = [10] * 100  # All 10s
        t, p = t_test_one_sample(data, 0)
        assert p < 0.01  # Should reject H0: mean=0


class TestAutocorrelation:
    def test_random_near_zero(self):
        import random
        random.seed(42)
        data = [random.random() for _ in range(1000)]
        ac = autocorrelation(data, 1)
        assert abs(ac) < 0.1

    def test_periodic_high(self):
        data = [math.sin(2 * math.pi * i / 10) for i in range(100)]
        ac = autocorrelation(data, 10)
        assert ac > 0.9


class TestDescribe:
    def test_keys(self):
        result = describe([1, 2, 3, 4, 5])
        assert "count" in result
        assert "mean" in result
        assert "std_dev" in result
        assert "min" in result
        assert "max" in result

    def test_values_correct(self):
        result = describe([1, 2, 3, 4, 5])
        assert result["count"] == 5
        assert result["mean"] == 3.0


class TestGeometricHarmonicMean:
    def test_geometric_mean(self):
        result = geometric_mean([1, 2, 4])
        assert abs(result - 2.0) < 0.01

    def test_harmonic_mean(self):
        result = harmonic_mean([1, 2, 4])
        expected = 3 / (1 + 0.5 + 0.25)
        assert abs(result - expected) < 0.01


class TestStatisticalAnalyzer:
    def setup_method(self):
        self.analyzer = StatisticalAnalyzer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_mean(self):
        assert self.analyzer.mean() == 5.5

    def test_describe_has_all_keys(self):
        d = self.analyzer.describe()
        for key in ["count", "mean", "median", "std_dev", "min", "max"]:
            assert key in d

    def test_autocorrelation_list(self):
        ac = self.analyzer.autocorrelation_function(3)
        assert len(ac) == 3

    def test_outliers_iqr(self):
        analyzer = StatisticalAnalyzer([1, 2, 3, 4, 5, 100])
        outliers = analyzer.outliers_iqr()
        assert 100 in outliers

    def test_outliers_zscore(self):
        analyzer = StatisticalAnalyzer([1, 2, 3, 4, 5, 100])
        outliers = analyzer.outliers_zscore(2.0)
        assert 100 in outliers

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            StatisticalAnalyzer([])
