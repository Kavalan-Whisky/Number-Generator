"""Tests for entropy_analyzer module."""
import pytest
import math
from src.core.analyzers.entropy_analyzer import (
    shannon_entropy, min_entropy, collision_entropy,
    approximate_entropy, sample_entropy,
    lempel_ziv_complexity, normalized_lz_complexity,
    compression_ratio, EntropyAnalyzer,
)


class TestShannonEntropy:
    def test_uniform_binary(self):
        # 50 zeros, 50 ones → entropy = 1 bit
        seq = [0] * 50 + [1] * 50
        h = shannon_entropy(seq)
        assert h == pytest.approx(1.0, abs=0.01)

    def test_constant_zero(self):
        seq = [5] * 100
        h = shannon_entropy(seq)
        assert h == pytest.approx(0.0)

    def test_maximum(self):
        # 4 symbols, equal probability → 2 bits
        seq = [0, 1, 2, 3] * 25
        h = shannon_entropy(seq)
        assert h == pytest.approx(2.0, abs=0.01)

    def test_range(self):
        seq = list(range(16)) * 10
        h = shannon_entropy(seq)
        assert 0 <= h <= math.log2(16)

    def test_empty(self):
        h = shannon_entropy([])
        assert h == 0.0 or math.isnan(h)


class TestMinEntropy:
    def test_constant(self):
        seq = [7] * 100
        h = min_entropy(seq)
        assert h == pytest.approx(0.0)

    def test_uniform(self):
        seq = list(range(8)) * 10
        h = min_entropy(seq)
        # min-entropy = -log2(max_prob) = -log2(1/8) = 3
        assert h == pytest.approx(3.0, abs=0.01)

    def test_nonnegative(self):
        seq = [1, 2, 3, 1, 2, 1]
        assert min_entropy(seq) >= 0


class TestCollisionEntropy:
    def test_constant(self):
        seq = [3] * 50
        h = collision_entropy(seq)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_uniform(self):
        seq = [0, 1] * 50
        h = collision_entropy(seq)
        # H2 should be near 1 for uniform binary
        assert h == pytest.approx(1.0, abs=0.1)

    def test_nonnegative(self):
        seq = list(range(5)) * 10
        assert collision_entropy(seq) >= 0


class TestApproximateEntropy:
    def test_constant_low(self):
        seq = [1.0] * 50
        apen = approximate_entropy(seq)
        assert apen <= 0.01 or math.isinf(apen) or math.isnan(apen)

    def test_returns_float(self):
        seq = [float(i % 6) for i in range(30)]
        result = approximate_entropy(seq)
        assert isinstance(result, float)


class TestSampleEntropy:
    def test_returns_float(self):
        seq = [float(i % 5) for i in range(40)]
        result = sample_entropy(seq)
        assert isinstance(result, float)

    def test_constant_low(self):
        seq = [3.0] * 40
        result = sample_entropy(seq)
        assert result == float("inf") or result <= 0.1


class TestLempelZivComplexity:
    def test_constant_less_than_random(self):
        # constant should have lower LZ complexity than a varying sequence
        const = [0] * 100
        varying = list(range(100))
        assert lempel_ziv_complexity(const) < lempel_ziv_complexity(varying)

    def test_unique_high(self):
        seq = list(range(50))
        c = lempel_ziv_complexity(seq)
        assert c > 0

    def test_normalized_range(self):
        seq = list(range(40))
        n = normalized_lz_complexity(seq)
        assert n > 0


class TestCompressionRatio:
    def test_constant_low_ratio(self):
        seq = [0] * 500
        ratio = compression_ratio(seq)
        assert ratio < 0.3

    def test_range(self):
        seq = list(range(100))
        ratio = compression_ratio(seq)
        assert ratio > 0


class TestEntropyAnalyzer:
    def setup_method(self):
        self.analyzer = EntropyAnalyzer()

    def test_analyze_returns_dict(self):
        seq = [i % 4 for i in range(100)]
        result = self.analyzer.analyze(seq)
        assert isinstance(result, dict)
        assert "shannon_entropy" in result

    def test_shannon_value(self):
        seq = [0, 1] * 50
        result = self.analyzer.analyze(seq)
        assert result["shannon_entropy"] == pytest.approx(1.0, abs=0.01)

    def test_all_nonnegative(self):
        seq = list(range(20)) * 5
        result = self.analyzer.analyze(seq)
        for k, v in result.items():
            if isinstance(v, float) and not math.isnan(v) and not math.isinf(v):
                assert v >= 0, f"Negative entropy: {k}={v}"
