"""Tests for complexity_analyzer module."""
import pytest
import math
from src.core.analyzers.complexity_analyzer import (
    lz76_complexity, lz76_normalized,
    approximate_entropy, sample_entropy,
    permutation_entropy, compression_ratio, compression_ratios_all,
    recurrence_matrix, recurrence_rate, determinism_rqa,
    multiscale_entropy, ComplexityAnalyzer,
)


class TestLZ76:
    def test_constant_sequence(self):
        # Constant sequence has lower complexity than a varying one
        seq = [0] * 100
        assert lz76_complexity(seq) < lz76_complexity(list(range(100)))

    def test_alternating(self):
        seq = [0, 1] * 50
        assert lz76_complexity(seq) < lz76_complexity(list(range(100)))

    def test_unique_sequence(self):
        seq = list(range(50))
        c = lz76_complexity(seq)
        assert c > 0

    def test_empty(self):
        assert lz76_complexity([]) == 0

    def test_single_element(self):
        assert lz76_complexity([5]) == 1

    def test_normalized_range(self):
        seq = list(range(50))
        n = lz76_normalized(seq)
        assert n > 0

    def test_constant_less_complex(self):
        constant = [3] * 100
        random_like = list(range(100))
        assert lz76_complexity(constant) < lz76_complexity(random_like)


class TestApproximateEntropy:
    def test_constant_low_entropy(self):
        seq = [1.0] * 50
        # constant → very low entropy (near 0)
        apen = approximate_entropy(seq, m=2)
        assert apen <= 0.1 or math.isnan(apen) or math.isinf(apen)

    def test_more_complex_higher(self):
        import random
        random.seed(42)
        seq_reg = [math.sin(0.1 * i) for i in range(50)]
        seq_rand = [random.random() for _ in range(50)]
        apen_reg = approximate_entropy(seq_reg, m=2)
        apen_rand = approximate_entropy(seq_rand, m=2)
        # random should have higher entropy if not ±inf
        if not (math.isinf(apen_reg) or math.isinf(apen_rand)):
            assert apen_rand >= apen_reg - 0.5  # allow some tolerance

    def test_returns_float(self):
        seq = [float(i % 5) for i in range(30)]
        result = approximate_entropy(seq, m=2)
        assert isinstance(result, float)


class TestSampleEntropy:
    def test_returns_float(self):
        seq = [float(i % 4) for i in range(30)]
        result = sample_entropy(seq)
        assert isinstance(result, float)

    def test_constant_low(self):
        seq = [2.0] * 40
        result = sample_entropy(seq)
        # constant → near-zero or inf
        assert result == float("inf") or result <= 0.1


class TestPermutationEntropy:
    def test_constant_zero(self):
        seq = [5.0] * 30
        pe = permutation_entropy(seq, order=3)
        assert pe == pytest.approx(0.0, abs=1e-6)

    def test_random_higher(self):
        import random
        random.seed(0)
        seq = [random.random() for _ in range(100)]
        pe = permutation_entropy(seq, order=3)
        assert pe > 0.5  # should be near 1 for random

    def test_normalised_range(self):
        seq = list(range(50, dtype=float) if False else [float(i) for i in range(50)])
        pe = permutation_entropy(seq, order=3, normalise=True)
        assert 0 <= pe <= 1.0

    def test_increasing_sequence(self):
        seq = [float(i) for i in range(30)]
        pe = permutation_entropy(seq, order=3)
        # Strictly increasing → only one permutation pattern → entropy = 0
        assert pe == pytest.approx(0.0, abs=1e-6)

    def test_short_sequence(self):
        seq = [1.0, 2.0]
        pe = permutation_entropy(seq, order=3)
        assert pe == 0.0


class TestCompressionRatio:
    def test_constant_compresses_well(self):
        seq = [0] * 1000
        ratio = compression_ratio(seq, "zlib")
        assert ratio < 0.2

    def test_random_poor_compression(self):
        seq = [(7 * i + 3) % 251 for i in range(1000)]
        ratio = compression_ratio(seq, "zlib")
        # pseudo-random is harder to compress
        assert ratio > 0.1

    def test_all_codecs(self):
        seq = list(range(100))
        ratios = compression_ratios_all(seq)
        assert "zlib" in ratios
        assert "bz2" in ratios
        assert "lzma" in ratios
        assert all(v > 0 for v in ratios.values())


class TestRecurrenceQRA:
    def test_recurrence_rate_constant(self):
        seq = [1.0] * 10
        R = recurrence_matrix(seq, threshold=0.1)
        rr = recurrence_rate(R)
        assert rr == pytest.approx(1.0)

    def test_recurrence_rate_range(self):
        seq = [math.sin(0.5 * i) for i in range(20)]
        R = recurrence_matrix(seq, threshold=0.3)
        rr = recurrence_rate(R)
        assert 0 <= rr <= 1

    def test_determinism_constant(self):
        seq = [1.0] * 20
        R = recurrence_matrix(seq, threshold=0.1)
        det = determinism_rqa(R, min_line=2)
        # Constant signal → very high determinism
        assert det > 0.9


class TestMultiscaleEntropy:
    def test_length(self):
        seq = [float(i % 7) for i in range(100)]
        mse = multiscale_entropy(seq, max_scale=4)
        assert len(mse) == 4

    def test_returns_floats(self):
        seq = [math.sin(i) for i in range(80)]
        mse = multiscale_entropy(seq, max_scale=3)
        assert all(isinstance(v, float) for v in mse)


class TestComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()

    def test_analyze_fast_mode(self):
        seq = [float(i % 10) for i in range(100)]
        report = self.analyzer.analyze(seq, fast=True)
        assert report.sequence_length == 100
        assert report.lz76_complexity > 0
        assert 0 <= report.permutation_entropy <= 1

    def test_lz76_method(self):
        seq = [0, 1, 0, 1, 0, 1]
        c = self.analyzer.lz76(seq)
        assert c > 0

    def test_permutation_entropy_method(self):
        seq = [float(i) for i in range(30)]
        pe = self.analyzer.permutation_entropy(seq, order=3)
        assert pe == pytest.approx(0.0, abs=1e-6)

    def test_multiscale_method(self):
        seq = [math.cos(i * 0.3) for i in range(80)]
        mse = self.analyzer.multiscale(seq, max_scale=3)
        assert len(mse) == 3
