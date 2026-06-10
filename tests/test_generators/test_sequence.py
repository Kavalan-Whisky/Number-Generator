"""Tests for mathematical sequence generators."""

import pytest
from src.core.generators.sequence_generator import (
    ArithmeticSequence, GeometricSequence, HarmonicSequence,
    TriangularNumbers, SquareNumbers, PentagonalNumbers, HexagonalNumbers,
    CatalanNumbers, BellNumbers, CollatzSequence, LookAndSay,
    RecamanSequence, PadovanSequence, PerrinSequence,
    SequenceGeneratorFactory,
)


class TestArithmeticSequence:
    def test_basic(self):
        s = ArithmeticSequence(1, 2)
        assert s.generate(5) == [1, 3, 5, 7, 9]

    def test_nth(self):
        s = ArithmeticSequence(0, 3)
        assert s.nth(4) == 12

    def test_sum(self):
        s = ArithmeticSequence(1, 1)
        assert s.sum(10) == 55

    def test_find_index(self):
        s = ArithmeticSequence(2, 3)
        assert s.find_index(11) == 3

    def test_find_index_not_found(self):
        s = ArithmeticSequence(2, 3)
        assert s.find_index(12) is None


class TestGeometricSequence:
    def test_basic(self):
        s = GeometricSequence(1, 2)
        assert s.generate(5) == [1, 2, 4, 8, 16]

    def test_sum(self):
        s = GeometricSequence(1, 2)
        assert s.sum(4) == 15

    def test_infinite_sum(self):
        s = GeometricSequence(1, 0.5)
        assert abs(s.infinite_sum() - 2.0) < 1e-9

    def test_no_infinite_sum_for_r_ge_1(self):
        s = GeometricSequence(1, 2)
        assert s.infinite_sum() is None


class TestHarmonicSequence:
    def test_generate(self):
        s = HarmonicSequence()
        seq = s.generate(4)
        assert seq == [1.0, 0.5, 1/3, 0.25]

    def test_partial_sum(self):
        s = HarmonicSequence()
        h1 = s.partial_sum(1)
        assert h1 == 1.0
        h2 = s.partial_sum(2)
        assert abs(h2 - 1.5) < 1e-10


class TestTriangularNumbers:
    def test_sequence(self):
        seq = TriangularNumbers.generate(6)
        assert seq == [1, 3, 6, 10, 15, 21]

    def test_is_triangular(self):
        for t in [1, 3, 6, 10, 15, 21]:
            assert TriangularNumbers.is_triangular(t)
        assert not TriangularNumbers.is_triangular(4)

    def test_index_of(self):
        assert TriangularNumbers.index_of(10) == 4


class TestSquareNumbers:
    def test_sequence(self):
        seq = SquareNumbers.generate(5)
        assert seq == [1, 4, 9, 16, 25]


class TestPentagonalNumbers:
    def test_sequence(self):
        seq = PentagonalNumbers.generate(5)
        assert seq == [1, 5, 12, 22, 35]

    def test_is_pentagonal(self):
        for p in [1, 5, 12, 22, 35]:
            assert PentagonalNumbers.is_pentagonal(p)


class TestCatalanNumbers:
    CATALAN = [1, 1, 2, 5, 14, 42, 132, 429]

    def test_basic(self):
        for i, expected in enumerate(self.CATALAN):
            assert CatalanNumbers.nth(i) == expected

    def test_generate(self):
        assert CatalanNumbers.generate(8) == self.CATALAN


class TestBellNumbers:
    BELL = [1, 1, 2, 5, 15, 52]

    def test_basic(self):
        for i, expected in enumerate(self.BELL):
            assert BellNumbers.nth(i) == expected

    def test_generate(self):
        assert BellNumbers.generate(6) == self.BELL

    def test_triangle(self):
        tri = BellNumbers.triangle(4)
        assert tri[0] == [1]
        assert tri[1][0] == tri[0][-1]


class TestCollatzSequence:
    def test_starts_with_n(self):
        seq = CollatzSequence.sequence(6)
        assert seq[0] == 6

    def test_ends_at_1(self):
        for n in range(1, 30):
            seq = CollatzSequence.sequence(n)
            assert seq[-1] == 1

    def test_stopping_time(self):
        assert CollatzSequence.stopping_time(1) == 0
        assert CollatzSequence.stopping_time(2) == 1

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            CollatzSequence.sequence(0)


class TestLookAndSay:
    def test_first_terms(self):
        terms = LookAndSay.generate(6)
        assert terms[0] == "1"
        assert terms[1] == "11"
        assert terms[2] == "21"
        assert terms[3] == "1211"

    def test_as_int(self):
        nums = LookAndSay.generate_as_int(4)
        assert nums[0] == 1
        assert nums[1] == 11


class TestRecamanSequence:
    def test_starts_at_zero(self):
        seq = RecamanSequence.generate(10)
        assert seq[0] == 0

    def test_length(self):
        seq = RecamanSequence.generate(20)
        assert len(seq) == 20

    def test_known_values(self):
        seq = RecamanSequence.generate(10)
        assert seq[:6] == [0, 1, 3, 6, 2, 7]


class TestPadovanSequence:
    def test_starts_with_ones(self):
        seq = PadovanSequence.generate(5)
        assert seq[:3] == [1, 1, 1]

    def test_recurrence(self):
        seq = PadovanSequence.generate(10)
        for i in range(3, 10):
            assert seq[i] == seq[i-2] + seq[i-3]


class TestPerrinSequence:
    def test_starts_correctly(self):
        seq = PerrinSequence.generate(5)
        assert seq[:3] == [3, 0, 2]

    def test_recurrence(self):
        seq = PerrinSequence.generate(10)
        for i in range(3, 10):
            assert seq[i] == seq[i-2] + seq[i-3]


class TestSequenceGeneratorFactory:
    def test_arithmetic(self):
        result = SequenceGeneratorFactory.generate("arithmetic", 5, start=1, difference=2)
        assert result == [1, 3, 5, 7, 9]

    def test_catalan(self):
        result = SequenceGeneratorFactory.generate("catalan", 5)
        assert result == [1, 1, 2, 5, 14]

    def test_triangular(self):
        result = SequenceGeneratorFactory.generate("triangular", 4)
        assert result == [1, 3, 6, 10]

    def test_recaman(self):
        result = SequenceGeneratorFactory.generate("recaman", 5)
        assert len(result) == 5

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            SequenceGeneratorFactory.generate("unknown_sequence", 5)

    def test_available_types(self):
        types = SequenceGeneratorFactory.available_types()
        assert "arithmetic" in types
        assert "catalan" in types
        assert "bell" in types
