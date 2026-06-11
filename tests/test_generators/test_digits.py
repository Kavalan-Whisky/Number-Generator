"""Tests for digits_generator module."""
import pytest
from src.core.generators.digits_generator import (
    pi_digits, pi_digits_string, e_digits, sqrt2_digits,
    golden_ratio_digits, champernowne_digits, thue_morse, thue_morse_iterator,
    isqrt_digits, DigitsGeneratorFactory,
)


class TestPiDigits:
    def test_first_few(self):
        digits = pi_digits(10)
        assert digits[:4] == [3, 1, 4, 1]

    def test_known_pi(self):
        # 3.14159265358979...
        digits = pi_digits(15)
        expected = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
        assert digits[:15] == expected

    def test_length(self):
        for n in [1, 5, 20, 50]:
            assert len(pi_digits(n)) == n

    def test_all_digits(self):
        digits = pi_digits(30)
        assert all(0 <= d <= 9 for d in digits)

    def test_pi_string(self):
        s = pi_digits_string(10)
        assert s.startswith("3.14159265")


class TestEDigits:
    def test_first_few(self):
        digits = e_digits(10)
        # e = 2.71828182845...
        assert digits[:4] == [2, 7, 1, 8]

    def test_length(self):
        assert len(e_digits(20)) == 20

    def test_all_decimal(self):
        assert all(0 <= d <= 9 for d in e_digits(30))


class TestSqrt2Digits:
    def test_first_few(self):
        digits = sqrt2_digits(10)
        # sqrt(2) = 1.41421356237...
        assert digits[0] == 1
        assert digits[1] == 4
        assert digits[2] == 1

    def test_length(self):
        assert len(sqrt2_digits(15)) == 15


class TestGoldenRatioDigits:
    def test_first_few(self):
        digits = golden_ratio_digits(10)
        # phi = 1.61803398874...
        assert digits[0] == 1
        assert digits[1] == 6
        assert digits[2] == 1

    def test_length(self):
        assert len(golden_ratio_digits(20)) == 20


class TestChampernowneDigits:
    def test_known(self):
        # 1 2 3 4 5 6 7 8 9 1 0 1 1 1 2 ...
        digits = champernowne_digits(9)
        assert digits == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_next_digits(self):
        digits = champernowne_digits(11)
        assert digits[9] == 1
        assert digits[10] == 0

    def test_length(self):
        for n in [5, 15, 30]:
            assert len(champernowne_digits(n)) == n


class TestThueMorse:
    def test_first_terms(self):
        # 0 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 ...
        expected = [0, 1, 1, 0, 1, 0, 0, 1]
        result = thue_morse(8)
        assert result == expected

    def test_length(self):
        assert len(thue_morse(32)) == 32

    def test_binary(self):
        seq = thue_morse(50)
        assert all(b in (0, 1) for b in seq)

    def test_iterator(self):
        it = thue_morse_iterator()
        seq = [next(it) for _ in range(8)]
        assert seq == [0, 1, 1, 0, 1, 0, 0, 1]

    def test_complement_property(self):
        # TM[2n] + TM[2n+1] == 1 (they differ)
        seq = thue_morse(16)
        for i in range(0, 16, 2):
            assert seq[i] != seq[i + 1]


class TestIsqrtDigits:
    def test_sqrt2(self):
        digits = isqrt_digits(2, 10)
        assert digits[0] == 1
        assert digits[1] == 4

    def test_sqrt3(self):
        digits = isqrt_digits(3, 5)
        assert digits[0] == 1
        assert digits[1] == 7

    def test_perfect_square(self):
        digits = isqrt_digits(4, 5)
        assert digits[0] == 2


class TestDigitsGeneratorFactory:
    def test_pi(self):
        d = DigitsGeneratorFactory.generate("pi", 10)
        assert d[0] == 3

    def test_e(self):
        d = DigitsGeneratorFactory.generate("e", 10)
        assert d[0] == 2

    def test_thue_morse(self):
        seq = DigitsGeneratorFactory.generate("thue_morse", 8)
        assert seq[0] == 0
        assert seq[1] == 1

    def test_available_constants(self):
        consts = DigitsGeneratorFactory.available_constants()
        assert "pi" in consts
        assert "e" in consts

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            DigitsGeneratorFactory.generate("unknown_const", 10)
