"""Tests for numeral_systems transformer module.

Note: most functions return string representations of the numeral system.
Only factoradic returns a list of integers.
"""
import pytest
from src.core.transformers.numeral_systems import (
    to_balanced_ternary, from_balanced_ternary,
    to_factoradic, from_factoradic,
    to_zeckendorf, from_zeckendorf, is_valid_zeckendorf,
    to_gray, from_gray, gray_sequence,
    to_bcd, from_bcd,
    to_negabinary, from_negabinary,
    to_bijective, from_bijective,
    NumeralSystemConverter,
)


class TestBalancedTernary:
    def test_zero(self):
        # Returns string representation
        assert from_balanced_ternary(to_balanced_ternary(0)) == 0

    def test_one(self):
        assert from_balanced_ternary(to_balanced_ternary(1)) == 1

    def test_two(self):
        assert from_balanced_ternary(to_balanced_ternary(2)) == 2

    def test_roundtrip_positive(self):
        for n in range(1, 21):
            assert from_balanced_ternary(to_balanced_ternary(n)) == n

    def test_roundtrip_negative(self):
        for n in range(-20, 0):
            assert from_balanced_ternary(to_balanced_ternary(n)) == n

    def test_string_output(self):
        result = to_balanced_ternary(5)
        assert isinstance(result, str)

    def test_negative(self):
        assert from_balanced_ternary(to_balanced_ternary(-5)) == -5


class TestFactoradic:
    def test_zero(self):
        assert from_factoradic(to_factoradic(0)) == 0

    def test_one(self):
        assert from_factoradic(to_factoradic(1)) == 1

    def test_roundtrip(self):
        for n in range(50):
            assert from_factoradic(to_factoradic(n)) == n

    def test_known_5(self):
        f = to_factoradic(5)
        assert from_factoradic(f) == 5

    def test_known_23(self):
        assert from_factoradic(to_factoradic(23)) == 23

    def test_list_output(self):
        # factoradic returns a list
        result = to_factoradic(5)
        assert isinstance(result, list)


class TestZeckendorf:
    def test_one(self):
        assert from_zeckendorf(to_zeckendorf(1)) == 1

    def test_roundtrip(self):
        for n in range(1, 50):
            assert from_zeckendorf(to_zeckendorf(n)) == n

    def test_no_adjacent_ones(self):
        for n in range(1, 50):
            z = to_zeckendorf(n)
            assert "11" not in z, f"Adjacent 1s in Zeckendorf({n}): {z}"

    def test_is_valid(self):
        for n in range(1, 30):
            z = to_zeckendorf(n)
            assert is_valid_zeckendorf(z)

    def test_invalid_zeckendorf(self):
        assert not is_valid_zeckendorf("110")  # adjacent 1s as string


class TestGrayCode:
    def test_zero(self):
        assert to_gray(0) == 0
        assert from_gray(0) == 0

    def test_known(self):
        assert to_gray(1) == 1
        assert to_gray(2) == 3
        assert to_gray(3) == 2
        assert to_gray(4) == 6

    def test_roundtrip(self):
        for n in range(256):
            assert from_gray(to_gray(n)) == n

    def test_consecutive_differ_by_one_bit(self):
        for n in range(63):
            g1 = to_gray(n)
            g2 = to_gray(n + 1)
            diff = g1 ^ g2
            assert diff & (diff - 1) == 0

    def test_sequence_length(self):
        seq = gray_sequence(8)
        assert len(seq) == 8


class TestBCD:
    def test_zero(self):
        assert from_bcd(to_bcd(0)) == 0

    def test_single_digit(self):
        for d in range(10):
            assert from_bcd(to_bcd(d)) == d

    def test_roundtrip(self):
        for n in [0, 9, 12, 99, 100, 999, 1234]:
            assert from_bcd(to_bcd(n)) == n

    def test_known_42(self):
        assert from_bcd(to_bcd(42)) == 42

    def test_string_output(self):
        result = to_bcd(42)
        assert isinstance(result, str)


class TestNegabinary:
    def test_zero(self):
        assert from_negabinary(to_negabinary(0)) == 0

    def test_one(self):
        assert from_negabinary(to_negabinary(1)) == 1

    def test_roundtrip_positive(self):
        for n in range(30):
            assert from_negabinary(to_negabinary(n)) == n

    def test_roundtrip_negative(self):
        for n in range(-15, 0):
            assert from_negabinary(to_negabinary(n)) == n

    def test_string_output(self):
        result = to_negabinary(5)
        assert isinstance(result, str)
        assert all(c in "01" for c in result)


class TestBijectiveBase:
    def test_roundtrip_base10(self):
        for n in range(1, 50):
            assert from_bijective(to_bijective(n, 10), 10) == n

    def test_roundtrip_base2(self):
        for n in range(1, 20):
            assert from_bijective(to_bijective(n, 2), 2) == n

    def test_one_is_one(self):
        # In bijective base-10, digit 1 represents value 1
        assert from_bijective(to_bijective(1, 10), 10) == 1


class TestNumeralSystemConverter:
    def test_balanced_ternary_roundtrip(self):
        for n in range(-10, 11):
            enc = NumeralSystemConverter.encode(n, "balanced_ternary")
            assert NumeralSystemConverter.decode(enc, "balanced_ternary") == n

    def test_gray_code_roundtrip(self):
        for n in range(50):
            enc = NumeralSystemConverter.encode(n, "gray")
            assert NumeralSystemConverter.decode(enc, "gray") == n

    def test_zeckendorf_roundtrip(self):
        for n in range(1, 30):
            enc = NumeralSystemConverter.encode(n, "zeckendorf")
            assert NumeralSystemConverter.decode(enc, "zeckendorf") == n

    def test_available_systems(self):
        systems = NumeralSystemConverter.available_systems()
        assert "balanced_ternary" in systems
        assert "gray" in systems
        assert "zeckendorf" in systems

    def test_unknown_system(self):
        with pytest.raises(ValueError):
            NumeralSystemConverter.encode(5, "unknown_base")
