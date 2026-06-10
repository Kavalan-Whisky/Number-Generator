"""Tests for number formatters and sequence transformers."""

import pytest
from src.core.transformers.number_formatter import (
    to_binary, to_octal, to_hex, to_base_n, from_base_n,
    to_roman_numeral, from_roman_numeral,
    to_scientific, to_engineering, to_words, to_morse_code,
    from_morse_code, NumberFormatter,
)
from src.core.transformers.sequence_transformer import (
    running_sum, running_product, running_max, running_min,
    moving_average, exponential_moving_average, differences,
    normalize, standardize, interleave, chunk, window_slide,
    delta_encode, delta_decode, SequenceTransformer,
)


class TestBinaryOctalHex:
    def test_binary(self):
        assert to_binary(10) == "1010"
        assert to_binary(255) == "11111111"
        assert to_binary(0) == "0"

    def test_binary_negative(self):
        assert to_binary(-5) == "-101"

    def test_binary_width(self):
        assert to_binary(5, width=8) == "00000101"

    def test_octal(self):
        assert to_octal(8) == "10"
        assert to_octal(255) == "377"

    def test_hex(self):
        assert to_hex(255) == "ff"
        assert to_hex(255, uppercase=True) == "FF"
        assert to_hex(0) == "0"


class TestBaseN:
    def test_base_2(self):
        assert to_base_n(10, 2) == "1010"

    def test_base_16(self):
        assert to_base_n(255, 16) == "FF"

    def test_base_8(self):
        assert to_base_n(8, 8) == "10"

    def test_roundtrip(self):
        for n in range(1, 100):
            for base in [2, 8, 16]:
                s = to_base_n(n, base)
                assert from_base_n(s, base) == n

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            to_base_n(10, 1)


class TestRomanNumerals:
    def test_basic(self):
        assert to_roman_numeral(1) == "I"
        assert to_roman_numeral(4) == "IV"
        assert to_roman_numeral(9) == "IX"
        assert to_roman_numeral(14) == "XIV"
        assert to_roman_numeral(40) == "XL"
        assert to_roman_numeral(90) == "XC"
        assert to_roman_numeral(400) == "CD"
        assert to_roman_numeral(900) == "CM"
        assert to_roman_numeral(1994) == "MCMXCIV"

    def test_from_roman(self):
        assert from_roman_numeral("IV") == 4
        assert from_roman_numeral("XIV") == 14
        assert from_roman_numeral("MCMXCIV") == 1994

    def test_roundtrip(self):
        for n in range(1, 200):
            assert from_roman_numeral(to_roman_numeral(n)) == n

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            to_roman_numeral(0)
        with pytest.raises(ValueError):
            to_roman_numeral(4000)


class TestScientificNotation:
    def test_basic(self):
        result = to_scientific(1234.5)
        assert "e" in result.lower() or "E" in result

    def test_small_number(self):
        result = to_scientific(0.001)
        assert result


class TestNumberToWords:
    def test_zero(self):
        assert to_words(0) == "zero"

    def test_small(self):
        assert to_words(1) == "one"
        assert to_words(11) == "eleven"
        assert to_words(20) == "twenty"

    def test_hundred(self):
        assert "hundred" in to_words(100)

    def test_negative(self):
        assert to_words(-5) == "negative five"

    def test_thousand(self):
        assert "thousand" in to_words(1000)

    def test_compound(self):
        result = to_words(21)
        assert "twenty" in result and "one" in result


class TestMorseCode:
    def test_digits(self):
        assert to_morse_code("1") == ".----"
        assert to_morse_code("0") == "-----"

    def test_multi_digit(self):
        result = to_morse_code("42")
        assert "....-" in result and "..---" in result

    def test_roundtrip(self):
        original = "123"
        encoded = to_morse_code(original)
        decoded = from_morse_code(encoded)
        assert decoded == original


class TestNumberFormatter:
    def test_format_sequence_binary(self):
        result = NumberFormatter.format_sequence([5, 10, 15], "binary")
        assert result == ["101", "1010", "1111"]

    def test_format_sequence_hex(self):
        result = NumberFormatter.format_sequence([255, 16], "hex")
        assert result == ["ff", "10"]

    def test_format_sequence_roman(self):
        result = NumberFormatter.format_sequence([1, 4, 9], "roman")
        assert result == ["I", "IV", "IX"]

    def test_unknown_format(self):
        with pytest.raises(ValueError):
            NumberFormatter.format_sequence([1], "unknown_fmt")


class TestRunningAggregates:
    def test_running_sum(self):
        result = running_sum([1, 2, 3, 4, 5])
        assert result == [1, 3, 6, 10, 15]

    def test_running_product(self):
        result = running_product([1, 2, 3, 4])
        assert result == [1, 2, 6, 24]

    def test_running_max(self):
        result = running_max([3, 1, 4, 1, 5, 9])
        assert result == [3, 3, 4, 4, 5, 9]

    def test_running_min(self):
        result = running_min([3, 1, 4, 1, 5])
        assert result == [3, 1, 1, 1, 1]


class TestMovingAverages:
    def test_moving_average(self):
        result = moving_average([1, 2, 3, 4, 5], window=3)
        assert result == [2.0, 3.0, 4.0]

    def test_ema(self):
        result = exponential_moving_average([1, 2, 3, 4, 5], alpha=0.5)
        assert len(result) == 5
        assert result[0] == 1.0

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            moving_average([1, 2], window=5)


class TestDifferences:
    def test_first_order(self):
        result = differences([1, 4, 9, 16], order=1)
        assert result == [3, 5, 7]

    def test_second_order(self):
        result = differences([1, 4, 9, 16], order=2)
        assert result == [2, 2]


class TestNormalize:
    def test_normalize_range(self):
        result = normalize([0, 5, 10])
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert result[1] == pytest.approx(0.5)

    def test_standardize(self):
        result = standardize([1, 2, 3, 4, 5])
        import math
        mean_z = sum(result) / len(result)
        assert abs(mean_z) < 1e-10

    def test_constant_normalize(self):
        result = normalize([5, 5, 5])
        assert result == [0.5, 0.5, 0.5]


class TestInterleaveChunk:
    def test_interleave(self):
        result = interleave([1, 2, 3], [4, 5, 6])
        assert result == [1, 4, 2, 5, 3, 6]

    def test_chunk(self):
        result = chunk([1, 2, 3, 4, 5, 6], size=2)
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_window_slide(self):
        result = window_slide([1, 2, 3, 4, 5], window=3)
        assert result == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


class TestDeltaEncodeDecode:
    def test_encode(self):
        result = delta_encode([5, 8, 12, 15])
        assert result[0] == 5
        assert result[1] == 3
        assert result[2] == 4

    def test_decode(self):
        data = [1, 2, 3, 4, 5]
        encoded = delta_encode(data)
        decoded = delta_decode(encoded)
        assert decoded == data

    def test_roundtrip(self):
        data = [10, 25, 17, 100, 50]
        assert delta_decode(delta_encode(data)) == data


class TestSequenceTransformer:
    def setup_method(self):
        self.t = SequenceTransformer([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_normalize(self):
        result = self.t.normalize()
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_sort(self):
        t = SequenceTransformer([3, 1, 4, 1, 5, 9])
        assert t.sort() == [1, 1, 3, 4, 5, 9]

    def test_reverse(self):
        assert self.t.reverse() == [5, 4, 3, 2, 1]

    def test_unique(self):
        t = SequenceTransformer([1, 2, 2, 3, 3, 3])
        assert t.unique() == [1, 2, 3]
