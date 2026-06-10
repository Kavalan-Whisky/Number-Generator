"""
Number Formatter - Convert numbers to various representations.
Binary, octal, hex, roman numerals, scientific notation, words, morse code.
"""

import math
from typing import List, Optional, Union


def to_binary(n: int, width: Optional[int] = None) -> str:
    """Convert integer to binary string."""
    if n < 0:
        return "-" + to_binary(-n, width)
    result = bin(n)[2:]
    if width:
        result = result.zfill(width)
    return result


def to_octal(n: int) -> str:
    """Convert integer to octal string."""
    if n < 0:
        return "-" + to_octal(-n)
    return oct(n)[2:]


def to_hex(n: int, uppercase: bool = False) -> str:
    """Convert integer to hexadecimal string."""
    if n < 0:
        return "-" + to_hex(-n, uppercase)
    h = hex(n)[2:]
    return h.upper() if uppercase else h


def to_base_n(n: int, base: int, digits: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    """Convert integer to any base."""
    if base < 2 or base > len(digits):
        raise ValueError(f"Base must be between 2 and {len(digits)}")
    if n == 0:
        return digits[0]
    if n < 0:
        return "-" + to_base_n(-n, base, digits)
    result = []
    while n:
        result.append(digits[n % base])
        n //= base
    return "".join(reversed(result))


def from_base_n(s: str, base: int) -> int:
    """Convert string in given base to integer."""
    return int(s, base)


ROMAN_NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman_numeral(n: int) -> str:
    """Convert integer to Roman numeral."""
    if n <= 0 or n > 3999:
        raise ValueError("Roman numerals support 1-3999")
    result = []
    for value, symbol in ROMAN_NUMERALS:
        while n >= value:
            result.append(symbol)
            n -= value
    return "".join(result)


def from_roman_numeral(s: str) -> int:
    """Convert Roman numeral to integer."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper()
    result = 0
    prev = 0
    for char in reversed(s):
        if char not in values:
            raise ValueError(f"Invalid Roman numeral character: {char}")
        curr = values[char]
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def to_scientific(n: float, precision: int = 6) -> str:
    """Convert number to scientific notation string."""
    if n == 0:
        return f"0.{'0' * precision}e+00"
    exp = int(math.floor(math.log10(abs(n))))
    mantissa = n / (10**exp)
    return f"{mantissa:.{precision}f}e{exp:+03d}"


def to_engineering(n: float, precision: int = 3) -> str:
    """Convert to engineering notation (exponent multiple of 3)."""
    if n == 0:
        return f"0.{'0' * precision}e+00"
    exp = int(math.floor(math.log10(abs(n))))
    eng_exp = (exp // 3) * 3
    mantissa = n / (10**eng_exp)
    return f"{mantissa:.{precision}f}e{eng_exp:+03d}"


# Number to words
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]


def _words_under_1000(n: int) -> str:
    if n == 0:
        return ""
    if n < 20:
        return ONES[n]
    if n < 100:
        tens_part = TENS[n // 10]
        ones_part = ONES[n % 10]
        return tens_part + ("-" + ones_part if ones_part else "")
    hundreds = n // 100
    remainder = n % 100
    parts = [ONES[hundreds] + " hundred"]
    if remainder:
        parts.append(_words_under_1000(remainder))
    return " ".join(parts)


def to_words(n: Union[int, float]) -> str:
    """Convert number to English words."""
    if isinstance(n, float):
        int_part = int(n)
        dec_part = round(abs(n) - abs(int_part), 10)
        words = to_words(int_part)
        if dec_part > 0:
            dec_str = str(dec_part)[2:]  # Remove "0."
            dec_words = " ".join(to_words(int(d)) for d in dec_str if d != "0")
            words += " point " + dec_words
        return words

    n = int(n)
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + to_words(-n)

    parts = []
    scale_idx = 0
    while n > 0:
        chunk = n % 1000
        if chunk != 0:
            chunk_words = _words_under_1000(chunk)
            if scale_idx > 0:
                chunk_words += " " + SCALES[scale_idx]
            parts.append(chunk_words)
        n //= 1000
        scale_idx += 1
        if scale_idx >= len(SCALES) and n > 0:
            break

    return " ".join(reversed(parts))


MORSE_CODE = {
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "!": "-.-.--",
    "-": "-....-", " ": "/",
}


def to_morse_code(s: str) -> str:
    """Convert string (digits/letters) to morse code."""
    result = []
    for char in str(s).upper():
        if char in MORSE_CODE:
            result.append(MORSE_CODE[char])
        else:
            result.append("?")
    return " ".join(result)


def from_morse_code(s: str) -> str:
    """Decode morse code string."""
    reverse = {v: k for k, v in MORSE_CODE.items()}
    result = []
    for code in s.split(" "):
        if code in reverse:
            result.append(reverse[code])
        elif code:
            result.append("?")
    return "".join(result)


def to_fraction_string(n: float, max_denom: int = 100) -> str:
    """Convert float to fraction string."""
    from fractions import Fraction
    frac = Fraction(n).limit_denominator(max_denom)
    return str(frac)


def format_with_commas(n: Union[int, float]) -> str:
    """Format number with thousands separators."""
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.2f}"


def to_percentage(n: float, precision: int = 2) -> str:
    """Convert decimal to percentage string."""
    return f"{n * 100:.{precision}f}%"


class NumberFormatter:
    """Class-based interface for number formatting."""

    @staticmethod
    def binary(n: int, width: Optional[int] = None) -> str:
        return to_binary(n, width)

    @staticmethod
    def octal(n: int) -> str:
        return to_octal(n)

    @staticmethod
    def hexadecimal(n: int, uppercase: bool = False) -> str:
        return to_hex(n, uppercase)

    @staticmethod
    def base(n: int, b: int) -> str:
        return to_base_n(n, b)

    @staticmethod
    def roman(n: int) -> str:
        return to_roman_numeral(n)

    @staticmethod
    def from_roman(s: str) -> int:
        return from_roman_numeral(s)

    @staticmethod
    def scientific(n: float, precision: int = 6) -> str:
        return to_scientific(n, precision)

    @staticmethod
    def engineering(n: float, precision: int = 3) -> str:
        return to_engineering(n, precision)

    @staticmethod
    def words(n: Union[int, float]) -> str:
        return to_words(n)

    @staticmethod
    def morse(s: str) -> str:
        return to_morse_code(s)

    @staticmethod
    def format_sequence(data: List[Union[int, float]], fmt: str = "decimal") -> List[str]:
        """Format a list of numbers."""
        formatters = {
            "binary": lambda n: to_binary(int(n)),
            "octal": lambda n: to_octal(int(n)),
            "hex": lambda n: to_hex(int(n)),
            "roman": lambda n: to_roman_numeral(int(n)) if 1 <= int(n) <= 3999 else str(n),
            "scientific": to_scientific,
            "words": to_words,
            "decimal": str,
            "commas": format_with_commas,
        }
        if fmt not in formatters:
            raise ValueError(f"Unknown format: {fmt}. Available: {list(formatters.keys())}")
        return [formatters[fmt](n) for n in data]
