"""
Numeral system conversions.
Balanced ternary, factorial base (factoradic), Fibonacci base (Zeckendorf),
Gray code, BCD, negabinary, and bijective base-k.
"""

from typing import List


# ---- Balanced ternary (digits -1, 0, 1 written as T, 0, 1) ----

def to_balanced_ternary(n: int) -> str:
    """Convert an integer to balanced ternary using 'T' for -1."""
    if n == 0:
        return "0"
    digits = []
    while n != 0:
        n, rem = divmod(n, 3)
        if rem == 2:
            rem = -1
            n += 1
        digits.append({-1: "T", 0: "0", 1: "1"}[rem])
    return "".join(reversed(digits))


def from_balanced_ternary(s: str) -> int:
    """Convert a balanced ternary string (with 'T' = -1) to an integer."""
    value = 0
    for ch in s:
        if ch == "T":
            d = -1
        elif ch in "01":
            d = int(ch)
        else:
            raise ValueError(f"Invalid balanced ternary digit: {ch}")
        value = value * 3 + d
    return value


# ---- Factorial base (factoradic) ----

def to_factoradic(n: int) -> List[int]:
    """Convert a non-negative integer to factorial base digits (most significant first).

    Digit at place i (from the right, starting at 0) multiplies i! and
    satisfies 0 <= d <= i + 1. Returns [0] for 0.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [0]
    digits = []
    i = 1
    while n > 0:
        n, rem = divmod(n, i)
        digits.append(rem)
        i += 1
    return list(reversed(digits))


def from_factoradic(digits: List[int]) -> int:
    """Convert factorial base digits (most significant first) to an integer."""
    value = 0
    n = len(digits)
    fact = 1
    factorials = [1]
    for i in range(1, n):
        fact *= i
        factorials.append(fact)
    for idx, d in enumerate(digits):
        place = n - 1 - idx  # digit multiplies place!
        if d < 0 or d > place + 1:
            raise ValueError(f"Invalid factoradic digit {d} at place {place}")
        value += d * factorials[place]
    return value


# ---- Fibonacci base (Zeckendorf representation) ----

def to_zeckendorf(n: int) -> str:
    """Zeckendorf representation: binary string over Fibonacci weights
    (F2=1, F3=2, F4=3, F5=5, ...), no two consecutive 1s."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return "0"
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    fibs.pop()  # last fib > n
    bits = []
    for f in reversed(fibs):
        if f <= n:
            bits.append("1")
            n -= f
        else:
            bits.append("0")
    return "".join(bits).lstrip("0") or "0"


def from_zeckendorf(s: str) -> int:
    """Convert a Fibonacci-base binary string back to an integer."""
    if any(c not in "01" for c in s):
        raise ValueError("Zeckendorf string must be binary")
    value = 0
    a, b = 1, 2  # F2, F3
    for ch in reversed(s):
        if ch == "1":
            value += a
        a, b = b, a + b
    return value


def is_valid_zeckendorf(s: str) -> bool:
    """Valid Zeckendorf strings are binary with no two consecutive 1s."""
    return all(c in "01" for c in s) and "11" not in s


# ---- Gray code ----

def to_gray(n: int) -> int:
    """Convert a non-negative integer to its Gray code."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n ^ (n >> 1)


def from_gray(g: int) -> int:
    """Convert a Gray code back to the original integer."""
    if g < 0:
        raise ValueError("g must be non-negative")
    n = g
    mask = g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def gray_sequence(count: int) -> List[int]:
    """First `count` Gray codes: 0, 1, 3, 2, 6, 7, 5, 4, ..."""
    return [to_gray(i) for i in range(count)]


# ---- BCD (Binary-Coded Decimal) ----

def to_bcd(n: int) -> str:
    """Encode a non-negative integer as BCD: 4 bits per decimal digit."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return "".join(format(int(d), "04b") for d in str(n))


def from_bcd(s: str) -> int:
    """Decode a BCD bit string back to an integer."""
    if len(s) % 4 != 0 or not s:
        raise ValueError("BCD string length must be a positive multiple of 4")
    digits = []
    for i in range(0, len(s), 4):
        nibble = s[i:i + 4]
        if any(c not in "01" for c in nibble):
            raise ValueError("BCD string must be binary")
        d = int(nibble, 2)
        if d > 9:
            raise ValueError(f"Invalid BCD nibble: {nibble}")
        digits.append(str(d))
    return int("".join(digits))


# ---- Negabinary (base -2) ----

def to_negabinary(n: int) -> str:
    """Convert any integer to base -2 (digits 0/1)."""
    if n == 0:
        return "0"
    digits = []
    while n != 0:
        n, rem = divmod(n, -2)
        if rem < 0:
            rem += 2
            n += 1
        digits.append(str(rem))
    return "".join(reversed(digits))


def from_negabinary(s: str) -> int:
    """Convert a base -2 string back to an integer."""
    value = 0
    for ch in s:
        if ch not in "01":
            raise ValueError(f"Invalid negabinary digit: {ch}")
        value = value * -2 + int(ch)
    return value


# ---- Bijective base-k (digits 1..k, no zero) ----

def to_bijective(n: int, base: int = 10) -> List[int]:
    """Convert a positive integer to bijective base-k digits (most significant first).

    In bijective base-k, digits run 1..k and every positive integer has a
    unique representation. Returns [] for 0.
    """
    if base < 1:
        raise ValueError("base must be >= 1")
    if n < 0:
        raise ValueError("n must be non-negative")
    digits = []
    while n > 0:
        q, r = divmod(n, base)
        if r == 0:
            r = base
            q -= 1
        digits.append(r)
        n = q
    return list(reversed(digits))


def from_bijective(digits: List[int], base: int = 10) -> int:
    """Convert bijective base-k digits (most significant first) to an integer."""
    value = 0
    for d in digits:
        if d < 1 or d > base:
            raise ValueError(f"Bijective base-{base} digits must be in 1..{base}")
        value = value * base + d
    return value


def to_bijective_26(n: int) -> str:
    """Spreadsheet-style column name (bijective base-26: A=1 ... Z=26)."""
    return "".join(chr(ord("A") + d - 1) for d in to_bijective(n, 26))


def from_bijective_26(s: str) -> int:
    """Convert a spreadsheet column name back to an integer (A=1)."""
    digits = []
    for ch in s.upper():
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid column letter: {ch}")
        digits.append(ord(ch) - ord("A") + 1)
    return from_bijective(digits, 26)


class NumeralSystemConverter:
    """Unified interface for numeral system conversions by name."""

    SYSTEMS = ("balanced_ternary", "factoradic", "zeckendorf", "gray",
               "bcd", "negabinary", "bijective")

    @staticmethod
    def available_systems() -> List[str]:
        return list(NumeralSystemConverter.SYSTEMS)

    @staticmethod
    def encode(n: int, system: str, **kwargs):
        encoders = {
            "balanced_ternary": to_balanced_ternary,
            "factoradic": to_factoradic,
            "zeckendorf": to_zeckendorf,
            "gray": to_gray,
            "bcd": to_bcd,
            "negabinary": to_negabinary,
            "bijective": lambda x: to_bijective(x, kwargs.get("base", 10)),
        }
        if system not in encoders:
            raise ValueError(f"Unknown numeral system: {system}")
        return encoders[system](n)

    @staticmethod
    def decode(value, system: str, **kwargs) -> int:
        decoders = {
            "balanced_ternary": from_balanced_ternary,
            "factoradic": from_factoradic,
            "zeckendorf": from_zeckendorf,
            "gray": from_gray,
            "bcd": from_bcd,
            "negabinary": from_negabinary,
            "bijective": lambda v: from_bijective(v, kwargs.get("base", 10)),
        }
        if system not in decoders:
            raise ValueError(f"Unknown numeral system: {system}")
        return decoders[system](value)
