"""
Digit-sequence generators for mathematical constants.
Pi (spigot), e, sqrt(2) (long-division method), golden ratio,
Champernowne constant, and the Thue-Morse sequence.
"""

from typing import Iterator, List


def pi_digits(count: int) -> List[int]:
    """First `count` decimal digits of pi (3, 1, 4, 1, 5, ...).

    Uses Gibbons' unbounded streaming spigot algorithm (exact integer math).
    """
    if count <= 0:
        return []
    digits = []
    q, r, t, k, n, l = 1, 0, 1, 1, 3, 3
    while len(digits) < count:
        if 4 * q + r - t < n * t:
            digits.append(n)
            q, r, n = 10 * q, 10 * (r - n * t), (10 * (3 * q + r)) // t - 10 * n
        else:
            q, r, t, k, n, l = (q * k, (2 * q + r) * l, t * l, k + 1,
                                (q * (7 * k + 2) + r * l) // (t * l), l + 2)
    return digits


def pi_digits_string(count: int) -> str:
    """Pi as a string like '3.14159...' with `count` digits total."""
    d = pi_digits(count)
    if not d:
        return ""
    if len(d) == 1:
        return str(d[0])
    return f"{d[0]}." + "".join(str(x) for x in d[1:])


def e_digits(count: int) -> List[int]:
    """First `count` decimal digits of e (2, 7, 1, 8, ...).

    Computes e via the factorial series with exact integer scaling.
    """
    if count <= 0:
        return []
    # Scale by 10^(count + guard); sum 1/k! until terms vanish
    guard = 10
    scale = 10 ** (count + guard)
    total = 0
    term = scale
    k = 0
    while term > 0:
        total += term
        k += 1
        term //= k
    s = str(total // 10 ** guard)
    return [int(c) for c in s[:count]]


def isqrt_digits(n: int, count: int) -> List[int]:
    """Decimal digits of sqrt(n) using the long-division (digit-by-digit) method."""
    if count <= 0:
        return []
    if n < 0:
        raise ValueError("n must be non-negative")
    # Split integer part into pairs of digits
    s = str(n)
    if len(s) % 2:
        s = "0" + s
    pairs = [int(s[i:i + 2]) for i in range(0, len(s), 2)]
    digits = []
    remainder = 0
    root = 0
    i = 0
    while len(digits) < count:
        remainder = remainder * 100 + (pairs[i] if i < len(pairs) else 0)
        i += 1
        # Find largest x with (20*root + x) * x <= remainder
        x = 9
        while (20 * root + x) * x > remainder:
            x -= 1
        remainder -= (20 * root + x) * x
        root = root * 10 + x
        digits.append(x)
    # Drop leading zeros of the integer part (keep at least one digit logic simple)
    int_len = len(pairs)
    int_digits = digits[:int_len]
    while len(int_digits) > 1 and int_digits[0] == 0:
        int_digits.pop(0)
    result = int_digits + digits[int_len:]
    return result[:count]


def sqrt2_digits(count: int) -> List[int]:
    """First `count` decimal digits of sqrt(2): 1, 4, 1, 4, 2, ..."""
    return isqrt_digits(2, count)


def golden_ratio_digits(count: int) -> List[int]:
    """First `count` decimal digits of phi = (1 + sqrt(5)) / 2: 1, 6, 1, 8, ...

    Computed exactly: phi * 10^k = (10^k + isqrt(5 * 10^(2k))) / 2.
    """
    if count <= 0:
        return []
    guard = 10
    k = count + guard
    sqrt5_scaled = _isqrt(5 * 10 ** (2 * k))
    phi_scaled = (10 ** k + sqrt5_scaled) // 2
    s = str(phi_scaled)
    return [int(c) for c in s[:count]]


def _isqrt(n: int) -> int:
    """Integer square root (Newton's method, exact)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def champernowne_digits(count: int, base: int = 10) -> List[int]:
    """Digits of the Champernowne constant 0.123456789101112... in the given base."""
    if count <= 0:
        return []
    digits = []
    n = 1
    while len(digits) < count:
        m = n
        rep = []
        while m:
            rep.append(m % base)
            m //= base
        digits.extend(reversed(rep))
        n += 1
    return digits[:count]


def thue_morse(count: int) -> List[int]:
    """First `count` terms of the Thue-Morse sequence: 0,1,1,0,1,0,0,1,..."""
    if count <= 0:
        return []
    return [bin(i).count("1") % 2 for i in range(count)]


def thue_morse_iterator() -> Iterator[int]:
    """Infinite Thue-Morse iterator."""
    i = 0
    while True:
        yield bin(i).count("1") % 2
        i += 1


class DigitsGeneratorFactory:
    """Factory for digit sequences of mathematical constants."""

    CONSTANTS = ("pi", "e", "sqrt2", "phi", "champernowne", "thue_morse")

    @staticmethod
    def available_constants() -> List[str]:
        return list(DigitsGeneratorFactory.CONSTANTS)

    @staticmethod
    def generate(constant: str, count: int, **kwargs) -> List[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        funcs = {
            "pi": pi_digits,
            "e": e_digits,
            "sqrt2": sqrt2_digits,
            "phi": golden_ratio_digits,
            "champernowne": lambda c: champernowne_digits(c, kwargs.get("base", 10)),
            "thue_morse": thue_morse,
        }
        if constant not in funcs:
            raise ValueError(f"Unknown constant: {constant}")
        return funcs[constant](count)
