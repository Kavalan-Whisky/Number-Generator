"""
Number theory special-number generators.
Perfect, amicable, happy, narcissistic, Kaprekar, Harshad, palindromic,
automorphic, vampire, and Armstrong numbers.
"""

from itertools import permutations
from typing import List, Tuple


def sum_proper_divisors(n: int) -> int:
    """Sum of proper divisors of n (excluding n itself)."""
    if n <= 1:
        return 0
    total = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            total += i
            j = n // i
            if j != i:
                total += j
        i += 1
    return total


def is_perfect(n: int) -> bool:
    """A perfect number equals the sum of its proper divisors."""
    return n > 1 and sum_proper_divisors(n) == n


def perfect_numbers(count: int, limit: int = 10_000_000) -> List[int]:
    """First `count` perfect numbers up to limit (via Mersenne primes)."""
    result = []
    p = 2
    while len(result) < count:
        m = (1 << p) - 1
        if _is_prime(m):
            perfect = (1 << (p - 1)) * m
            if perfect > limit:
                break
            result.append(perfect)
        p += 1
        if p > 64:
            break
    return result


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def amicable_pairs(limit: int) -> List[Tuple[int, int]]:
    """Amicable pairs (a, b) with a < b <= limit and sigma(a)-a = b, sigma(b)-b = a."""
    pairs = []
    for a in range(2, limit + 1):
        b = sum_proper_divisors(a)
        if b > a and b <= limit and sum_proper_divisors(b) == a:
            pairs.append((a, b))
    return pairs


def is_happy(n: int) -> bool:
    """A happy number reaches 1 under repeated sum of squared digits."""
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1


def happy_numbers(count: int, start: int = 1) -> List[int]:
    result = []
    n = max(1, start)
    while len(result) < count:
        if is_happy(n):
            result.append(n)
        n += 1
    return result


def is_narcissistic(n: int) -> bool:
    """n equals the sum of its digits each raised to the number of digits."""
    if n < 0:
        return False
    digits = str(n)
    k = len(digits)
    return n == sum(int(d) ** k for d in digits)


def narcissistic_numbers(count: int, start: int = 0) -> List[int]:
    result = []
    n = max(0, start)
    while len(result) < count:
        if is_narcissistic(n):
            result.append(n)
        n += 1
    return result


# Armstrong numbers are the same as narcissistic numbers.
is_armstrong = is_narcissistic
armstrong_numbers = narcissistic_numbers


def is_kaprekar(n: int) -> bool:
    """n^2 can be split into two parts summing to n (right part non-zero)."""
    if n < 1:
        return False
    sq = str(n * n)
    for i in range(1, len(sq) + 1):
        left = sq[:-i] or "0"
        right = sq[-i:]
        if int(right) > 0 and int(left) + int(right) == n:
            return True
    return n == 1


def kaprekar_numbers(count: int, start: int = 1) -> List[int]:
    result = []
    n = max(1, start)
    while len(result) < count:
        if is_kaprekar(n):
            result.append(n)
        n += 1
    return result


def is_harshad(n: int) -> bool:
    """n is divisible by the sum of its digits."""
    if n <= 0:
        return False
    return n % sum(int(d) for d in str(n)) == 0


def harshad_numbers(count: int, start: int = 1) -> List[int]:
    result = []
    n = max(1, start)
    while len(result) < count:
        if is_harshad(n):
            result.append(n)
        n += 1
    return result


def is_palindromic(n: int, base: int = 10) -> bool:
    if n < 0:
        return False
    if base == 10:
        s = str(n)
        return s == s[::-1]
    digits = []
    m = n
    if m == 0:
        return True
    while m:
        digits.append(m % base)
        m //= base
    return digits == digits[::-1]


def palindromic_numbers(count: int, start: int = 0, base: int = 10) -> List[int]:
    result = []
    n = max(0, start)
    while len(result) < count:
        if is_palindromic(n, base):
            result.append(n)
        n += 1
    return result


def is_automorphic(n: int) -> bool:
    """n^2 ends with the digits of n (e.g., 25^2 = 625)."""
    if n < 0:
        return False
    return str(n * n).endswith(str(n))


def automorphic_numbers(count: int, start: int = 0) -> List[int]:
    result = []
    n = max(0, start)
    while len(result) < count:
        if is_automorphic(n):
            result.append(n)
        n += 1
    return result


def is_vampire(n: int) -> bool:
    """True vampire number: even digit count, product of two fangs of half
    length each, with the same multiset of digits; fangs cannot both end in 0."""
    s = str(n)
    if len(s) % 2 != 0:
        return False
    half = len(s) // 2
    sorted_digits = sorted(s)
    seen = set()
    for perm in permutations(s):
        a_str = "".join(perm[:half])
        b_str = "".join(perm[half:])
        a, b = int(a_str), int(b_str)
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if len(str(a)) != half or len(str(b)) != half:
            continue
        if a % 10 == 0 and b % 10 == 0:
            continue
        if a * b == n and sorted(str(a) + str(b)) == sorted_digits:
            return True
    return False


def vampire_numbers(count: int, start: int = 1000) -> List[int]:
    """Find vampire numbers by enumerating fang products (efficient)."""
    result = set()
    digits = 4
    while len(result) < count and digits <= 8:
        half = digits // 2
        lo, hi = 10 ** (half - 1), 10 ** half
        n_lo, n_hi = 10 ** (digits - 1), 10 ** digits
        for a in range(lo, hi):
            for b in range(a, hi):
                p = a * b
                if p < max(n_lo, start) or p >= n_hi:
                    continue
                if a % 10 == 0 and b % 10 == 0:
                    continue
                if sorted(str(p)) == sorted(str(a) + str(b)):
                    result.add(p)
        digits += 2
    return sorted(result)[:count]


class NumberTheoryFactory:
    """Factory for special-number generation by name."""

    TYPES = ("perfect", "happy", "narcissistic", "armstrong", "kaprekar",
             "harshad", "palindromic", "automorphic", "vampire")

    @staticmethod
    def available_types() -> List[str]:
        return list(NumberTheoryFactory.TYPES)

    @staticmethod
    def generate(number_type: str, count: int, start: int = None) -> List[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        funcs = {
            "perfect": lambda c, s: perfect_numbers(c),
            "happy": lambda c, s: happy_numbers(c, s if s is not None else 1),
            "narcissistic": lambda c, s: narcissistic_numbers(c, s if s is not None else 0),
            "armstrong": lambda c, s: armstrong_numbers(c, s if s is not None else 0),
            "kaprekar": lambda c, s: kaprekar_numbers(c, s if s is not None else 1),
            "harshad": lambda c, s: harshad_numbers(c, s if s is not None else 1),
            "palindromic": lambda c, s: palindromic_numbers(c, s if s is not None else 0),
            "automorphic": lambda c, s: automorphic_numbers(c, s if s is not None else 0),
            "vampire": lambda c, s: vampire_numbers(c, s if s is not None else 1000),
        }
        if number_type not in funcs:
            raise ValueError(f"Unknown number type: {number_type}")
        return funcs[number_type](count, start)
