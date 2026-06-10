"""Mathematical utility functions for number generation and analysis."""

import math
from typing import List, Tuple, Optional, Dict


def gcd(a: int, b: int) -> int:
    """Compute Greatest Common Divisor using Euclidean algorithm."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Compute Least Common Multiple."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (gcd, x, y) such that a*x + b*y = gcd."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def modular_inverse(a: int, m: int) -> int:
    """Compute modular inverse of a mod m. Raises ValueError if inverse doesn't exist."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Modular inverse of {a} mod {m} does not exist (gcd={g})")
    return x % m


def modular_exponentiation(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation using square-and-multiply."""
    if mod == 1:
        return 0
    result = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod
    return result


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """Solve system of congruences x ≡ r_i (mod m_i) using CRT."""
    if len(remainders) != len(moduli):
        raise ValueError("Remainders and moduli must have the same length")

    N = 1
    for m in moduli:
        N *= m

    result = 0
    for r, m in zip(remainders, moduli):
        Ni = N // m
        inv = modular_inverse(Ni, m)
        result += r * Ni * inv

    return result % N


def is_perfect_number(n: int) -> bool:
    """Check if n is a perfect number (equal to sum of proper divisors)."""
    if n < 2:
        return False
    return sum_of_proper_divisors(n) == n


def is_abundant(n: int) -> bool:
    """Check if n is abundant (sum of proper divisors > n)."""
    if n < 1:
        return False
    return sum_of_proper_divisors(n) > n


def is_deficient(n: int) -> bool:
    """Check if n is deficient (sum of proper divisors < n)."""
    if n < 1:
        return True
    return sum_of_proper_divisors(n) < n


def totient(n: int) -> int:
    """Euler's totient function φ(n) - count integers up to n coprime to n."""
    if n <= 0:
        raise ValueError("n must be positive")
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def mobius_function(n: int) -> int:
    """Möbius function μ(n). Returns 0 if n has squared prime factor, else (-1)^k where k is # prime factors."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 1

    factors = prime_factorization(n)
    for p, exp in factors.items():
        if exp > 1:
            return 0
    return (-1) ** len(factors)


def digital_root(n: int) -> int:
    """Compute digital root of n (repeatedly sum digits until single digit)."""
    n = abs(n)
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def digit_sum(n: int) -> int:
    """Sum of digits of n."""
    return sum(int(d) for d in str(abs(n)))


def digit_product(n: int) -> int:
    """Product of digits of n."""
    result = 1
    for d in str(abs(n)):
        result *= int(d)
    return result


def number_of_divisors(n: int) -> int:
    """Count the number of divisors of n."""
    if n <= 0:
        raise ValueError("n must be positive")
    count = 0
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            count += 2
            if i * i == n:
                count -= 1
    return count


def sum_of_divisors(n: int) -> int:
    """Sum of all divisors of n (including 1 and n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    total = 0
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total


def sum_of_proper_divisors(n: int) -> int:
    """Sum of proper divisors of n (all divisors except n itself)."""
    return sum_of_divisors(n) - n


def prime_factorization(n: int) -> Dict[int, int]:
    """Return prime factorization as dict {prime: exponent}."""
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def continued_fraction_expansion(n: int, d: int, max_terms: int = 20) -> List[int]:
    """Compute continued fraction expansion of n/d."""
    result = []
    for _ in range(max_terms):
        result.append(n // d)
        n, d = d, n % d
        if d == 0:
            break
    return result


def convergents(cf: List[int]) -> List[Tuple[int, int]]:
    """Compute convergents from continued fraction coefficients."""
    result = []
    h_prev, h_curr = 1, cf[0]
    k_prev, k_curr = 0, 1
    result.append((h_curr, k_curr))

    for a in cf[1:]:
        h_next = a * h_curr + h_prev
        k_next = a * k_curr + k_prev
        result.append((h_next, k_next))
        h_prev, h_curr = h_curr, h_next
        k_prev, k_curr = k_curr, k_next

    return result


def isqrt(n: int) -> int:
    """Integer square root."""
    return math.isqrt(n)


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Get smallest power of two >= n."""
    if n <= 0:
        return 1
    if is_power_of_two(n):
        return n
    return 1 << n.bit_length()


def floor_log2(n: int) -> int:
    """Floor of log base 2 of n."""
    if n <= 0:
        raise ValueError("n must be positive")
    return n.bit_length() - 1


def hamming_weight(n: int) -> int:
    """Count number of set bits (popcount) in n."""
    return bin(n).count('1')


def reverse_digits(n: int) -> int:
    """Reverse the digits of n."""
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])


def is_palindrome_number(n: int) -> bool:
    """Check if n is a palindromic number."""
    s = str(abs(n))
    return s == s[::-1]


def factorial(n: int) -> int:
    """Compute n! using math.factorial."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return math.factorial(n)


def binomial_coefficient(n: int, k: int) -> int:
    """Compute C(n, k) = n! / (k! * (n-k)!)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def stirling_second_kind(n: int, k: int) -> int:
    """Stirling numbers of the second kind S(n,k)."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    return k * stirling_second_kind(n - 1, k) + stirling_second_kind(n - 1, k - 1)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division returning default if b is zero."""
    return a / b if b != 0 else default
