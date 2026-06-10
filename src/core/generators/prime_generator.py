"""
Prime Number Generators - Multiple algorithms for prime generation and testing.
Includes Sieve of Eratosthenes, Miller-Rabin, Segmented Sieve, etc.
"""

import math
import random
from typing import Iterator, List, Optional, Tuple, Generator


class SieveOfEratosthenes:
    """Classic Sieve of Eratosthenes for finding all primes up to n."""

    def __init__(self, limit: int = 1000):
        self.limit = limit
        self._primes: Optional[List[int]] = None

    def _compute(self) -> None:
        """Compute primes using the sieve."""
        sieve = bytearray([1]) * (self.limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(math.isqrt(self.limit)) + 1):
            if sieve[i]:
                sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
        self._primes = [i for i, v in enumerate(sieve) if v]

    @property
    def primes(self) -> List[int]:
        if self._primes is None:
            self._compute()
        return self._primes

    def get_primes(self, count: Optional[int] = None) -> List[int]:
        """Return all primes, optionally limiting count."""
        p = self.primes
        return p[:count] if count else p

    def is_prime(self, n: int) -> bool:
        if n > self.limit:
            return miller_rabin_is_prime(n)
        return n in set(self.primes)

    def nth_prime(self, n: int) -> int:
        """Return the nth prime (1-indexed)."""
        primes = self.primes
        if n > len(primes):
            raise ValueError(f"nth prime {n} exceeds computed range (limit={self.limit})")
        return primes[n - 1]


class SegmentedSieve:
    """
    Segmented Sieve of Eratosthenes.
    Memory-efficient for generating primes in large ranges.
    """

    def __init__(self, segment_size: int = 32768):
        self.segment_size = segment_size

    def primes_up_to(self, n: int) -> List[int]:
        """Return all primes <= n using segmented sieve."""
        if n < 2:
            return []
        sqrt_n = int(math.isqrt(n))
        # Base sieve up to sqrt(n)
        small_primes = SieveOfEratosthenes(sqrt_n).primes
        primes = list(small_primes)

        # Process segments
        low = sqrt_n + 1
        while low <= n:
            high = min(low + self.segment_size - 1, n)
            size = high - low + 1
            sieve = bytearray([1]) * size
            for p in small_primes:
                start = ((low + p - 1) // p) * p
                if start == p:
                    start += p
                for j in range(start - low, size, p):
                    sieve[j] = 0
            for i in range(size):
                if sieve[i]:
                    primes.append(low + i)
            low += self.segment_size

        return primes

    def primes_in_range(self, low: int, high: int) -> List[int]:
        """Return all primes in [low, high]."""
        all_primes = self.primes_up_to(high)
        return [p for p in all_primes if p >= low]

    def generate(self, count: int, start: int = 2) -> List[int]:
        """Generate count primes starting from start."""
        # Estimate upper bound using prime number theorem
        if count < 6:
            limit = 15
        else:
            limit = max(start + count * 20, int(count * (math.log(count) + math.log(math.log(count + 3)) + 3)))
        while True:
            primes = [p for p in self.primes_up_to(limit) if p >= start]
            if len(primes) >= count:
                return primes[:count]
            limit *= 2


def miller_rabin_is_prime(n: int, rounds: int = 20) -> bool:
    """
    Miller-Rabin probabilistic primality test.
    With 20 rounds, error probability < 4^(-20) ≈ 10^-12.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Deterministic witnesses for n < 3,215,031,751
    small_witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Use deterministic witnesses for small n
    if n < 3_215_031_751:
        witnesses = [2, 3, 5, 7]
    elif n < 3_317_044_064_679_887_385_961_981:
        witnesses = small_witnesses
    else:
        witnesses = [random.randrange(2, n - 1) for _ in range(rounds)]

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime(n: int) -> bool:
    """Fast primality check combining trial division and Miller-Rabin."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Trial division for small factors
    i = 5
    while i * i <= n and i < 1000:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    if i * i <= n:
        return miller_rabin_is_prime(n)
    return True


def next_prime(n: int) -> int:
    """Find the next prime greater than n."""
    candidate = n + 1 if n % 2 == 0 else n + 2
    if candidate == 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not is_prime(candidate):
        candidate += 2
    return candidate


def previous_prime(n: int) -> Optional[int]:
    """Find the largest prime less than n."""
    if n <= 2:
        return None
    if n == 3:
        return 2
    candidate = n - 1 if n % 2 == 0 else n - 2
    while candidate >= 2:
        if is_prime(candidate):
            return candidate
        candidate -= 2
    return None


def twin_primes(limit: int) -> List[Tuple[int, int]]:
    """Find all twin prime pairs (p, p+2) where both are prime and <= limit."""
    primes = set(SieveOfEratosthenes(limit).primes)
    return [(p, p + 2) for p in primes if p + 2 in primes and p + 2 <= limit]


def prime_gaps(count: int, start: int = 2) -> List[Tuple[int, int, int]]:
    """
    Return list of (p1, p2, gap) for consecutive prime pairs.
    """
    sieve = SegmentedSieve()
    primes = sieve.generate(count + 1, start)
    return [(primes[i], primes[i + 1], primes[i + 1] - primes[i]) for i in range(len(primes) - 1)]


def prime_counting_function(n: int) -> int:
    """pi(n): count of primes <= n."""
    return len(SieveOfEratosthenes(n).primes)


def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """Return prime factorization as list of (prime, exponent) pairs."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                exp += 1
                n //= d
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def mersenne_prime_check(p: int) -> bool:
    """Check if 2^p - 1 is a Mersenne prime using Lucas-Lehmer test."""
    if not is_prime(p):
        return False
    if p == 2:
        return True
    M = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % M
    return s == 0


def sophie_germain_primes(limit: int) -> List[int]:
    """Find Sophie Germain primes p where both p and 2p+1 are prime."""
    sieve = set(SieveOfEratosthenes(limit * 2 + 2).primes)
    return [p for p in sieve if p <= limit and (2 * p + 1) in sieve]


class SundaramSieve:
    """
    Sieve of Sundaram: alternative sieve algorithm.
    Generates odd primes by eliminating composites.
    """

    def primes_up_to(self, n: int) -> List[int]:
        """Return all primes <= n."""
        if n < 2:
            return []
        k = (n - 2) // 2
        sieve = bytearray([1]) * (k + 1)
        i = 1
        while i <= k:
            j = i
            while i + j + 2 * i * j <= k:
                sieve[i + j + 2 * i * j] = 0
                j += 1
            i += 1
        primes = [2] if n >= 2 else []
        for i in range(1, k + 1):
            if sieve[i]:
                p = 2 * i + 1
                if p <= n:
                    primes.append(p)
        return primes

    def generate(self, count: int, start: int = 2) -> List[int]:
        limit = max(start + count * 15, 20)
        while True:
            primes = [p for p in self.primes_up_to(limit) if p >= start]
            if len(primes) >= count:
                return primes[:count]
            limit *= 2


class PrimeGeneratorFactory:
    """Factory for creating prime generators and running primality tests."""

    @staticmethod
    def sieve(limit: int = 1000) -> SieveOfEratosthenes:
        return SieveOfEratosthenes(limit)

    @staticmethod
    def segmented_sieve(segment_size: int = 32768) -> SegmentedSieve:
        return SegmentedSieve(segment_size)

    @staticmethod
    def sundaram_sieve() -> SundaramSieve:
        return SundaramSieve()

    @staticmethod
    def generate_primes(count: int, start: int = 2, algorithm: str = "sieve") -> List[int]:
        """Generate count primes starting from start."""
        if algorithm == "sieve":
            return SegmentedSieve().generate(count, start)
        elif algorithm == "sundaram":
            return SundaramSieve().generate(count, start)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def is_prime(n: int, method: str = "miller_rabin") -> bool:
        if method == "miller_rabin":
            return miller_rabin_is_prime(n)
        elif method == "trial":
            return is_prime(n)
        raise ValueError(f"Unknown method: {method}")
