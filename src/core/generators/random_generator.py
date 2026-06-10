"""
Random Number Generators - Multiple PRNG implementations from scratch.
Includes LCG, Xorshift, PCG, LFSR, Blum-Blum-Shub, Middle-Square algorithms.
"""

import time
import math
from typing import Iterator, List, Optional, Tuple


class LinearCongruentialGenerator:
    """
    Linear Congruential Generator: X_{n+1} = (a * X_n + c) mod m
    Classic PRNG with configurable parameters.
    """

    # Well-known parameter sets
    PARAMS = {
        "numerical_recipes": {"a": 1664525, "c": 1013904223, "m": 2**32},
        "borland": {"a": 22695477, "c": 1, "m": 2**32},
        "ansi_c": {"a": 1103515245, "c": 12345, "m": 2**31},
        "java": {"a": 25214903917, "c": 11, "m": 2**48},
        "glibc": {"a": 1103515245, "c": 12345, "m": 2**31},
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        a: int = 1664525,
        c: int = 1013904223,
        m: int = 2**32,
        preset: Optional[str] = None,
    ):
        if preset and preset in self.PARAMS:
            params = self.PARAMS[preset]
            a, c, m = params["a"], params["c"], params["m"]
        self.a = a
        self.c = c
        self.m = m
        self.state = seed if seed is not None else int(time.time() * 1000) % m
        self._initial_seed = self.state

    def next_int(self) -> int:
        """Generate the next integer."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_float(self) -> float:
        """Generate a float in [0, 1)."""
        return self.next_int() / self.m

    def next_range(self, low: int, high: int) -> int:
        """Generate an integer in [low, high]."""
        return low + self.next_int() % (high - low + 1)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        """Generate a list of integers."""
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_seed

    def period_estimate(self) -> int:
        """Hull-Dobell theorem: full period iff gcd(c, m)=1, etc."""
        from math import gcd
        if gcd(self.c, self.m) == 1:
            return self.m
        return -1  # period unknown without full test

    def __repr__(self) -> str:
        return f"LCG(a={self.a}, c={self.c}, m={self.m}, state={self.state})"


class Xorshift32Generator:
    """
    Xorshift 32-bit PRNG by George Marsaglia (2003).
    Very fast, passes many statistical tests.
    """

    def __init__(self, seed: Optional[int] = None):
        seed = seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
        self.state = seed if seed != 0 else 1
        self._initial_seed = self.state

    def next_int(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def next_float(self) -> float:
        return self.next_int() / 0xFFFFFFFF

    def next_range(self, low: int, high: int) -> int:
        return low + self.next_int() % (high - low + 1)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_seed


class Xorshift64Generator:
    """
    Xorshift 64-bit PRNG. Longer period than 32-bit variant.
    """

    def __init__(self, seed: Optional[int] = None):
        seed = seed if seed is not None else int(time.time() * 1000000)
        self.state = seed if seed != 0 else 1
        self._initial_seed = self.state

    def next_int(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return self.state

    def next_float(self) -> float:
        return self.next_int() / 0xFFFFFFFFFFFFFFFF

    def next_range(self, low: int, high: int) -> int:
        return low + self.next_int() % (high - low + 1)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_seed


class PCGGenerator:
    """
    Permuted Congruential Generator (PCG) by Melissa O'Neill.
    Excellent statistical properties, small state.
    PCG-XSH-RR 64-bit state, 32-bit output.
    """

    MULTIPLIER = 6364136223846793005
    INCREMENT = 1442695040888963407

    def __init__(self, seed: Optional[int] = None, increment: Optional[int] = None):
        self.state = 0
        self.inc = (increment if increment is not None else self.INCREMENT) | 1
        # Initialize
        self._step()
        seed_val = seed if seed is not None else int(time.time() * 1000000)
        self.state += seed_val
        self._step()
        self._initial_state = self.state
        self._initial_inc = self.inc

    def _step(self) -> None:
        self.state = (self.state * self.MULTIPLIER + self.inc) & 0xFFFFFFFFFFFFFFFF

    def next_int(self) -> int:
        old_state = self.state
        self._step()
        # Output function: XSH-RR (xorshift high, random rotate)
        xorshifted = ((old_state >> 18) ^ old_state) >> 27
        rot = old_state >> 59
        result = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
        return result

    def next_float(self) -> float:
        return self.next_int() / 0xFFFFFFFF

    def next_range(self, low: int, high: int) -> int:
        span = high - low + 1
        # Rejection sampling for unbiased results
        threshold = (-span % span) & 0xFFFFFFFF
        while True:
            r = self.next_int()
            if r >= threshold:
                return low + (r % span)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_state
        self.inc = self._initial_inc


class LFSRGenerator:
    """
    Linear Feedback Shift Register (LFSR) PRNG.
    Uses configurable tap positions for the feedback polynomial.
    Supports both Fibonacci and Galois configurations.
    """

    # Maximal-length polynomials for various bit sizes
    DEFAULT_TAPS = {
        8: [8, 6, 5, 4],
        16: [16, 15, 13, 4],
        32: [32, 22, 2, 1],
        64: [64, 63, 61, 60],
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        bits: int = 32,
        taps: Optional[List[int]] = None,
        mode: str = "fibonacci",
    ):
        self.bits = bits
        self.mask = (1 << bits) - 1
        self.taps = taps if taps is not None else self.DEFAULT_TAPS.get(bits, [bits, bits - 1])
        self.mode = mode
        seed_val = seed if seed is not None else int(time.time() * 1000) & self.mask
        self.state = seed_val if seed_val != 0 else 1
        self._initial_seed = self.state

    def _feedback_bit_fibonacci(self) -> int:
        bit = 0
        for tap in self.taps:
            bit ^= (self.state >> (tap - 1)) & 1
        return bit

    def _next_fibonacci(self) -> int:
        bit = self._feedback_bit_fibonacci()
        self.state = ((self.state >> 1) | (bit << (self.bits - 1))) & self.mask
        return self.state

    def _next_galois(self) -> int:
        lsb = self.state & 1
        self.state >>= 1
        if lsb:
            tap_mask = 0
            for tap in self.taps:
                tap_mask |= (1 << (tap - 1))
            self.state ^= tap_mask
        self.state &= self.mask
        return self.state

    def next_int(self) -> int:
        if self.mode == "galois":
            return self._next_galois()
        return self._next_fibonacci()

    def next_bit(self) -> int:
        return self.next_int() & 1

    def next_float(self) -> float:
        return self.next_int() / self.mask

    def next_range(self, low: int, high: int) -> int:
        return low + self.next_int() % (high - low + 1)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_seed

    def period(self) -> int:
        """Theoretical maximum period for maximal-length LFSR."""
        return (1 << self.bits) - 1


class BlumBlumShubGenerator:
    """
    Blum Blum Shub (BBS) cryptographic PRNG.
    Based on the difficulty of factoring M = p*q.
    Secure but slow; best for cryptographic applications.
    """

    def __init__(
        self,
        p: Optional[int] = None,
        q: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # Default Blum primes (p ≡ q ≡ 3 mod 4)
        if p is None:
            p = 11
        if q is None:
            q = 23
        # Verify Blum primes
        if p % 4 != 3 or q % 4 != 3:
            raise ValueError("p and q must be Blum primes (p ≡ q ≡ 3 mod 4)")
        self.M = p * q
        seed_val = seed if seed is not None else int(time.time() * 1000)
        # Seed must be coprime to M
        from math import gcd
        s = seed_val % self.M
        while s < 2 or gcd(s, self.M) != 1:
            s = (s + 1) % self.M
            if s < 2:
                s = 2
        self.state = (s * s) % self.M
        self._initial_state = self.state

    def next_bit(self) -> int:
        self.state = (self.state * self.state) % self.M
        return self.state & 1

    def next_int(self, bits: int = 32) -> int:
        result = 0
        for i in range(bits):
            result = (result << 1) | self.next_bit()
        return result

    def next_float(self) -> float:
        return self.next_int(32) / (2**32)

    def next_range(self, low: int, high: int) -> int:
        span = high - low + 1
        bits_needed = span.bit_length()
        while True:
            r = self.next_int(bits_needed)
            if r < span:
                return low + r

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_state


class MiddleSquareGenerator:
    """
    Middle-Square Method by John von Neumann (1946).
    Historical interest; not suitable for serious use due to short cycles.
    """

    def __init__(self, seed: Optional[int] = None, digits: int = 8):
        self.digits = digits
        if digits % 2 != 0:
            raise ValueError("digits must be even")
        seed_val = seed if seed is not None else int(time.time() * 1000) % (10 ** digits)
        self.state = seed_val
        self._initial_seed = self.state

    def next_int(self) -> int:
        squared = self.state ** 2
        s = str(squared).zfill(self.digits * 2)
        start = self.digits // 2
        middle = s[start: start + self.digits]
        self.state = int(middle)
        return self.state

    def next_float(self) -> float:
        return self.next_int() / (10 ** self.digits)

    def next_range(self, low: int, high: int) -> int:
        return low + self.next_int() % (high - low + 1)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self.state = self._initial_seed


class MersenneTwisterGenerator:
    """
    Mersenne Twister MT19937 PRNG.
    Period of 2^19937-1. Industry standard.
    """

    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: Optional[int] = None):
        self.mt = [0] * self.N
        self.index = self.N + 1
        seed_val = seed if seed is not None else int(time.time() * 1000)
        self._seed(seed_val)
        self._initial_seed = seed_val

    def _seed(self, seed: int) -> None:
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self.N):
            self.mt[i] = (
                1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i
            ) & 0xFFFFFFFF
        self.index = self.N

    def _generate_numbers(self) -> None:
        mag01 = [0, self.MATRIX_A]
        for i in range(self.N):
            y = (self.mt[i] & self.UPPER_MASK) + (self.mt[(i + 1) % self.N] & self.LOWER_MASK)
            self.mt[i] = self.mt[(i + self.M) % self.N] ^ (y >> 1) ^ mag01[y & 1]
        self.index = 0

    def next_int(self) -> int:
        if self.index >= self.N:
            self._generate_numbers()
        y = self.mt[self.index]
        self.index += 1
        # Tempering
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def next_float(self) -> float:
        return self.next_int() / 0xFFFFFFFF

    def next_range(self, low: int, high: int) -> int:
        return low + self.next_int() % (high - low + 1)

    def generate(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]

    def reset(self) -> None:
        self._seed(self._initial_seed)


class RandomGeneratorFactory:
    """Factory class for creating random number generators."""

    ALGORITHMS = {
        "lcg": LinearCongruentialGenerator,
        "xorshift32": Xorshift32Generator,
        "xorshift64": Xorshift64Generator,
        "pcg": PCGGenerator,
        "lfsr": LFSRGenerator,
        "bbs": BlumBlumShubGenerator,
        "middle_square": MiddleSquareGenerator,
        "mersenne": MersenneTwisterGenerator,
    }

    @classmethod
    def create(cls, algorithm: str, seed: Optional[int] = None, **kwargs):
        """Create a generator by algorithm name."""
        algo = algorithm.lower().replace("-", "_")
        if algo not in cls.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Available: {list(cls.ALGORITHMS.keys())}"
            )
        gen_class = cls.ALGORITHMS[algo]
        try:
            return gen_class(seed=seed, **kwargs)
        except TypeError:
            return gen_class(seed=seed)

    @classmethod
    def available_algorithms(cls) -> List[str]:
        return list(cls.ALGORITHMS.keys())

    @classmethod
    def benchmark(cls, count: int = 10000) -> dict:
        """Benchmark all generators."""
        import time
        results = {}
        for name, cls_gen in cls.ALGORITHMS.items():
            try:
                gen = cls_gen(seed=42)
                start = time.perf_counter()
                for _ in range(count):
                    gen.next_int()
                elapsed = time.perf_counter() - start
                results[name] = {
                    "time_seconds": elapsed,
                    "numbers_per_second": count / elapsed,
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        return results
