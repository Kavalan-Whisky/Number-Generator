"""
Fibonacci and Related Sequence Generators.
Includes classic, matrix exponentiation, Lucas sequence, and generalizations.
"""

import math
from functools import lru_cache
from typing import Iterator, List, Optional, Tuple


def _matrix_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """2x2 matrix multiplication."""
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
    ]


def _matrix_power(M: List[List[int]], n: int) -> List[List[int]]:
    """Fast matrix exponentiation: M^n in O(log n)."""
    if n == 1:
        return M
    if n % 2 == 0:
        half = _matrix_power(M, n // 2)
        return _matrix_multiply(half, half)
    return _matrix_multiply(M, _matrix_power(M, n - 1))


class FibonacciGenerator:
    """Classic Fibonacci sequence generator with multiple implementations."""

    @staticmethod
    @lru_cache(maxsize=None)
    def recursive(n: int) -> int:
        """Memoized recursive Fibonacci."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n <= 1:
            return n
        return FibonacciGenerator.recursive(n - 1) + FibonacciGenerator.recursive(n - 2)

    @staticmethod
    def iterative(n: int) -> int:
        """Iterative Fibonacci in O(n)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    @staticmethod
    def matrix_exponentiation(n: int) -> int:
        """Fast Fibonacci using matrix exponentiation: O(log n)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return 0
        M = [[1, 1], [1, 0]]
        result = _matrix_power(M, n)
        return result[0][1]

    @staticmethod
    def closed_form(n: int) -> int:
        """Binet's formula (approximate for large n due to float precision)."""
        phi = (1 + math.sqrt(5)) / 2
        return round(phi**n / math.sqrt(5))

    def generate(self, count: int, method: str = "iterative", start_index: int = 0) -> List[int]:
        """Generate count Fibonacci numbers starting from index start_index."""
        methods = {
            "recursive": self.recursive,
            "iterative": self.iterative,
            "matrix": self.matrix_exponentiation,
            "closed_form": self.closed_form,
        }
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Use: {list(methods.keys())}")
        func = methods[method]
        return [func(i) for i in range(start_index, start_index + count)]

    def sequence_up_to(self, limit: int) -> List[int]:
        """Return all Fibonacci numbers <= limit."""
        result = []
        a, b = 0, 1
        while a <= limit:
            result.append(a)
            a, b = b, a + b
        return result

    def is_fibonacci(self, n: int) -> bool:
        """Check if n is a Fibonacci number."""
        if n < 0:
            return False
        # n is Fibonacci iff 5n^2 + 4 or 5n^2 - 4 is a perfect square
        def is_perfect_square(x: int) -> bool:
            s = int(math.isqrt(x))
            return s * s == x

        return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)

    def index_of(self, n: int) -> Optional[int]:
        """Find the index of n in the Fibonacci sequence, or None."""
        a, b, idx = 0, 1, 0
        while a < n:
            a, b, idx = b, a + b, idx + 1
        return idx if a == n else None


class LucasSequence:
    """
    Lucas sequence and related sequences.
    Lucas: 2, 1, 3, 4, 7, 11, 18, 29, ...
    """

    @staticmethod
    def lucas(n: int) -> int:
        """Return the nth Lucas number."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return 2
        if n == 1:
            return 1
        a, b = 2, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    @staticmethod
    def generate(count: int, start_index: int = 0) -> List[int]:
        """Generate count Lucas numbers."""
        return [LucasSequence.lucas(i) for i in range(start_index, start_index + count)]

    @staticmethod
    def lucas_u(p: int, q: int, n: int) -> int:
        """General Lucas sequence U_n(P, Q)."""
        if n == 0:
            return 0
        if n == 1:
            return 1
        u_prev, u_curr = 0, 1
        for _ in range(n - 1):
            u_prev, u_curr = u_curr, p * u_curr - q * u_prev
        return u_curr

    @staticmethod
    def lucas_v(p: int, q: int, n: int) -> int:
        """General Lucas sequence V_n(P, Q)."""
        if n == 0:
            return 2
        if n == 1:
            return p
        v_prev, v_curr = 2, p
        for _ in range(n - 1):
            v_prev, v_curr = v_curr, p * v_curr - q * v_prev
        return v_curr


class PisanoPeriod:
    """
    Pisano period: period of Fibonacci numbers mod m.
    """

    @staticmethod
    def compute(m: int) -> int:
        """Compute the Pisano period pi(m)."""
        if m == 1:
            return 1
        prev, curr = 0, 1
        for i in range(1, m * m * 6 + 1):
            prev, curr = curr, (prev + curr) % m
            if prev == 0 and curr == 1:
                return i
        raise ValueError(f"Pisano period not found for m={m}")

    @staticmethod
    def fibonacci_mod(n: int, m: int) -> int:
        """Efficiently compute F(n) mod m using Pisano period."""
        period = PisanoPeriod.compute(m)
        return FibonacciGenerator.iterative(n % period) % m

    @staticmethod
    def generate_sequence_mod(count: int, m: int) -> List[int]:
        """Generate Fibonacci sequence mod m."""
        result = []
        a, b = 0, 1
        for _ in range(count):
            result.append(a % m)
            a, b = b, (a + b) % m
        return result


class GeneralizedFibonacci:
    """
    Generalized Fibonacci sequences: Tribonacci, Tetranacci, etc.
    Also supports custom initial values and recurrence relations.
    """

    def __init__(self, initial: List[int], coefficients: Optional[List[int]] = None):
        """
        Initialize with starting values and linear recurrence coefficients.
        For standard Fibonacci: initial=[0,1], coefficients=[1,1]
        For Tribonacci: initial=[0,0,1], coefficients=[1,1,1]
        """
        self.initial = list(initial)
        self.order = len(initial)
        if coefficients is None:
            self.coefficients = [1] * self.order
        else:
            if len(coefficients) != self.order:
                raise ValueError("coefficients must match initial values length")
            self.coefficients = coefficients

    def nth(self, n: int) -> int:
        """Return the nth term of the sequence."""
        if n < len(self.initial):
            return self.initial[n]
        state = list(self.initial)
        for _ in range(n - len(self.initial) + 1):
            next_val = sum(c * v for c, v in zip(self.coefficients, reversed(state)))
            state.append(next_val)
            state.pop(0)
        return state[-1]

    def generate(self, count: int, start_index: int = 0) -> List[int]:
        """Generate count terms starting at index start_index."""
        # Efficient sequential generation
        state = list(self.initial)
        if count <= len(self.initial) and start_index == 0:
            return state[:count]

        # Generate enough values
        result_list = list(self.initial)
        while len(result_list) < start_index + count:
            next_val = sum(c * v for c, v in zip(self.coefficients, result_list[-self.order:]))
            result_list.append(next_val)

        return result_list[start_index: start_index + count]

    @classmethod
    def tribonacci(cls) -> "GeneralizedFibonacci":
        return cls(initial=[0, 0, 1], coefficients=[1, 1, 1])

    @classmethod
    def tetranacci(cls) -> "GeneralizedFibonacci":
        return cls(initial=[0, 0, 0, 1], coefficients=[1, 1, 1, 1])

    @classmethod
    def pentanacci(cls) -> "GeneralizedFibonacci":
        return cls(initial=[0, 0, 0, 0, 1], coefficients=[1, 1, 1, 1, 1])

    @classmethod
    def padovan(cls) -> "GeneralizedFibonacci":
        """Padovan sequence: P(n) = P(n-2) + P(n-3)"""
        # Special case with non-standard coefficients
        return cls(initial=[1, 1, 1], coefficients=[0, 1, 1])

    @classmethod
    def perrin(cls) -> "GeneralizedFibonacci":
        """Perrin sequence: a(n) = a(n-2) + a(n-3)"""
        return cls(initial=[3, 0, 2], coefficients=[0, 1, 1])


class FibonacciVariants:
    """Additional Fibonacci-related sequences."""

    @staticmethod
    def negafibonacci(n: int) -> int:
        """Fibonacci for negative indices: F(-n) = (-1)^(n+1) * F(n)."""
        if n >= 0:
            return FibonacciGenerator.iterative(n)
        n_abs = -n
        return ((-1) ** (n_abs + 1)) * FibonacciGenerator.iterative(n_abs)

    @staticmethod
    def fibonorial(n: int) -> int:
        """Product F(1) * F(2) * ... * F(n)."""
        result = 1
        a, b = 0, 1
        for i in range(1, n + 1):
            if i == 1:
                pass
            else:
                result *= a
            a, b = b, a + b
        # Recompute properly
        result = 1
        gen = FibonacciGenerator()
        for i in range(1, n + 1):
            f = gen.iterative(i)
            result *= f
        return result

    @staticmethod
    def fibonacci_word(n: int) -> str:
        """nth Fibonacci word: string formed by concatenation."""
        if n == 1:
            return "1"
        if n == 2:
            return "0"
        a, b = "1", "0"
        for _ in range(n - 2):
            a, b = b, b + a
        return b

    @staticmethod
    def zeckendorf_representation(n: int) -> List[int]:
        """
        Zeckendorf's theorem: every positive integer has a unique representation
        as sum of non-consecutive Fibonacci numbers.
        Returns the Fibonacci indices used.
        """
        if n <= 0:
            return []
        gen = FibonacciGenerator()
        fibs = gen.sequence_up_to(n)
        # Remove 0
        fibs = [f for f in fibs if f > 0]
        indices = []
        remaining = n
        for f in reversed(fibs):
            if f <= remaining:
                remaining -= f
                indices.append(fibs.index(f) + 1)
            if remaining == 0:
                break
        return sorted(indices)
