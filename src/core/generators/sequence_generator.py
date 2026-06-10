"""
Mathematical Sequence Generators.
Arithmetic, geometric, harmonic, special number sequences, and combinatorial sequences.
"""

import math
from functools import lru_cache
from typing import Iterator, List, Optional, Tuple
from fractions import Fraction


class ArithmeticSequence:
    """Arithmetic sequence: a, a+d, a+2d, ..."""

    def __init__(self, start: float = 0, difference: float = 1):
        self.start = start
        self.difference = difference

    def nth(self, n: int) -> float:
        """Return the nth term (0-indexed)."""
        return self.start + n * self.difference

    def generate(self, count: int) -> List[float]:
        return [self.nth(i) for i in range(count)]

    def sum(self, count: int) -> float:
        """Sum of first count terms."""
        return count * (2 * self.start + (count - 1) * self.difference) / 2

    def find_index(self, value: float) -> Optional[int]:
        """Find the index of a value in the sequence."""
        if self.difference == 0:
            return 0 if value == self.start else None
        idx = (value - self.start) / self.difference
        if idx >= 0 and abs(idx - round(idx)) < 1e-9:
            return int(round(idx))
        return None


class GeometricSequence:
    """Geometric sequence: a, ar, ar^2, ..."""

    def __init__(self, start: float = 1, ratio: float = 2):
        self.start = start
        self.ratio = ratio

    def nth(self, n: int) -> float:
        return self.start * (self.ratio ** n)

    def generate(self, count: int) -> List[float]:
        return [self.nth(i) for i in range(count)]

    def sum(self, count: int) -> float:
        """Sum of first count terms."""
        if self.ratio == 1:
            return self.start * count
        return self.start * (1 - self.ratio**count) / (1 - self.ratio)

    def infinite_sum(self) -> Optional[float]:
        """Sum of infinite terms if |r| < 1."""
        if abs(self.ratio) < 1:
            return self.start / (1 - self.ratio)
        return None


class HarmonicSequence:
    """Harmonic sequence: 1, 1/2, 1/3, 1/4, ..."""

    def __init__(self, start: int = 1, scale: float = 1.0):
        self.start = start
        self.scale = scale

    def nth(self, n: int) -> float:
        """Return nth term (1-indexed)."""
        return self.scale / (self.start + n - 1)

    def generate(self, count: int) -> List[float]:
        return [self.nth(i) for i in range(1, count + 1)]

    def partial_sum(self, n: int) -> float:
        """Harmonic number H_n."""
        return self.scale * sum(1.0 / (self.start + i - 1) for i in range(1, n + 1))

    def harmonic_number(n: int) -> float:
        """H_n = sum(1/k for k in 1..n)."""
        return sum(1.0 / k for k in range(1, n + 1))


class TriangularNumbers:
    """Triangular numbers: T_n = n(n+1)/2"""

    @staticmethod
    def nth(n: int) -> int:
        return n * (n + 1) // 2

    @staticmethod
    def generate(count: int, start: int = 1) -> List[int]:
        return [TriangularNumbers.nth(i) for i in range(start, start + count)]

    @staticmethod
    def is_triangular(n: int) -> bool:
        # n is triangular iff 8n+1 is a perfect square
        discriminant = 8 * n + 1
        s = int(math.isqrt(discriminant))
        return s * s == discriminant

    @staticmethod
    def index_of(n: int) -> Optional[int]:
        if not TriangularNumbers.is_triangular(n):
            return None
        return (int(math.isqrt(8 * n + 1)) - 1) // 2


class SquareNumbers:
    @staticmethod
    def nth(n: int) -> int:
        return n * n

    @staticmethod
    def generate(count: int, start: int = 1) -> List[int]:
        return [i * i for i in range(start, start + count)]


class PentagonalNumbers:
    """Pentagonal numbers: P_n = n(3n-1)/2"""

    @staticmethod
    def nth(n: int) -> int:
        return n * (3 * n - 1) // 2

    @staticmethod
    def generate(count: int, start: int = 1) -> List[int]:
        return [PentagonalNumbers.nth(i) for i in range(start, start + count)]

    @staticmethod
    def is_pentagonal(n: int) -> bool:
        # n is pentagonal iff (1 + sqrt(24n+1)) / 6 is a positive integer
        discriminant = 24 * n + 1
        s = int(math.isqrt(discriminant))
        if s * s != discriminant:
            return False
        return (1 + s) % 6 == 0


class HexagonalNumbers:
    """Hexagonal numbers: H_n = n(2n-1)"""

    @staticmethod
    def nth(n: int) -> int:
        return n * (2 * n - 1)

    @staticmethod
    def generate(count: int, start: int = 1) -> List[int]:
        return [HexagonalNumbers.nth(i) for i in range(start, start + count)]


class CatalanNumbers:
    """Catalan numbers: C_n = binomial(2n, n) / (n+1)"""

    @staticmethod
    @lru_cache(maxsize=None)
    def nth(n: int) -> int:
        if n == 0:
            return 1
        return math.comb(2 * n, n) // (n + 1)

    @staticmethod
    def generate(count: int, start: int = 0) -> List[int]:
        return [CatalanNumbers.nth(i) for i in range(start, start + count)]

    @staticmethod
    def recursive(n: int) -> int:
        """Recursive definition: C_n = sum(C_i * C_{n-1-i})"""
        if n == 0:
            return 1
        total = 0
        for i in range(n):
            total += CatalanNumbers.recursive(i) * CatalanNumbers.recursive(n - 1 - i)
        return total


class BellNumbers:
    """Bell numbers: B_n = number of partitions of a set of n elements."""

    @staticmethod
    def nth(n: int) -> int:
        """Using Bell triangle."""
        if n == 0:
            return 1
        # Bell triangle
        row = [1]
        for _ in range(n):
            new_row = [row[-1]]
            for j in range(len(row)):
                new_row.append(new_row[-1] + row[j])
            row = new_row
        return row[0]

    @staticmethod
    def generate(count: int, start: int = 0) -> List[int]:
        return [BellNumbers.nth(i) for i in range(start, start + count)]

    @staticmethod
    def triangle(rows: int) -> List[List[int]]:
        """Generate Bell triangle."""
        triangle = [[1]]
        for i in range(1, rows):
            row = [triangle[i - 1][-1]]
            for j in range(len(triangle[i - 1])):
                row.append(row[-1] + triangle[i - 1][j])
            triangle.append(row)
        return triangle


class EulerNumbers:
    """Euler numbers (alternating sequence)."""

    @staticmethod
    def nth(n: int) -> int:
        """Compute nth Euler number E_n."""
        if n % 2 == 1:
            return 0
        # Use zigzag numbers / secant formula
        # E_n = (-1)^(n/2) * T(n, n) where T is the zigzag triangle
        k = n // 2
        # Build zigzag triangle up to row n
        T = [[0] * (n + 2) for _ in range(n + 2)]
        T[1][1] = 1
        for i in range(2, n + 2):
            if i % 2 == 0:
                for j in range(i - 1, 0, -1):
                    T[i][j] = T[i][j + 1] + T[i - 1][j]
            else:
                for j in range(2, i + 1):
                    T[i][j] = T[i][j - 1] + T[i - 1][j]
        val = T[n + 1][1] if (n + 1) % 2 == 1 else T[n + 1][n + 1]
        return int((-1) ** k) * val

    @staticmethod
    def generate(count: int) -> List[int]:
        return [EulerNumbers.nth(i) for i in range(count)]


class BernoulliNumbers:
    """Bernoulli numbers using Faulhaber's formula approach."""

    @staticmethod
    def nth(n: int) -> Fraction:
        """Compute nth Bernoulli number as exact fraction."""
        if n == 1:
            return Fraction(-1, 2)
        if n % 2 == 1 and n > 1:
            return Fraction(0)
        B = [Fraction(0)] * (n + 1)
        B[0] = Fraction(1)
        for m in range(1, n + 1):
            total = Fraction(0)
            for k in range(m):
                total += math.comb(m + 1, k) * B[k]
            B[m] = -total / (m + 1)
        return B[n]

    @staticmethod
    def generate(count: int) -> List[Fraction]:
        return [BernoulliNumbers.nth(i) for i in range(count)]


class CollatzSequence:
    """Collatz (3n+1) conjecture sequence."""

    @staticmethod
    def sequence(n: int) -> List[int]:
        """Generate Collatz sequence starting from n until reaching 1."""
        if n <= 0:
            raise ValueError("n must be positive")
        seq = [n]
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            seq.append(n)
        return seq

    @staticmethod
    def stopping_time(n: int) -> int:
        """Number of steps to reach 1."""
        return len(CollatzSequence.sequence(n)) - 1

    @staticmethod
    def max_value(n: int) -> int:
        """Maximum value reached in Collatz sequence from n."""
        return max(CollatzSequence.sequence(n))

    @staticmethod
    def generate_stopping_times(count: int, start: int = 1) -> List[int]:
        """Generate stopping times for n = start to start+count-1."""
        return [CollatzSequence.stopping_time(i) for i in range(start, start + count)]


class LookAndSay:
    """Look-and-say sequence: 1, 11, 21, 1211, 111221, ..."""

    @staticmethod
    def next_term(s: str) -> str:
        """Generate next term from current."""
        result = []
        i = 0
        while i < len(s):
            digit = s[i]
            count = 1
            while i + count < len(s) and s[i + count] == digit:
                count += 1
            result.append(str(count) + digit)
            i += count
        return "".join(result)

    @staticmethod
    def generate(count: int, start: str = "1") -> List[str]:
        """Generate count terms of the look-and-say sequence."""
        terms = [start]
        for _ in range(count - 1):
            terms.append(LookAndSay.next_term(terms[-1]))
        return terms

    @staticmethod
    def generate_as_int(count: int) -> List[int]:
        return [int(s) for s in LookAndSay.generate(count)]


class RecamanSequence:
    """Recamán's sequence: starts at 0, each step tries to subtract then adds."""

    @staticmethod
    def generate(count: int) -> List[int]:
        seq = [0]
        seen = {0}
        for n in range(1, count):
            candidate = seq[n - 1] - n
            if candidate > 0 and candidate not in seen:
                seq.append(candidate)
            else:
                seq.append(seq[n - 1] + n)
            seen.add(seq[-1])
        return seq


class PadovanSequence:
    """Padovan sequence: P(n) = P(n-2) + P(n-3), starts 1,1,1."""

    @staticmethod
    def generate(count: int) -> List[int]:
        if count <= 0:
            return []
        seq = [1, 1, 1]
        while len(seq) < count:
            seq.append(seq[-2] + seq[-3])
        return seq[:count]

    @staticmethod
    def nth(n: int) -> int:
        if n < 3:
            return 1
        seq = [1, 1, 1]
        for _ in range(n - 2):
            seq.append(seq[-2] + seq[-3])
        return seq[-1]


class PerrinSequence:
    """Perrin sequence: P(n) = P(n-2) + P(n-3), starts 3,0,2."""

    @staticmethod
    def generate(count: int) -> List[int]:
        if count <= 0:
            return []
        seq = [3, 0, 2]
        while len(seq) < count:
            seq.append(seq[-2] + seq[-3])
        return seq[:count]

    @staticmethod
    def nth(n: int) -> int:
        if n == 0:
            return 3
        if n == 1:
            return 0
        if n == 2:
            return 2
        seq = [3, 0, 2]
        for _ in range(n - 2):
            seq.append(seq[-2] + seq[-3])
        return seq[-1]

    @staticmethod
    def perrin_pseudoprime(n: int) -> bool:
        """Check if n divides P(n) - a necessary but not sufficient condition for primality."""
        return PerrinSequence.nth(n) % n == 0


class PolygonalNumbers:
    """General polygonal numbers."""

    @staticmethod
    def nth(s: int, n: int) -> int:
        """nth s-gonal number."""
        return n * ((s - 2) * n - (s - 4)) // 2

    @staticmethod
    def generate(s: int, count: int, start: int = 1) -> List[int]:
        return [PolygonalNumbers.nth(s, i) for i in range(start, start + count)]

    @staticmethod
    def is_polygonal(s: int, x: int) -> bool:
        """Check if x is an s-gonal number."""
        # x = n*((s-2)n - (s-4))/2  => solve for n
        a = s - 2
        b = -(s - 4)
        c = -2 * x
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False
        sqrt_d = math.isqrt(discriminant)
        if sqrt_d * sqrt_d != discriminant:
            return False
        n = (-b + sqrt_d) / (2 * a)
        return n > 0 and abs(n - round(n)) < 1e-9


class SequenceGeneratorFactory:
    """Factory for creating and generating various sequences."""

    SEQUENCE_TYPES = {
        "arithmetic": lambda **kw: ArithmeticSequence(**kw),
        "geometric": lambda **kw: GeometricSequence(**kw),
        "harmonic": lambda **kw: HarmonicSequence(**kw),
        "triangular": TriangularNumbers,
        "square": SquareNumbers,
        "pentagonal": PentagonalNumbers,
        "hexagonal": HexagonalNumbers,
        "catalan": CatalanNumbers,
        "bell": BellNumbers,
        "euler": EulerNumbers,
        "collatz": CollatzSequence,
        "recaman": RecamanSequence,
        "padovan": PadovanSequence,
        "perrin": PerrinSequence,
    }

    @classmethod
    def generate(cls, seq_type: str, count: int, **kwargs) -> List:
        """Generate a sequence by type name."""
        seq_type = seq_type.lower()
        if seq_type == "arithmetic":
            s = ArithmeticSequence(**{k: v for k, v in kwargs.items() if k in ("start", "difference")})
            return s.generate(count)
        elif seq_type == "geometric":
            s = GeometricSequence(**{k: v for k, v in kwargs.items() if k in ("start", "ratio")})
            return s.generate(count)
        elif seq_type == "harmonic":
            s = HarmonicSequence()
            return s.generate(count)
        elif seq_type == "triangular":
            return TriangularNumbers.generate(count)
        elif seq_type == "square":
            return SquareNumbers.generate(count)
        elif seq_type == "pentagonal":
            return PentagonalNumbers.generate(count)
        elif seq_type == "hexagonal":
            return HexagonalNumbers.generate(count)
        elif seq_type == "catalan":
            return CatalanNumbers.generate(count)
        elif seq_type == "bell":
            return BellNumbers.generate(count)
        elif seq_type == "collatz":
            n = kwargs.get("n", 27)
            return CollatzSequence.sequence(n)[:count]
        elif seq_type == "recaman":
            return RecamanSequence.generate(count)
        elif seq_type == "padovan":
            return PadovanSequence.generate(count)
        elif seq_type == "perrin":
            return PerrinSequence.generate(count)
        elif seq_type == "look_and_say":
            return LookAndSay.generate_as_int(count)
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

    @classmethod
    def available_types(cls) -> List[str]:
        return list(cls.SEQUENCE_TYPES.keys()) + ["look_and_say"]
