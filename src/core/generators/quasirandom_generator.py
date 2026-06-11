"""
Quasi-random (low-discrepancy) sequence generators.
Halton, Sobol (1D/2D), Van der Corput, golden-ratio (Kronecker), Latin hypercube.
"""

import math
import random
from typing import List, Tuple


def van_der_corput(n: int, base: int = 2) -> float:
    """The nth Van der Corput value in the given base (n >= 1)."""
    if base < 2:
        raise ValueError("base must be >= 2")
    result, denom = 0.0, 1.0
    while n > 0:
        denom *= base
        n, rem = divmod(n, base)
        result += rem / denom
    return result


def van_der_corput_sequence(count: int, base: int = 2, start: int = 1) -> List[float]:
    return [van_der_corput(i, base) for i in range(start, start + count)]


class HaltonSequence:
    """Multi-dimensional Halton sequence using distinct prime bases."""

    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def __init__(self, dimensions: int = 1, bases: List[int] = None):
        if bases is not None:
            self.bases = bases
        else:
            if dimensions > len(self.PRIMES):
                raise ValueError(f"dimensions must be <= {len(self.PRIMES)}")
            self.bases = self.PRIMES[:dimensions]
        self.index = 0

    def next(self) -> List[float]:
        self.index += 1
        return [van_der_corput(self.index, b) for b in self.bases]

    def generate(self, count: int) -> List[List[float]]:
        return [self.next() for _ in range(count)]

    def generate_1d(self, count: int) -> List[float]:
        return [p[0] for p in self.generate(count)]


class SobolSequence:
    """Sobol sequence for 1 or 2 dimensions using standard direction numbers.

    Dimension 1 is Van der Corput base 2; dimension 2 uses the primitive
    polynomial x + 1 with initial direction number m1 = 1.
    """

    BITS = 30

    def __init__(self, dimensions: int = 1):
        if dimensions not in (1, 2):
            raise ValueError("SobolSequence supports 1 or 2 dimensions")
        self.dimensions = dimensions
        self.index = 0
        self.x = [0] * dimensions  # current integer state (Gray-code update)
        self.v = [self._direction_numbers(d) for d in range(dimensions)]

    def _direction_numbers(self, dim: int) -> List[int]:
        v = [0] * (self.BITS + 1)
        if dim == 0:
            for i in range(1, self.BITS + 1):
                v[i] = 1 << (self.BITS - i)
        else:
            # Dimension 2: polynomial x + 1 (degree 1, a = 0), m = [1]
            m = [0, 1]  # 1-indexed
            s = 1
            for i in range(s + 1, self.BITS + 1):
                new = m[i - s] ^ (m[i - s] << s)
                m.append(new)
            for i in range(1, self.BITS + 1):
                v[i] = m[i] << (self.BITS - i)
        return v

    def next(self) -> List[float]:
        # Index of lowest zero bit of self.index (1-based)
        c = 1
        value = self.index
        while value & 1:
            value >>= 1
            c += 1
        self.index += 1
        out = []
        for d in range(self.dimensions):
            self.x[d] ^= self.v[d][c]
            out.append(self.x[d] / float(1 << self.BITS))
        return out

    def generate(self, count: int) -> List[List[float]]:
        return [self.next() for _ in range(count)]

    def generate_1d(self, count: int) -> List[float]:
        return [p[0] for p in self.generate(count)]


class GoldenRatioSequence:
    """Kronecker (additive recurrence) sequence x_n = frac(x0 + n / phi).

    Uses the golden ratio conjugate for optimal 1D low discrepancy.
    """

    PHI = (1 + math.sqrt(5)) / 2

    def __init__(self, x0: float = 0.0, alpha: float = None):
        self.x0 = x0 % 1.0
        self.alpha = alpha if alpha is not None else 1.0 / self.PHI
        self.n = 0

    def next(self) -> float:
        self.n += 1
        return (self.x0 + self.n * self.alpha) % 1.0

    def generate(self, count: int) -> List[float]:
        return [self.next() for _ in range(count)]


def latin_hypercube(count: int, dimensions: int = 2, seed: int = None) -> List[List[float]]:
    """Latin hypercube sample: count points in [0,1)^dimensions, one per stratum."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    columns = []
    for _ in range(dimensions):
        perm = list(range(count))
        rng.shuffle(perm)
        columns.append([(perm[i] + rng.random()) / count for i in range(count)])
    return [[columns[d][i] for d in range(dimensions)] for i in range(count)]


class QuasiRandomFactory:
    """Factory for quasi-random sequences by name."""

    TYPES = ("halton", "sobol", "van_der_corput", "golden", "lhs")

    @staticmethod
    def available_types() -> List[str]:
        return list(QuasiRandomFactory.TYPES)

    @staticmethod
    def generate(sequence: str, count: int, **kwargs) -> List:
        if count < 0:
            raise ValueError("count must be non-negative")
        if sequence == "halton":
            base = kwargs.get("base")
            if base is not None:
                return HaltonSequence(bases=[int(base)]).generate_1d(count)
            return HaltonSequence(kwargs.get("dimensions", 1)).generate_1d(count)
        if sequence == "sobol":
            dims = kwargs.get("dimensions", 1)
            gen = SobolSequence(dims)
            return gen.generate_1d(count) if dims == 1 else gen.generate(count)
        if sequence == "van_der_corput":
            return van_der_corput_sequence(count, kwargs.get("base", 2))
        if sequence == "golden":
            return GoldenRatioSequence(kwargs.get("x0", 0.0)).generate(count)
        if sequence == "lhs":
            return latin_hypercube(count, kwargs.get("dimensions", 2),
                                   kwargs.get("seed"))
        raise ValueError(f"Unknown quasi-random sequence: {sequence}")
