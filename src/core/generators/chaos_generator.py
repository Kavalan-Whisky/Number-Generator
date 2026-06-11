"""
Chaos-based number generators.
Logistic map, Tent map, Henon map, Lorenz system, Arnold's cat map, and a chaos-based PRNG.
"""

import math
from typing import Iterator, List, Tuple


class LogisticMap:
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n).

    Chaotic for most r in (3.57, 4].
    """

    def __init__(self, r: float = 3.99, x0: float = 0.5):
        if not (0 < x0 < 1):
            raise ValueError("x0 must be in (0, 1)")
        if not (0 < r <= 4):
            raise ValueError("r must be in (0, 4]")
        self.r = r
        self.x = x0

    def step(self) -> float:
        self.x = self.r * self.x * (1.0 - self.x)
        return self.x

    def iterate(self) -> Iterator[float]:
        while True:
            yield self.step()

    def generate(self, count: int, discard: int = 0) -> List[float]:
        """Generate count values, optionally discarding a transient."""
        for _ in range(discard):
            self.step()
        return [self.step() for _ in range(count)]


class TentMap:
    """Tent map: x_{n+1} = mu * min(x_n, 1 - x_n)."""

    def __init__(self, mu: float = 1.999, x0: float = 0.4):
        if not (0 < x0 < 1):
            raise ValueError("x0 must be in (0, 1)")
        if not (0 < mu <= 2):
            raise ValueError("mu must be in (0, 2]")
        self.mu = mu
        self.x = x0

    def step(self) -> float:
        self.x = self.mu * min(self.x, 1.0 - self.x)
        return self.x

    def generate(self, count: int, discard: int = 0) -> List[float]:
        for _ in range(discard):
            self.step()
        return [self.step() for _ in range(count)]


class HenonMap:
    """Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n.

    Classic chaotic attractor for a=1.4, b=0.3.
    """

    def __init__(self, a: float = 1.4, b: float = 0.3,
                 x0: float = 0.1, y0: float = 0.1):
        self.a = a
        self.b = b
        self.x = x0
        self.y = y0

    def step(self) -> Tuple[float, float]:
        x_new = 1.0 - self.a * self.x * self.x + self.y
        y_new = self.b * self.x
        self.x, self.y = x_new, y_new
        return self.x, self.y

    def generate(self, count: int, discard: int = 0) -> List[Tuple[float, float]]:
        for _ in range(discard):
            self.step()
        return [self.step() for _ in range(count)]

    def generate_x(self, count: int, discard: int = 0) -> List[float]:
        return [p[0] for p in self.generate(count, discard)]


class LorenzSystem:
    """Lorenz system integrated with forward Euler.

    dx/dt = sigma (y - x); dy/dt = x (rho - z) - y; dz/dt = x y - beta z.
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0 / 3.0,
                 x0: float = 1.0, y0: float = 1.0, z0: float = 1.0,
                 dt: float = 0.01):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x, self.y, self.z = x0, y0, z0
        self.dt = dt

    def step(self) -> Tuple[float, float, float]:
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        return self.x, self.y, self.z

    def generate(self, count: int, discard: int = 0) -> List[Tuple[float, float, float]]:
        for _ in range(discard):
            self.step()
        return [self.step() for _ in range(count)]

    def generate_x(self, count: int, discard: int = 0) -> List[float]:
        return [p[0] for p in self.generate(count, discard)]


class ArnoldCatMap:
    """Arnold's cat map on an N x N grid.

    (x, y) -> (2x + y mod N, x + y mod N). Periodic for any integer point.
    """

    def __init__(self, n: int = 256, x0: int = 1, y0: int = 1):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.x = x0 % n
        self.y = y0 % n

    def step(self) -> Tuple[int, int]:
        x_new = (2 * self.x + self.y) % self.n
        y_new = (self.x + self.y) % self.n
        self.x, self.y = x_new, y_new
        return self.x, self.y

    def generate(self, count: int) -> List[Tuple[int, int]]:
        return [self.step() for _ in range(count)]

    def period(self, max_iter: int = 10_000_000) -> int:
        """Return the period of the current starting point."""
        start = (self.x, self.y)
        x, y = start
        for i in range(1, max_iter + 1):
            x, y = (2 * x + y) % self.n, (x + y) % self.n
            if (x, y) == start:
                return i
        raise RuntimeError("period not found within max_iter")


class ChaosPRNG:
    """Pseudo-random number generator built on the logistic map.

    Extracts bits from the chaotic orbit; not cryptographically secure.
    """

    def __init__(self, seed: int = 12345, r: float = 3.9999):
        # Derive x0 in (0, 1) from the integer seed
        x0 = ((seed * 2654435761) % 1000003 + 1) / 1000005.0
        self.map = LogisticMap(r=r, x0=x0)
        # Burn in transient
        for _ in range(100):
            self.map.step()

    def random(self) -> float:
        """Uniform-ish float in [0, 1) via bit extraction."""
        bits = 0
        for _ in range(32):
            bits = (bits << 1) | (1 if self.map.step() > 0.5 else 0)
        return bits / 4294967296.0

    def randint(self, low: int, high: int) -> int:
        """Random integer in [low, high]."""
        if low > high:
            raise ValueError("low must be <= high")
        return low + int(self.random() * (high - low + 1))

    def generate(self, count: int, min_val: int = 0, max_val: int = 1000) -> List[int]:
        return [self.randint(min_val, max_val) for _ in range(count)]

    def generate_floats(self, count: int) -> List[float]:
        return [self.random() for _ in range(count)]


class ChaosGeneratorFactory:
    """Factory for chaos generators."""

    MAPS = ("logistic", "tent", "henon", "lorenz", "arnold")

    @staticmethod
    def available_maps() -> List[str]:
        return list(ChaosGeneratorFactory.MAPS)

    @staticmethod
    def generate(map_name: str, count: int, discard: int = 100, **kwargs) -> List[float]:
        """Generate a sequence of floats from the named chaotic map."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if map_name == "logistic":
            return LogisticMap(**kwargs).generate(count, discard)
        if map_name == "tent":
            return TentMap(**kwargs).generate(count, discard)
        if map_name == "henon":
            return HenonMap(**kwargs).generate_x(count, discard)
        if map_name == "lorenz":
            return LorenzSystem(**kwargs).generate_x(count, discard)
        if map_name == "arnold":
            cat = ArnoldCatMap(**kwargs)
            return [float(p[0]) for p in cat.generate(count)]
        raise ValueError(f"Unknown chaotic map: {map_name}")
