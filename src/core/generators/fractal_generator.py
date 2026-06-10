"""
Fractal Number Sequence Generators.
Extracts numeric sequences from Mandelbrot, Julia, and other fractal computations.
"""

import math
from typing import Iterator, List, Optional, Tuple, Complex


class MandelbrotGenerator:
    """
    Generates numeric sequences based on Mandelbrot set computations.
    c = x + yi; z_{n+1} = z_n^2 + c, count iterations to diverge.
    """

    def __init__(self, max_iterations: int = 100, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def iterate_count(self, cx: float, cy: float) -> int:
        """Return iteration count for point (cx, cy) in Mandelbrot set."""
        zx, zy = 0.0, 0.0
        for i in range(self.max_iterations):
            if zx * zx + zy * zy > self.escape_radius ** 2:
                return i
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
        return self.max_iterations

    def smooth_iterate_count(self, cx: float, cy: float) -> float:
        """Smooth iteration count using escape-time algorithm."""
        zx, zy = 0.0, 0.0
        for i in range(self.max_iterations):
            r2 = zx * zx + zy * zy
            if r2 > self.escape_radius ** 2:
                # Smooth coloring
                return i + 1 - math.log(math.log(math.sqrt(r2))) / math.log(2)
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
        return float(self.max_iterations)

    def generate_sequence(self, count: int, x_range: Tuple[float, float] = (-2.5, 1.0),
                          y_range: Tuple[float, float] = (-1.25, 1.25)) -> List[int]:
        """Generate iteration counts for evenly spaced points."""
        x_min, x_max = x_range
        y_min, y_max = y_range
        result = []
        for i in range(count):
            t = i / max(count - 1, 1)
            cx = x_min + t * (x_max - x_min)
            cy = y_min + t * (y_max - y_min)
            result.append(self.iterate_count(cx, cy))
        return result

    def orbit(self, cx: float, cy: float) -> List[Tuple[float, float]]:
        """Return the orbit of point c under Mandelbrot iteration."""
        zx, zy = 0.0, 0.0
        orbit = [(zx, zy)]
        for _ in range(self.max_iterations):
            if zx * zx + zy * zy > self.escape_radius ** 2:
                break
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
            orbit.append((zx, zy))
        return orbit

    def orbit_magnitudes(self, cx: float, cy: float) -> List[float]:
        """Return |z_n| for each iteration in the orbit."""
        return [math.sqrt(x**2 + y**2) for x, y in self.orbit(cx, cy)]

    def interior_distance(self, cx: float, cy: float) -> float:
        """Estimate distance to Mandelbrot set boundary."""
        zx, zy = 0.0, 0.0
        dzx, dzy = 0.0, 0.0
        for _ in range(self.max_iterations):
            dzx, dzy = 2 * (zx * dzx - zy * dzy) + 1, 2 * (zx * dzy + zy * dzx)
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
            if zx * zx + zy * zy > 1e10:
                r = math.sqrt(zx * zx + zy * zy)
                dr = math.sqrt(dzx * dzx + dzy * dzy)
                if dr == 0:
                    return 0.0
                return r * math.log(r) / dr
        return 0.0


class JuliaSetGenerator:
    """
    Generates numeric sequences from Julia set computations.
    Fixed c, varying z: z_{n+1} = z_n^2 + c
    """

    # Famous Julia set parameters
    FAMOUS_C = {
        "dendrite": (0.0, 1.0),
        "san_marco": (-0.75, 0.1),
        "douady_rabbit": (-0.123, 0.745),
        "siegel_disk": (-0.391, -0.587),
        "basilica": (-1.0, 0.0),
        "airplane": (-1.755, 0.0),
    }

    def __init__(
        self,
        cx: float = -0.7,
        cy: float = 0.27015,
        max_iterations: int = 100,
        escape_radius: float = 2.0,
    ):
        self.cx = cx
        self.cy = cy
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "JuliaSetGenerator":
        """Create a Julia generator from a preset."""
        if name not in cls.FAMOUS_C:
            raise ValueError(f"Unknown preset: {name}. Available: {list(cls.FAMOUS_C.keys())}")
        cx, cy = cls.FAMOUS_C[name]
        return cls(cx=cx, cy=cy, **kwargs)

    def iterate_count(self, zx: float, zy: float) -> int:
        """Return iteration count for starting point (zx, zy)."""
        for i in range(self.max_iterations):
            if zx * zx + zy * zy > self.escape_radius ** 2:
                return i
            zx, zy = zx * zx - zy * zy + self.cx, 2 * zx * zy + self.cy
        return self.max_iterations

    def generate_sequence(self, count: int) -> List[int]:
        """Generate Julia iteration counts for points on unit circle."""
        result = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            zx = math.cos(angle)
            zy = math.sin(angle)
            result.append(self.iterate_count(zx, zy))
        return result

    def orbit(self, zx: float, zy: float) -> List[Tuple[float, float]]:
        """Return the orbit of point z under Julia iteration."""
        orbit = [(zx, zy)]
        for _ in range(self.max_iterations):
            if zx * zx + zy * zy > self.escape_radius ** 2:
                break
            zx, zy = zx * zx - zy * zy + self.cx, 2 * zx * zy + self.cy
            orbit.append((zx, zy))
        return orbit

    def filled_julia_grid(self, width: int = 20, height: int = 20) -> List[List[int]]:
        """Return iteration counts on a grid."""
        grid = []
        for j in range(height):
            row = []
            for i in range(width):
                zx = -2 + 4 * i / (width - 1)
                zy = -2 + 4 * j / (height - 1)
                row.append(self.iterate_count(zx, zy))
            grid.append(row)
        return grid


class LSystemFractal:
    """
    L-System based fractal string/number generator.
    Generates sequences from L-system production rules.
    """

    SYSTEMS = {
        "dragon": {
            "axiom": "FX",
            "rules": {"X": "X+YF+", "Y": "-FX-Y"},
            "angle": 90,
        },
        "sierpinski": {
            "axiom": "A",
            "rules": {"A": "B-A-B", "B": "A+B+A"},
            "angle": 60,
        },
        "hilbert": {
            "axiom": "A",
            "rules": {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"},
            "angle": 90,
        },
        "koch": {
            "axiom": "F",
            "rules": {"F": "F+F-F-F+F"},
            "angle": 90,
        },
    }

    def __init__(self, system: str = "koch"):
        if system not in self.SYSTEMS:
            raise ValueError(f"Unknown L-system: {system}")
        config = self.SYSTEMS[system]
        self.axiom = config["axiom"]
        self.rules = config["rules"]
        self.angle = config["angle"]

    def expand(self, generations: int) -> str:
        """Expand L-system for given number of generations."""
        current = self.axiom
        for _ in range(generations):
            current = "".join(self.rules.get(c, c) for c in current)
        return current

    def to_numeric_sequence(self, generations: int) -> List[int]:
        """Convert L-system string to numeric sequence based on F-move distance."""
        s = self.expand(generations)
        # Count F moves between turns
        sequence = []
        count = 0
        for c in s:
            if c == "F":
                count += 1
            elif c in ("+", "-") and count > 0:
                sequence.append(count)
                count = 0
        if count > 0:
            sequence.append(count)
        return sequence

    def character_frequencies(self, generations: int) -> dict:
        """Count character frequencies in expanded L-system."""
        s = self.expand(generations)
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        return freq


class CantorSetGenerator:
    """
    Cantor set sequence generator.
    Generates intervals remaining after successive removals.
    """

    @staticmethod
    def intervals(generations: int) -> List[Tuple[float, float]]:
        """Return intervals remaining after n generations of middle-third removal."""
        intervals = [(0.0, 1.0)]
        for _ in range(generations):
            new_intervals = []
            for lo, hi in intervals:
                third = (hi - lo) / 3
                new_intervals.append((lo, lo + third))
                new_intervals.append((hi - third, hi))
            intervals = new_intervals
        return intervals

    @staticmethod
    def endpoint_sequence(generations: int) -> List[float]:
        """Return sorted list of endpoints of Cantor set intervals."""
        intervals = CantorSetGenerator.intervals(generations)
        endpoints = set()
        for lo, hi in intervals:
            endpoints.add(lo)
            endpoints.add(hi)
        return sorted(endpoints)

    @staticmethod
    def count(generations: int) -> int:
        """Number of intervals after n generations: 2^n."""
        return 2**generations


class FractalGeneratorFactory:
    """Factory for fractal-based number sequence generators."""

    @staticmethod
    def mandelbrot(max_iterations: int = 100) -> MandelbrotGenerator:
        return MandelbrotGenerator(max_iterations=max_iterations)

    @staticmethod
    def julia(cx: float = -0.7, cy: float = 0.27015, max_iterations: int = 100) -> JuliaSetGenerator:
        return JuliaSetGenerator(cx=cx, cy=cy, max_iterations=max_iterations)

    @staticmethod
    def julia_preset(name: str) -> JuliaSetGenerator:
        return JuliaSetGenerator.from_preset(name)

    @staticmethod
    def lsystem(system: str = "koch") -> LSystemFractal:
        return LSystemFractal(system)

    @staticmethod
    def cantor() -> CantorSetGenerator:
        return CantorSetGenerator()
