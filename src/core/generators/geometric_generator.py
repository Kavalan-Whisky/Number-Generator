"""
Geometric and figurate number generators.

Includes polygonal numbers, polyhedral numbers, centered polygonal numbers,
gnomonic numbers, star numbers, and 3-D figurate sequences.
"""

import math
from typing import List, Iterator


# ---------------------------------------------------------------------------
# Polygonal numbers
# ---------------------------------------------------------------------------

def triangular(n: int) -> int:
    return n * (n + 1) // 2


def square_num(n: int) -> int:
    return n * n


def pentagonal(n: int) -> int:
    return n * (3 * n - 1) // 2


def hexagonal(n: int) -> int:
    return n * (2 * n - 1)


def heptagonal(n: int) -> int:
    return n * (5 * n - 3) // 2


def octagonal(n: int) -> int:
    return n * (3 * n - 2)


def nonagonal(n: int) -> int:
    return n * (7 * n - 5) // 2


def decagonal(n: int) -> int:
    return n * (4 * n - 3)


def polygonal(s: int, n: int) -> int:
    """s-gonal number for index n (1-based)."""
    return n * ((s - 2) * n - (s - 4)) // 2


def polygonal_sequence(s: int, count: int) -> List[int]:
    return [polygonal(s, n) for n in range(1, count + 1)]


# ---------------------------------------------------------------------------
# Centered polygonal numbers
# ---------------------------------------------------------------------------

def centered_triangular(n: int) -> int:
    return (3 * n * n - 3 * n + 2) // 2


def centered_square(n: int) -> int:
    return 2 * n * n - 2 * n + 1


def centered_pentagonal(n: int) -> int:
    return (5 * n * n - 5 * n + 2) // 2


def centered_hexagonal(n: int) -> int:
    return 3 * n * n - 3 * n + 1


def centered_heptagonal(n: int) -> int:
    return (7 * n * n - 7 * n + 2) // 2


def centered_octagonal(n: int) -> int:
    return 4 * n * n - 4 * n + 1


def centered_polygonal(s: int, n: int) -> int:
    """Centered s-gonal number at index n (0-based: n=0 → 1)."""
    return s * n * (n - 1) // 2 + 1 if n > 0 else 1


def centered_polygonal_sequence(s: int, count: int) -> List[int]:
    return [centered_polygonal(s, n) for n in range(count)]


# ---------------------------------------------------------------------------
# Star numbers
# ---------------------------------------------------------------------------

def star_number(n: int) -> int:
    """6n(n-1)+1 — hexagram star number."""
    return 6 * n * (n - 1) + 1


def star_numbers(count: int) -> List[int]:
    return [star_number(n) for n in range(1, count + 1)]


# ---------------------------------------------------------------------------
# Polyhedral numbers (3-D)
# ---------------------------------------------------------------------------

def tetrahedral(n: int) -> int:
    return n * (n + 1) * (n + 2) // 6


def cube_num(n: int) -> int:
    return n ** 3


def octahedral(n: int) -> int:
    return n * (2 * n * n + 1) // 3


def dodecahedral(n: int) -> int:
    return n * (3 * n - 1) * (3 * n - 2) // 2


def icosahedral(n: int) -> int:
    return n * (5 * n * n - 5 * n + 2) // 2


def truncated_tetrahedral(n: int) -> int:
    return 23 * n * n - 27 * n + 10 if n >= 1 else 0


def square_pyramidal(n: int) -> int:
    return n * (n + 1) * (2 * n + 1) // 6


def pentagonal_pyramidal(n: int) -> int:
    return n * n * (n + 1) // 2


def polyhedral_sequence(fn, count: int) -> List[int]:
    return [fn(n) for n in range(1, count + 1)]


# ---------------------------------------------------------------------------
# Gnomonic numbers (differences of consecutive polygonals)
# ---------------------------------------------------------------------------

def gnomons(s: int, count: int) -> List[int]:
    seq = polygonal_sequence(s, count + 1)
    return [seq[i + 1] - seq[i] for i in range(count)]


# ---------------------------------------------------------------------------
# Oblong / pronic numbers
# ---------------------------------------------------------------------------

def pronic(n: int) -> int:
    """n*(n+1) — oblong number."""
    return n * (n + 1)


def pronic_sequence(count: int) -> List[int]:
    return [pronic(n) for n in range(count)]


# ---------------------------------------------------------------------------
# Cross / plus numbers
# ---------------------------------------------------------------------------

def cross_number(n: int) -> int:
    """Points in a + cross of arm-length n: 4n+1."""
    return 4 * n + 1


# ---------------------------------------------------------------------------
# Lazy caterer / pancake numbers
# ---------------------------------------------------------------------------

def lazy_caterer(n: int) -> int:
    """Maximum pieces from n straight cuts: n(n+1)/2 + 1."""
    return n * (n + 1) // 2 + 1


def pancake_number(n: int) -> int:
    """Cake numbers (3-D cuts): n(n^2+5)/6 + 1."""
    return n * (n * n + 5) // 6 + 1


def lazy_caterer_sequence(count: int) -> List[int]:
    return [lazy_caterer(n) for n in range(count)]


# ---------------------------------------------------------------------------
# Composite figurate sequences
# ---------------------------------------------------------------------------

class FigurateGenerator:
    """Unified interface for all figurate number sequences."""

    POLYGONAL = {
        3: triangular, 4: square_num, 5: pentagonal, 6: hexagonal,
        7: heptagonal, 8: octagonal, 9: nonagonal, 10: decagonal,
    }

    POLYHEDRAL = {
        "tetrahedral": tetrahedral,
        "cube": cube_num,
        "octahedral": octahedral,
        "dodecahedral": dodecahedral,
        "icosahedral": icosahedral,
        "square_pyramidal": square_pyramidal,
        "pentagonal_pyramidal": pentagonal_pyramidal,
    }

    def generate_polygonal(self, sides: int, count: int) -> List[int]:
        if sides in self.POLYGONAL:
            fn = self.POLYGONAL[sides]
            return [fn(n) for n in range(1, count + 1)]
        return polygonal_sequence(sides, count)

    def generate_centered(self, sides: int, count: int) -> List[int]:
        return centered_polygonal_sequence(sides, count)

    def generate_polyhedral(self, shape: str, count: int) -> List[int]:
        fn = self.POLYHEDRAL.get(shape)
        if fn is None:
            raise ValueError(f"Unknown polyhedral shape: {shape}")
        return polyhedral_sequence(fn, count)

    def is_triangular(self, n: int) -> bool:
        """Check whether n is a triangular number."""
        if n < 0:
            return False
        disc = 1 + 8 * n
        sq = int(math.isqrt(disc))
        return sq * sq == disc and (sq - 1) % 2 == 0

    def is_square(self, n: int) -> bool:
        sq = int(math.isqrt(n))
        return sq * sq == n

    def is_pentagonal(self, n: int) -> bool:
        if n < 1:
            return False
        disc = 1 + 24 * n
        sq = int(math.isqrt(disc))
        return sq * sq == disc and (sq + 1) % 6 == 0

    def is_hexagonal(self, n: int) -> bool:
        if n < 1:
            return False
        disc = 1 + 8 * n
        sq = int(math.isqrt(disc))
        return sq * sq == disc and (sq + 1) % 4 == 0

    def figurate_iter(self, s: int) -> Iterator[int]:
        n = 1
        while True:
            yield polygonal(s, n)
            n += 1
