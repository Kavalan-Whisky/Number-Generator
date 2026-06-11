"""
Graph-theory-based number sequences.

Chromatic polynomials, graph invariants, OEIS graph sequences
(graph enumeration, Ramsey numbers, tree counting), and
integer sequences derived from well-known graph families.
"""

import math
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Basic graph type
# ---------------------------------------------------------------------------

Graph = Dict[int, List[int]]


def complete_graph(n: int) -> Graph:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def cycle_graph(n: int) -> Graph:
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


def path_graph(n: int) -> Graph:
    g: Graph = {i: [] for i in range(n)}
    for i in range(n - 1):
        g[i].append(i + 1)
        g[i + 1].append(i)
    return g


def star_graph(n: int) -> Graph:
    """Star with center 0 and n leaves."""
    g: Graph = {i: [] for i in range(n + 1)}
    for i in range(1, n + 1):
        g[0].append(i)
        g[i].append(0)
    return g


def wheel_graph(n: int) -> Graph:
    g = cycle_graph(n)
    hub = n
    g[hub] = list(range(n))
    for i in range(n):
        g[i].append(hub)
    return g


def petersen_graph() -> Graph:
    outer = cycle_graph(5)
    inner = {5: [6, 9], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 5]}
    spokes = {0: [5], 1: [6], 2: [7], 3: [8], 4: [9]}
    g: Graph = {}
    for v in range(10):
        nbrs = []
        if v < 5:
            nbrs = outer[v] + spokes[v]
        else:
            nbrs = inner[v]
            nbrs += [v - 5]
        g[v] = nbrs
    return g


# ---------------------------------------------------------------------------
# Graph invariant sequences
# ---------------------------------------------------------------------------

def chromatic_polynomial_path(n: int, k: int) -> int:
    """Chromatic polynomial of P_n evaluated at k: k*(k-1)^(n-1)."""
    if n == 0:
        return 1
    return k * (k - 1) ** (n - 1)


def chromatic_polynomial_cycle(n: int, k: int) -> int:
    """Chromatic polynomial of C_n: (k-1)^n + (-1)^n * (k-1)."""
    return (k - 1) ** n + (-1) ** n * (k - 1)


def chromatic_polynomial_complete(n: int, k: int) -> int:
    """Chromatic polynomial of K_n: falling factorial k^(n)."""
    result = 1
    for i in range(n):
        result *= (k - i)
    return result


def independence_number_path(n: int) -> int:
    """Independence number of P_n = ceil(n/2)."""
    return (n + 1) // 2


def independence_number_cycle(n: int) -> int:
    return n // 2


def domination_number_path(n: int) -> int:
    return math.ceil(n / 3)


def clique_number_complete(n: int) -> int:
    return n


# ---------------------------------------------------------------------------
# Eulerian / Hamiltonian number sequences
# ---------------------------------------------------------------------------

def eulerian_number(n: int, k: int) -> int:
    """A(n,k): number of permutations of [n] with exactly k ascents."""
    if k < 0 or k >= n:
        return 0 if k != 0 or n != 0 else 1
    return sum(
        (-1) ** j * math.comb(n + 1, j) * (k + 1 - j) ** n
        for j in range(k + 2)
    )


def eulerian_numbers_row(n: int) -> List[int]:
    return [eulerian_number(n, k) for k in range(n)]


def eulerian_triangle(rows: int) -> List[List[int]]:
    return [eulerian_numbers_row(n) for n in range(1, rows + 1)]


# ---------------------------------------------------------------------------
# Stirling numbers
# ---------------------------------------------------------------------------

def stirling_first(n: int, k: int, memo: Optional[dict] = None) -> int:
    """Unsigned Stirling numbers of the first kind c(n,k)."""
    if memo is None:
        memo = {}
    if (n, k) in memo:
        return memo[(n, k)]
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    result = (n - 1) * stirling_first(n - 1, k, memo) + stirling_first(n - 1, k - 1, memo)
    memo[(n, k)] = result
    return result


def stirling_second(n: int, k: int, memo: Optional[dict] = None) -> int:
    """Stirling numbers of the second kind S(n,k)."""
    if memo is None:
        memo = {}
    if (n, k) in memo:
        return memo[(n, k)]
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    result = k * stirling_second(n - 1, k, memo) + stirling_second(n - 1, k - 1, memo)
    memo[(n, k)] = result
    return result


def stirling_first_row(n: int) -> List[int]:
    return [stirling_first(n, k) for k in range(n + 1)]


def stirling_second_row(n: int) -> List[int]:
    return [stirling_second(n, k) for k in range(n + 1)]


# ---------------------------------------------------------------------------
# Ramsey-related known values
# ---------------------------------------------------------------------------

RAMSEY_KNOWN: Dict[Tuple[int, int], int] = {
    (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18, (3, 7): 23,
    (3, 8): 28, (3, 9): 36, (4, 4): 18, (4, 5): 25,
}


def ramsey_number(r: int, s: int) -> Optional[int]:
    """Return known Ramsey number R(r,s) or None if unknown."""
    key = (min(r, s), max(r, s))
    return RAMSEY_KNOWN.get(key)


def ramsey_sequence(max_r: int = 5) -> List[Tuple[int, int, int]]:
    """Return all known Ramsey numbers R(r,s) with r<=s<=max_r."""
    result = []
    for r in range(3, max_r + 1):
        for s in range(r, max_r + 1):
            v = ramsey_number(r, s)
            if v is not None:
                result.append((r, s, v))
    return result


# ---------------------------------------------------------------------------
# Spanning trees — Cayley / Kirchhoff
# ---------------------------------------------------------------------------

def cayley_formula(n: int) -> int:
    """Number of labelled trees on n vertices: n^(n-2)."""
    if n < 2:
        return 1
    return n ** (n - 2)


def cayley_sequence(count: int) -> List[int]:
    return [cayley_formula(n) for n in range(2, count + 2)]


# ---------------------------------------------------------------------------
# Graph coloring sequences for small graphs
# ---------------------------------------------------------------------------

def chromatic_sequence_paths(k: int, max_n: int = 10) -> List[int]:
    """Chromatic polynomial of P_n evaluated at k, for n=1..max_n."""
    return [chromatic_polynomial_path(n, k) for n in range(1, max_n + 1)]


def chromatic_sequence_cycles(k: int, max_n: int = 10) -> List[int]:
    return [chromatic_polynomial_cycle(n, k) for n in range(3, max_n + 3)]


# ---------------------------------------------------------------------------
# Graph-based PRNG (random walk on a graph)
# ---------------------------------------------------------------------------

class GraphWalkGenerator:
    """Pseudo-random number generation via random walk on a graph."""

    def __init__(self, graph: Graph, start: int = 0, seed: int = 42):
        self.graph = graph
        self.current = start
        self._state = seed

    def _lcg_step(self) -> int:
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state

    def step(self) -> int:
        nbrs = self.graph[self.current]
        idx = self._lcg_step() % len(nbrs)
        self.current = nbrs[idx]
        return self.current

    def walk(self, steps: int) -> List[int]:
        return [self.step() for _ in range(steps)]


# ---------------------------------------------------------------------------
# Unified generator class
# ---------------------------------------------------------------------------

class GraphSequenceGenerator:
    """High-level interface for graph-derived integer sequences."""

    def eulerian_triangle(self, rows: int) -> List[List[int]]:
        return eulerian_triangle(rows)

    def eulerian_flat(self, rows: int) -> List[int]:
        return [v for row in eulerian_triangle(rows) for v in row]

    def stirling_first_triangle(self, rows: int) -> List[List[int]]:
        return [stirling_first_row(n) for n in range(rows + 1)]

    def stirling_second_triangle(self, rows: int) -> List[List[int]]:
        return [stirling_second_row(n) for n in range(rows + 1)]

    def cayley_sequence(self, count: int) -> List[int]:
        return cayley_sequence(count)

    def chromatic_paths(self, k: int, count: int) -> List[int]:
        return chromatic_sequence_paths(k, count)

    def ramsey_numbers(self) -> List[Tuple[int, int, int]]:
        return ramsey_sequence()
