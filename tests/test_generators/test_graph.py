"""Tests for graph_generator module."""
import pytest
from src.core.generators.graph_generator import (
    complete_graph, cycle_graph, path_graph, star_graph,
    chromatic_polynomial_path, chromatic_polynomial_cycle, chromatic_polynomial_complete,
    eulerian_number, eulerian_numbers_row, eulerian_triangle,
    stirling_first, stirling_second, stirling_first_row, stirling_second_row,
    ramsey_number, ramsey_sequence, cayley_formula, cayley_sequence,
    GraphWalkGenerator, GraphSequenceGenerator,
)


class TestBasicGraphs:
    def test_complete_degree(self):
        g = complete_graph(5)
        assert all(len(nbrs) == 4 for nbrs in g.values())

    def test_cycle_degree(self):
        g = cycle_graph(6)
        assert all(len(nbrs) == 2 for nbrs in g.values())

    def test_path_endpoints(self):
        g = path_graph(5)
        assert len(g[0]) == 1
        assert len(g[4]) == 1
        assert len(g[2]) == 2

    def test_star_center_degree(self):
        g = star_graph(4)
        assert len(g[0]) == 4


class TestChromaticPolynomials:
    def test_path_k2(self):
        # P_n at k=2: 2*1^(n-1) = 2
        assert chromatic_polynomial_path(3, 2) == 2

    def test_path_k3(self):
        # P_3 at k=3: 3*2*2 = 12
        assert chromatic_polynomial_path(3, 3) == 12

    def test_cycle_3_k3(self):
        # C_3 = K_3, chromatic poly at k=3: (3-1)^3 + (-1)^3*(3-1) = 8-2 = 6
        assert chromatic_polynomial_cycle(3, 3) == 6

    def test_complete_k_equal_n(self):
        # K_n at k=n: n! ways to color
        import math
        n = 4
        assert chromatic_polynomial_complete(n, n) == math.factorial(n)

    def test_complete_insufficient_colors(self):
        # K_4 at k=3: not possible → 0
        assert chromatic_polynomial_complete(4, 3) == 0


class TestEulerianNumbers:
    def test_eulerian_1_0(self):
        assert eulerian_number(1, 0) == 1

    def test_eulerian_3_1(self):
        # A(3,1) = 4
        assert eulerian_number(3, 1) == 4

    def test_eulerian_row_sum(self):
        # Sum of row n = n!
        import math
        for n in range(1, 7):
            assert sum(eulerian_numbers_row(n)) == math.factorial(n)

    def test_triangle_rows(self):
        tri = eulerian_triangle(5)
        assert len(tri) == 5
        assert tri[0] == [1]  # n=1: only A(1,0)=1

    def test_known_a4_row(self):
        # A(4,k) = [1, 11, 11, 1]
        row = eulerian_numbers_row(4)
        assert row == [1, 11, 11, 1]


class TestStirlingNumbers:
    def test_stirling_second_1_1(self):
        assert stirling_second(1, 1) == 1

    def test_stirling_second_4_2(self):
        assert stirling_second(4, 2) == 7

    def test_stirling_first_4_2(self):
        # c(4,2) = 11
        assert stirling_first(4, 2) == 11

    def test_stirling_second_row_sum(self):
        # sum(S(n,k), k=0..n) = Bell(n)
        # Bell(4) = 15
        row = stirling_second_row(4)
        assert sum(row) == 15

    def test_stirling_first_row(self):
        row = stirling_first_row(4)
        assert len(row) == 5  # k = 0..4

    def test_stirling_second_bell_numbers(self):
        bell = [1, 1, 2, 5, 15, 52]
        for n, bn in enumerate(bell):
            assert sum(stirling_second_row(n)) == bn


class TestRamseyNumbers:
    def test_r33(self):
        assert ramsey_number(3, 3) == 6

    def test_r34(self):
        assert ramsey_number(3, 4) == 9

    def test_r44(self):
        assert ramsey_number(4, 4) == 18

    def test_symmetric(self):
        assert ramsey_number(3, 5) == ramsey_number(5, 3)

    def test_unknown_returns_none(self):
        assert ramsey_number(6, 6) is None

    def test_sequence_nonempty(self):
        seq = ramsey_sequence(5)
        assert len(seq) > 0


class TestCayleyFormula:
    def test_n2(self):
        assert cayley_formula(2) == 1  # 2^0

    def test_n3(self):
        assert cayley_formula(3) == 3  # 3^1

    def test_n4(self):
        assert cayley_formula(4) == 16  # 4^2

    def test_sequence(self):
        seq = cayley_sequence(5)
        assert seq[0] == 1  # n=2: 2^0=1
        assert seq[1] == 3  # n=3: 3^1=3
        assert seq[2] == 16  # n=4: 4^2=16


class TestGraphWalkGenerator:
    def test_walk_stays_in_graph(self):
        g = cycle_graph(6)
        gen = GraphWalkGenerator(g, start=0, seed=42)
        path = gen.walk(20)
        assert all(0 <= v < 6 for v in path)

    def test_walk_length(self):
        g = complete_graph(5)
        gen = GraphWalkGenerator(g, start=0, seed=1)
        path = gen.walk(50)
        assert len(path) == 50

    def test_reproducible(self):
        g = cycle_graph(4)
        gen1 = GraphWalkGenerator(g, start=0, seed=99)
        gen2 = GraphWalkGenerator(g, start=0, seed=99)
        assert gen1.walk(30) == gen2.walk(30)


class TestGraphSequenceGenerator:
    def setup_method(self):
        self.gen = GraphSequenceGenerator()

    def test_eulerian_triangle(self):
        tri = self.gen.eulerian_triangle(4)
        assert len(tri) == 4

    def test_eulerian_flat(self):
        flat = self.gen.eulerian_flat(4)
        # rows 1-4: sizes 1,2,3,4 → 10 elements
        assert len(flat) == 10

    def test_stirling_first_triangle(self):
        tri = self.gen.stirling_first_triangle(4)
        assert len(tri) == 5  # row 0..4

    def test_cayley_sequence(self):
        seq = self.gen.cayley_sequence(5)
        assert seq[0] == 1
        assert seq[2] == 16
