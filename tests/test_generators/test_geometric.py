"""Tests for geometric_generator module."""
import pytest
from src.core.generators.geometric_generator import (
    triangular, square_num, pentagonal, hexagonal,
    heptagonal, octagonal, polygonal, polygonal_sequence,
    centered_triangular, centered_square, centered_hexagonal,
    centered_polygonal, centered_polygonal_sequence,
    star_number, star_numbers,
    tetrahedral, octahedral, icosahedral, square_pyramidal,
    pronic, lazy_caterer, FigurateGenerator,
)


class TestPolygonalNumbers:
    def test_triangular_known(self):
        assert triangular(1) == 1
        assert triangular(2) == 3
        assert triangular(3) == 6
        assert triangular(4) == 10

    def test_square_known(self):
        assert square_num(1) == 1
        assert square_num(4) == 16
        assert square_num(7) == 49

    def test_pentagonal_known(self):
        assert pentagonal(1) == 1
        assert pentagonal(2) == 5
        assert pentagonal(3) == 12

    def test_hexagonal_known(self):
        assert hexagonal(1) == 1
        assert hexagonal(2) == 6
        assert hexagonal(3) == 15

    def test_heptagonal(self):
        assert heptagonal(1) == 1
        assert heptagonal(2) == 7

    def test_octagonal(self):
        assert octagonal(1) == 1
        assert octagonal(2) == 8

    def test_polygonal_general(self):
        assert polygonal(3, 4) == triangular(4)
        assert polygonal(4, 4) == square_num(4)
        assert polygonal(5, 4) == pentagonal(4)

    def test_polygonal_sequence_length(self):
        seq = polygonal_sequence(6, 10)
        assert len(seq) == 10

    def test_polygonal_sequence_increasing(self):
        seq = polygonal_sequence(5, 8)
        assert seq == sorted(seq)


class TestCenteredPolygonal:
    def test_centered_triangular(self):
        assert centered_triangular(1) == 1
        assert centered_triangular(2) == 4
        assert centered_triangular(3) == 10

    def test_centered_square(self):
        assert centered_square(1) == 1
        assert centered_square(2) == 5

    def test_centered_hexagonal(self):
        assert centered_hexagonal(1) == 1
        assert centered_hexagonal(2) == 7
        assert centered_hexagonal(3) == 19

    def test_centered_polygon_starts_at_1(self):
        for s in [3, 4, 5, 6]:
            assert centered_polygonal(s, 0) == 1

    def test_sequence_length(self):
        seq = centered_polygonal_sequence(6, 8)
        assert len(seq) == 8


class TestStarNumbers:
    def test_known(self):
        # star(1)=1, star(2)=13, star(3)=37
        assert star_number(1) == 1
        assert star_number(2) == 13
        assert star_number(3) == 37

    def test_sequence(self):
        seq = star_numbers(5)
        assert seq[0] == 1
        assert seq[1] == 13
        assert len(seq) == 5


class TestPolyhedralNumbers:
    def test_tetrahedral(self):
        assert tetrahedral(1) == 1
        assert tetrahedral(2) == 4
        assert tetrahedral(3) == 10

    def test_octahedral(self):
        assert octahedral(1) == 1
        assert octahedral(2) == 6

    def test_icosahedral(self):
        assert icosahedral(1) == 1
        assert icosahedral(2) == 12

    def test_square_pyramidal(self):
        assert square_pyramidal(1) == 1
        assert square_pyramidal(2) == 5
        assert square_pyramidal(3) == 14


class TestPronic:
    def test_known(self):
        assert pronic(0) == 0
        assert pronic(1) == 2
        assert pronic(4) == 20

    def test_always_even(self):
        for n in range(20):
            assert pronic(n) % 2 == 0


class TestLazyCaterer:
    def test_known(self):
        assert lazy_caterer(0) == 1
        assert lazy_caterer(1) == 2
        assert lazy_caterer(2) == 4
        assert lazy_caterer(3) == 7

    def test_increasing(self):
        vals = [lazy_caterer(n) for n in range(10)]
        assert vals == sorted(vals)


class TestFigurateGenerator:
    def setup_method(self):
        self.gen = FigurateGenerator()

    def test_polygonal_sides(self):
        seq = self.gen.generate_polygonal(6, 5)
        assert seq[0] == 1
        assert seq[1] == 6

    def test_centered(self):
        seq = self.gen.generate_centered(6, 5)
        assert seq[0] == 1

    def test_polyhedral(self):
        seq = self.gen.generate_polyhedral("tetrahedral", 5)
        assert seq[0] == 1
        assert seq[2] == 10

    def test_is_triangular(self):
        assert self.gen.is_triangular(1)
        assert self.gen.is_triangular(10)
        assert not self.gen.is_triangular(11)

    def test_is_square(self):
        assert self.gen.is_square(16)
        assert not self.gen.is_square(15)

    def test_is_pentagonal(self):
        assert self.gen.is_pentagonal(1)
        assert self.gen.is_pentagonal(5)
        assert not self.gen.is_pentagonal(4)

    def test_is_hexagonal(self):
        assert self.gen.is_hexagonal(1)
        assert self.gen.is_hexagonal(6)
        assert not self.gen.is_hexagonal(5)

    def test_figurate_iter(self):
        it = self.gen.figurate_iter(3)
        first5 = [next(it) for _ in range(5)]
        assert first5 == [1, 3, 6, 10, 15]

    def test_unknown_polyhedral(self):
        with pytest.raises(ValueError):
            self.gen.generate_polyhedral("unknown_shape", 5)
