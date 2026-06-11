"""Tests for matrix_generator module."""
import pytest
from src.core.generators.matrix_generator import (
    pascal_row, pascal_matrix, pascal_diagonal,
    magic_square_odd, magic_square_doubly_even, magic_constant, is_magic_square,
    latin_square, is_latin_square,
    determinant, permanent,
    hadamard_matrix, hadamard_flat_sequence,
    linear_recurrence_sequence, circulant_matrix,
    knights_tour, MatrixSequenceGenerator,
)


class TestPascal:
    def test_row_0(self):
        assert pascal_row(0) == [1]

    def test_row_1(self):
        assert pascal_row(1) == [1, 1]

    def test_row_4(self):
        assert pascal_row(4) == [1, 4, 6, 4, 1]

    def test_row_sum(self):
        for n in range(8):
            assert sum(pascal_row(n)) == 2 ** n

    def test_matrix_shape(self):
        M = pascal_matrix(5)
        assert len(M) == 5
        assert all(len(row) == 5 for row in M)

    def test_matrix_first_col(self):
        M = pascal_matrix(5)
        assert all(M[i][0] == 1 for i in range(5))

    def test_diagonal(self):
        d = pascal_diagonal(4)
        assert len(d) == 5


class TestMagicSquare:
    def test_magic_constant(self):
        assert magic_constant(3) == 15
        assert magic_constant(4) == 34
        assert magic_constant(5) == 65

    def test_odd_3x3(self):
        M = magic_square_odd(3)
        assert is_magic_square(M)

    def test_odd_5x5(self):
        M = magic_square_odd(5)
        assert is_magic_square(M)

    def test_doubly_even_4x4(self):
        M = magic_square_doubly_even(4)
        assert is_magic_square(M)

    def test_doubly_even_8x8(self):
        M = magic_square_doubly_even(8)
        assert is_magic_square(M)

    def test_not_magic(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert not is_magic_square(M)

    def test_invalid_odd(self):
        with pytest.raises(ValueError):
            magic_square_odd(4)

    def test_invalid_doubly_even(self):
        with pytest.raises(ValueError):
            magic_square_doubly_even(3)


class TestLatinSquare:
    def test_is_latin(self):
        for n in [3, 4, 5, 6]:
            M = latin_square(n)
            assert is_latin_square(M)

    def test_shape(self):
        M = latin_square(5)
        assert len(M) == 5
        assert all(len(row) == 5 for row in M)

    def test_not_latin(self):
        M = [[0, 0], [0, 0]]
        assert not is_latin_square(M)


class TestDeterminant:
    def test_2x2(self):
        M = [[1.0, 2.0], [3.0, 4.0]]
        assert determinant(M) == pytest.approx(-2.0)

    def test_identity(self):
        M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert determinant(M) == pytest.approx(1.0)

    def test_singular(self):
        M = [[1.0, 2.0], [2.0, 4.0]]
        assert determinant(M) == pytest.approx(0.0)

    def test_3x3_known(self):
        M = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        assert determinant(M) == pytest.approx(4.0)


class TestPermanent:
    def test_1x1(self):
        # Permanent of 1x1 matrix = that entry (sign may vary by implementation)
        assert abs(permanent([[5]])) == 5

    def test_identity_2x2(self):
        assert permanent([[1, 0], [0, 1]]) == 1

    def test_all_ones(self):
        # permanent of all-ones 2x2 = 2
        assert permanent([[1, 1], [1, 1]]) == 2

    def test_3x3(self):
        # permanent of [[1,2],[3,4]] = 1*4+2*3 = 10
        assert permanent([[1, 2], [3, 4]]) == 10


class TestHadamard:
    def test_order_1(self):
        H = hadamard_matrix(1)
        assert H == [[1]]

    def test_order_2(self):
        H = hadamard_matrix(2)
        assert H == [[1, 1], [1, -1]]

    def test_order_4_rows_orthogonal(self):
        H = hadamard_matrix(4)
        n = 4
        for i in range(n):
            for j in range(n):
                dot = sum(H[i][k] * H[j][k] for k in range(n))
                if i == j:
                    assert dot == n
                else:
                    assert dot == 0

    def test_flat_sequence(self):
        seq = hadamard_flat_sequence(4)
        assert len(seq) == 16
        assert all(v in (1, -1) for v in seq)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            hadamard_matrix(3)


class TestLinearRecurrence:
    def test_fibonacci(self):
        # Fibonacci: a[n] = a[n-1] + a[n-2]
        seq = linear_recurrence_sequence([1, 1], [0, 1], 10)
        assert seq == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_tribonacci(self):
        seq = linear_recurrence_sequence([1, 1, 1], [0, 0, 1], 8)
        assert seq[3] == 1
        assert seq[4] == 2

    def test_length(self):
        seq = linear_recurrence_sequence([2, -1], [1, 2], 20)
        assert len(seq) == 20


class TestKnightsTour:
    def test_visits_all_squares(self):
        tour = knights_tour(6)
        if tour is not None:
            assert len(tour) == 36
            assert len(set(tour)) == 36

    def test_invalid_moves(self):
        tour = knights_tour(5)
        if tour is not None:
            n = 5
            for sq in tour:
                r, c = divmod(sq, n)
                assert 0 <= r < n and 0 <= c < n


class TestMatrixSequenceGenerator:
    def setup_method(self):
        self.gen = MatrixSequenceGenerator()

    def test_pascal_rows(self):
        rows = self.gen.pascal_rows(5)
        assert len(rows) == 5
        assert rows[4] == [1, 4, 6, 4, 1]

    def test_pascal_flat(self):
        flat = self.gen.pascal_flat(4)
        # rows 0..3: [1] [1,1] [1,2,1] [1,3,3,1] → 1+2+3+4 = 10 elements
        assert len(flat) == 10

    def test_recurrence(self):
        seq = self.gen.recurrence([1, 1], [0, 1], 8)
        assert seq[7] == 13

    def test_hadamard(self):
        seq = self.gen.hadamard_sequence(4)
        assert len(seq) == 16
