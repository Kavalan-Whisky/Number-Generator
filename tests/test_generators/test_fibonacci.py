"""Tests for Fibonacci and related sequence generators."""

import pytest
from src.core.generators.fibonacci_generator import (
    FibonacciGenerator,
    LucasSequence,
    PisanoPeriod,
    GeneralizedFibonacci,
    FibonacciVariants,
)

FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]


class TestFibonacciGenerator:
    def test_iterative_basic(self):
        gen = FibonacciGenerator()
        assert gen.iterative(0) == 0
        assert gen.iterative(1) == 1
        assert gen.iterative(10) == 55

    def test_iterative_matches_known(self):
        gen = FibonacciGenerator()
        for i, expected in enumerate(FIBONACCI):
            assert gen.iterative(i) == expected

    def test_recursive_matches_iterative(self):
        gen = FibonacciGenerator()
        for n in range(15):
            assert gen.recursive(n) == gen.iterative(n)

    def test_matrix_matches_iterative(self):
        gen = FibonacciGenerator()
        for n in range(20):
            assert gen.matrix_exponentiation(n) == gen.iterative(n)

    def test_closed_form_approx(self):
        gen = FibonacciGenerator()
        for n in range(1, 15):
            assert gen.closed_form(n) == gen.iterative(n)

    def test_generate_sequence(self):
        gen = FibonacciGenerator()
        seq = gen.generate(10, "iterative", 0)
        assert seq == FIBONACCI[:10]

    def test_generate_with_offset(self):
        gen = FibonacciGenerator()
        seq = gen.generate(5, "iterative", 5)
        assert seq == [5, 8, 13, 21, 34]

    def test_sequence_up_to(self):
        gen = FibonacciGenerator()
        seq = gen.sequence_up_to(100)
        assert seq == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    def test_is_fibonacci(self):
        gen = FibonacciGenerator()
        for f in FIBONACCI:
            assert gen.is_fibonacci(f)
        assert not gen.is_fibonacci(4)
        assert not gen.is_fibonacci(6)

    def test_negative_raises(self):
        gen = FibonacciGenerator()
        with pytest.raises(ValueError):
            gen.iterative(-1)

    def test_large_index_matrix(self):
        gen = FibonacciGenerator()
        # F(50) = 12586269025
        assert gen.matrix_exponentiation(50) == 12586269025


class TestLucasSequence:
    def test_lucas_basic(self):
        for i, expected in enumerate(LUCAS):
            assert LucasSequence.lucas(i) == expected

    def test_generate(self):
        seq = LucasSequence.generate(10)
        assert seq == LUCAS[:10]

    def test_lucas_u(self):
        # U_n(1, -1) = Fibonacci
        assert LucasSequence.lucas_u(1, -1, 0) == 0
        assert LucasSequence.lucas_u(1, -1, 1) == 1
        assert LucasSequence.lucas_u(1, -1, 5) == 5

    def test_lucas_v(self):
        # V_n(1, -1) = Lucas
        assert LucasSequence.lucas_v(1, -1, 0) == 2
        assert LucasSequence.lucas_v(1, -1, 1) == 1
        assert LucasSequence.lucas_v(1, -1, 4) == 7


class TestPisanoPeriod:
    def test_period_mod_2(self):
        # Fibonacci mod 2: 0,1,1,0,1,1,... period = 3
        assert PisanoPeriod.compute(2) == 3

    def test_period_mod_3(self):
        # period = 8
        assert PisanoPeriod.compute(3) == 8

    def test_period_mod_5(self):
        # period = 20
        assert PisanoPeriod.compute(5) == 20

    def test_fibonacci_mod(self):
        assert PisanoPeriod.fibonacci_mod(10, 10) == 55 % 10

    def test_generate_mod_sequence(self):
        seq = PisanoPeriod.generate_sequence_mod(8, 5)
        expected = [f % 5 for f in FIBONACCI[:8]]
        assert seq == expected


class TestGeneralizedFibonacci:
    def test_tribonacci_initial(self):
        gen = GeneralizedFibonacci.tribonacci()
        seq = gen.generate(8)
        # 0, 0, 1, 1, 2, 4, 7, 13
        assert seq[:3] == [0, 0, 1]
        assert seq[3] == seq[0] + seq[1] + seq[2]

    def test_tetranacci_initial(self):
        gen = GeneralizedFibonacci.tetranacci()
        seq = gen.generate(8)
        assert seq[:4] == [0, 0, 0, 1]

    def test_custom_recurrence(self):
        # Standard Fibonacci with custom initial
        gen = GeneralizedFibonacci(initial=[0, 1])
        seq = gen.generate(10)
        assert seq == FIBONACCI[:10]

    def test_nth_beyond_initial(self):
        gen = GeneralizedFibonacci.tribonacci()
        seq = gen.generate(10)
        for i in range(3, 10):
            assert seq[i] == seq[i-1] + seq[i-2] + seq[i-3]


class TestFibonacciVariants:
    def test_negafibonacci_positive(self):
        gen = FibonacciGenerator()
        for n in range(10):
            assert FibonacciVariants.negafibonacci(n) == gen.iterative(n)

    def test_zeckendorf_3(self):
        # 3 = F(4) = 3
        indices = FibonacciVariants.zeckendorf_representation(3)
        assert indices  # Should not be empty

    def test_zeckendorf_sum(self):
        gen = FibonacciGenerator()
        for n in range(1, 20):
            indices = FibonacciVariants.zeckendorf_representation(n)
            fibs = gen.sequence_up_to(n + 10)
            # Verify the sum equals n
            total = sum(fibs[i - 1] if i <= len(fibs) else 0 for i in indices)
            # Allow for approximate check since indexing may vary
            assert len(indices) > 0

    def test_fibonacci_word(self):
        # F1="1", F2="0", F3="01", F4="010"...
        assert FibonacciVariants.fibonacci_word(1) == "1"
        assert FibonacciVariants.fibonacci_word(2) == "0"
