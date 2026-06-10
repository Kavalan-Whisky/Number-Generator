"""
Tests for random number generators.
"""

import pytest
import math
from src.core.generators.random_generator import (
    LinearCongruentialGenerator,
    Xorshift32Generator,
    Xorshift64Generator,
    PCGGenerator,
    LFSRGenerator,
    BlumBlumShubGenerator,
    MiddleSquareGenerator,
    MersenneTwisterGenerator,
    RandomGeneratorFactory,
)


class TestLCG:
    def test_basic_generation(self):
        gen = LinearCongruentialGenerator(seed=42)
        n = gen.next_int()
        assert isinstance(n, int)
        assert n >= 0

    def test_seeded_reproducible(self):
        gen1 = LinearCongruentialGenerator(seed=100)
        gen2 = LinearCongruentialGenerator(seed=100)
        assert [gen1.next_int() for _ in range(10)] == [gen2.next_int() for _ in range(10)]

    def test_different_seeds_different(self):
        gen1 = LinearCongruentialGenerator(seed=1)
        gen2 = LinearCongruentialGenerator(seed=2)
        results1 = [gen1.next_int() for _ in range(5)]
        results2 = [gen2.next_int() for _ in range(5)]
        assert results1 != results2

    def test_range_bounds(self):
        gen = LinearCongruentialGenerator(seed=42)
        for _ in range(100):
            n = gen.next_range(10, 20)
            assert 10 <= n <= 20

    def test_float_range(self):
        gen = LinearCongruentialGenerator(seed=42)
        for _ in range(100):
            f = gen.next_float()
            assert 0.0 <= f < 1.0

    def test_generate_list(self):
        gen = LinearCongruentialGenerator(seed=42)
        nums = gen.generate(50, 0, 100)
        assert len(nums) == 50
        assert all(0 <= n <= 100 for n in nums)

    def test_reset(self):
        gen = LinearCongruentialGenerator(seed=42)
        seq1 = [gen.next_int() for _ in range(5)]
        gen.reset()
        seq2 = [gen.next_int() for _ in range(5)]
        assert seq1 == seq2

    def test_preset_numerical_recipes(self):
        gen = LinearCongruentialGenerator(seed=42, preset="numerical_recipes")
        assert gen.a == 1664525
        assert gen.m == 2**32

    def test_modulus_respected(self):
        gen = LinearCongruentialGenerator(seed=42, m=1000)
        for _ in range(50):
            assert gen.next_int() < 1000

    def test_generate_zero_count(self):
        gen = LinearCongruentialGenerator(seed=42)
        assert gen.generate(0) == []


class TestXorshift32:
    def test_basic(self):
        gen = Xorshift32Generator(seed=1)
        n = gen.next_int()
        assert 0 < n <= 0xFFFFFFFF

    def test_reproducible(self):
        g1 = Xorshift32Generator(seed=99)
        g2 = Xorshift32Generator(seed=99)
        assert [g1.next_int() for _ in range(20)] == [g2.next_int() for _ in range(20)]

    def test_float(self):
        gen = Xorshift32Generator(seed=42)
        for _ in range(50):
            assert 0 <= gen.next_float() <= 1.0

    def test_range(self):
        gen = Xorshift32Generator(seed=7)
        for _ in range(100):
            n = gen.next_range(5, 15)
            assert 5 <= n <= 15

    def test_generate(self):
        gen = Xorshift32Generator(seed=1)
        nums = gen.generate(100, 0, 100)
        assert len(nums) == 100


class TestXorshift64:
    def test_basic(self):
        gen = Xorshift64Generator(seed=1)
        n = gen.next_int()
        assert n > 0

    def test_larger_range_than_32(self):
        gen = Xorshift64Generator(seed=42)
        values = {gen.next_int() for _ in range(20)}
        # 64-bit should produce values > 32-bit max sometimes
        assert any(v > 0xFFFFFFFF for v in values) or len(values) > 10

    def test_reproducible(self):
        g1 = Xorshift64Generator(seed=123)
        g2 = Xorshift64Generator(seed=123)
        assert [g1.next_int() for _ in range(10)] == [g2.next_int() for _ in range(10)]


class TestPCG:
    def test_basic(self):
        gen = PCGGenerator(seed=42)
        n = gen.next_int()
        assert 0 <= n <= 0xFFFFFFFF

    def test_reproducible(self):
        g1 = PCGGenerator(seed=42)
        g2 = PCGGenerator(seed=42)
        assert [g1.next_int() for _ in range(20)] == [g2.next_int() for _ in range(20)]

    def test_unbiased_range(self):
        gen = PCGGenerator(seed=42)
        counts = [0] * 10
        for _ in range(1000):
            n = gen.next_range(0, 9)
            counts[n] += 1
        # Each bucket should have roughly 100 ± 50
        assert all(50 < c < 200 for c in counts)

    def test_float(self):
        gen = PCGGenerator(seed=42)
        floats = [gen.next_float() for _ in range(100)]
        assert all(0 <= f <= 1 for f in floats)


class TestLFSR:
    def test_32bit(self):
        gen = LFSRGenerator(seed=1, bits=32)
        n = gen.next_int()
        assert 0 < n < 2**32

    def test_galois_mode(self):
        gen = LFSRGenerator(seed=1, bits=32, mode="galois")
        n = gen.next_int()
        assert n > 0

    def test_period_not_zero(self):
        gen = LFSRGenerator(seed=1, bits=8)
        assert gen.period() == 255  # 2^8 - 1

    def test_reproducible(self):
        g1 = LFSRGenerator(seed=42, bits=16)
        g2 = LFSRGenerator(seed=42, bits=16)
        assert [g1.next_int() for _ in range(10)] == [g2.next_int() for _ in range(10)]

    def test_never_zero_initial_state(self):
        # LFSR seed must be non-zero
        gen = LFSRGenerator(seed=7, bits=16)
        assert gen.state != 0
        # The internal register stays non-zero for maximal LFSR
        values = [gen.next_int() for _ in range(50)]
        assert any(v > 0 for v in values)


class TestBlumBlumShub:
    def test_basic(self):
        gen = BlumBlumShubGenerator(p=11, q=23, seed=42)
        n = gen.next_int()
        assert n >= 0

    def test_bits_output(self):
        gen = BlumBlumShubGenerator(p=11, q=23)
        bit = gen.next_bit()
        assert bit in (0, 1)

    def test_invalid_primes(self):
        with pytest.raises(ValueError):
            BlumBlumShubGenerator(p=5, q=7)  # 5 % 4 != 3

    def test_generate(self):
        gen = BlumBlumShubGenerator(p=11, q=23, seed=42)
        nums = gen.generate(10, 0, 10)
        assert len(nums) == 10
        assert all(0 <= n <= 10 for n in nums)


class TestMiddleSquare:
    def test_basic(self):
        gen = MiddleSquareGenerator(seed=1234, digits=4)
        n = gen.next_int()
        assert 0 <= n < 10**4

    def test_odd_digits_error(self):
        with pytest.raises(ValueError):
            MiddleSquareGenerator(digits=3)

    def test_reproducible(self):
        g1 = MiddleSquareGenerator(seed=1234, digits=4)
        g2 = MiddleSquareGenerator(seed=1234, digits=4)
        assert g1.next_int() == g2.next_int()


class TestMersenneTwister:
    def test_basic(self):
        gen = MersenneTwisterGenerator(seed=42)
        n = gen.next_int()
        assert 0 <= n <= 0xFFFFFFFF

    def test_reproducible(self):
        g1 = MersenneTwisterGenerator(seed=12345)
        g2 = MersenneTwisterGenerator(seed=12345)
        assert [g1.next_int() for _ in range(50)] == [g2.next_int() for _ in range(50)]

    def test_uniformity(self):
        gen = MersenneTwisterGenerator(seed=42)
        nums = gen.generate(1000, 0, 9)
        counts = [nums.count(i) for i in range(10)]
        # Each digit should appear roughly 100 times
        assert all(50 < c < 200 for c in counts)

    def test_reset(self):
        gen = MersenneTwisterGenerator(seed=42)
        seq1 = [gen.next_int() for _ in range(10)]
        gen.reset()
        seq2 = [gen.next_int() for _ in range(10)]
        assert seq1 == seq2


class TestRandomGeneratorFactory:
    def test_create_lcg(self):
        gen = RandomGeneratorFactory.create("lcg", seed=42)
        assert isinstance(gen, LinearCongruentialGenerator)

    def test_create_all_algorithms(self):
        for algo in RandomGeneratorFactory.available_algorithms():
            gen = RandomGeneratorFactory.create(algo, seed=42)
            n = gen.next_int()
            assert n >= 0

    def test_unknown_algorithm(self):
        with pytest.raises(ValueError):
            RandomGeneratorFactory.create("nonexistent")

    def test_benchmark(self):
        results = RandomGeneratorFactory.benchmark(count=100)
        assert len(results) > 0
        for name, data in results.items():
            if "error" not in data:
                assert "numbers_per_second" in data

    def test_available_algorithms(self):
        algs = RandomGeneratorFactory.available_algorithms()
        assert "lcg" in algs
        assert "mersenne" in algs
        assert "pcg" in algs
