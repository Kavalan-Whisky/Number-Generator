"""Tests for prime number generators."""

import pytest
from src.core.generators.prime_generator import (
    SieveOfEratosthenes,
    SegmentedSieve,
    SundaramSieve,
    miller_rabin_is_prime,
    is_prime,
    next_prime,
    previous_prime,
    twin_primes,
    prime_gaps,
    prime_counting_function,
    prime_factorization,
    PrimeGeneratorFactory,
)


KNOWN_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
KNOWN_COMPOSITES = [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]


class TestSieveOfEratosthenes:
    def test_primes_up_to_50(self):
        sieve = SieveOfEratosthenes(50)
        assert sieve.primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def test_all_known_primes_found(self):
        sieve = SieveOfEratosthenes(50)
        for p in KNOWN_PRIMES:
            assert p in sieve.primes

    def test_no_composites_in_primes(self):
        sieve = SieveOfEratosthenes(50)
        prime_set = set(sieve.primes)
        for c in KNOWN_COMPOSITES:
            assert c not in prime_set

    def test_count_primes_up_to_100(self):
        sieve = SieveOfEratosthenes(100)
        assert len(sieve.primes) == 25

    def test_nth_prime(self):
        sieve = SieveOfEratosthenes(1000)
        assert sieve.nth_prime(1) == 2
        assert sieve.nth_prime(6) == 13
        assert sieve.nth_prime(10) == 29

    def test_is_prime(self):
        sieve = SieveOfEratosthenes(100)
        assert sieve.is_prime(97)
        assert not sieve.is_prime(99)

    def test_get_primes_with_limit(self):
        sieve = SieveOfEratosthenes(100)
        first10 = sieve.get_primes(10)
        assert len(first10) == 10
        assert first10 == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


class TestSegmentedSieve:
    def test_primes_up_to_100(self):
        sieve = SegmentedSieve()
        primes = sieve.primes_up_to(100)
        assert len(primes) == 25
        assert primes[0] == 2
        assert primes[-1] == 97

    def test_primes_in_range(self):
        sieve = SegmentedSieve()
        primes = sieve.primes_in_range(50, 100)
        assert all(50 <= p <= 100 for p in primes)
        assert 53 in primes
        assert 97 in primes

    def test_generate_count(self):
        sieve = SegmentedSieve()
        primes = sieve.generate(50, 2)
        assert len(primes) == 50
        assert all(is_prime(p) for p in primes)

    def test_matches_basic_sieve(self):
        sieve = SegmentedSieve()
        basic = SieveOfEratosthenes(200).primes
        segmented = sieve.primes_up_to(200)
        assert basic == segmented


class TestMillerRabin:
    def test_known_primes(self):
        for p in KNOWN_PRIMES:
            assert miller_rabin_is_prime(p), f"{p} should be prime"

    def test_known_composites(self):
        for c in KNOWN_COMPOSITES:
            assert not miller_rabin_is_prime(c), f"{c} should not be prime"

    def test_large_prime(self):
        assert miller_rabin_is_prime(999999937)

    def test_large_composite(self):
        assert not miller_rabin_is_prime(999999936)

    def test_edge_cases(self):
        assert not miller_rabin_is_prime(0)
        assert not miller_rabin_is_prime(1)
        assert miller_rabin_is_prime(2)
        assert miller_rabin_is_prime(3)
        assert not miller_rabin_is_prime(4)


class TestIsPrime:
    def test_basic(self):
        assert is_prime(2)
        assert is_prime(997)
        assert not is_prime(1)
        assert not is_prime(100)

    def test_all_primes_up_to_100(self):
        primes = SieveOfEratosthenes(100).primes
        for n in range(2, 101):
            assert is_prime(n) == (n in primes)


class TestNextPreviousPrime:
    def test_next_prime(self):
        assert next_prime(10) == 11
        assert next_prime(11) == 13
        assert next_prime(100) == 101

    def test_previous_prime(self):
        assert previous_prime(11) == 7
        assert previous_prime(10) == 7
        assert previous_prime(3) == 2

    def test_previous_prime_none(self):
        assert previous_prime(2) is None
        assert previous_prime(1) is None


class TestTwinPrimes:
    def test_known_twin_primes(self):
        twins = twin_primes(20)
        assert (3, 5) in twins
        assert (5, 7) in twins
        assert (11, 13) in twins
        assert (17, 19) in twins

    def test_all_are_twin_primes(self):
        twins = twin_primes(100)
        for p1, p2 in twins:
            assert p2 - p1 == 2
            assert is_prime(p1) and is_prime(p2)


class TestPrimeGaps:
    def test_basic(self):
        gaps = prime_gaps(5)
        assert len(gaps) == 5
        for p1, p2, gap in gaps:
            assert p2 - p1 == gap
            assert is_prime(p1) and is_prime(p2)

    def test_first_gap(self):
        gaps = prime_gaps(1, 2)
        assert gaps[0] == (2, 3, 1)


class TestPrimeCounting:
    def test_pi_10(self):
        assert prime_counting_function(10) == 4  # 2,3,5,7

    def test_pi_100(self):
        assert prime_counting_function(100) == 25

    def test_pi_1000(self):
        assert prime_counting_function(1000) == 168


class TestPrimeFactorization:
    def test_basic(self):
        factors = prime_factorization(12)
        assert factors == [(2, 2), (3, 1)]

    def test_prime(self):
        factors = prime_factorization(13)
        assert factors == [(13, 1)]

    def test_power_of_2(self):
        factors = prime_factorization(64)
        assert factors == [(2, 6)]

    def test_large_number(self):
        factors = prime_factorization(360)
        product = 1
        for p, e in factors:
            product *= p ** e
        assert product == 360


class TestPrimeGeneratorFactory:
    def test_create_sieve(self):
        sieve = PrimeGeneratorFactory.sieve(100)
        assert len(sieve.primes) == 25

    def test_generate_primes(self):
        primes = PrimeGeneratorFactory.generate_primes(10, 2)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_is_prime_miller_rabin(self):
        assert PrimeGeneratorFactory.is_prime(97, "miller_rabin")
        assert not PrimeGeneratorFactory.is_prime(100, "miller_rabin")

    def test_generate_sieve_algorithm(self):
        primes = PrimeGeneratorFactory.generate_primes(20, 2, "sieve")
        assert len(primes) == 20
        assert all(is_prime(p) for p in primes)
