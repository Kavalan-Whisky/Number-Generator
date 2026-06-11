"""Tests for numbertheory_generator module."""
import pytest
from src.core.generators.numbertheory_generator import (
    is_perfect, perfect_numbers,
    amicable_pairs,
    is_happy, happy_numbers,
    is_narcissistic, narcissistic_numbers,
    is_kaprekar, kaprekar_numbers,
    is_harshad, harshad_numbers,
    is_palindromic, palindromic_numbers,
    is_automorphic, automorphic_numbers,
    is_vampire, vampire_numbers,
    is_armstrong, armstrong_numbers,
    NumberTheoryFactory,
)


class TestPerfectNumbers:
    def test_known_perfects(self):
        assert is_perfect(6)
        assert is_perfect(28)
        assert is_perfect(496)

    def test_non_perfect(self):
        assert not is_perfect(10)
        assert not is_perfect(100)
        assert not is_perfect(1)

    def test_generate_list(self):
        ps = perfect_numbers(1000)
        assert 6 in ps
        assert 28 in ps
        assert 496 in ps

    def test_generate_count(self):
        ps = perfect_numbers(500)
        assert all(is_perfect(p) for p in ps)


class TestAmicablePairs:
    def test_known_pair(self):
        pairs = amicable_pairs(1000)
        assert (220, 284) in pairs

    def test_symmetric(self):
        pairs = amicable_pairs(2000)
        for a, b in pairs:
            assert (b, a) in pairs or a < b  # either order is fine


class TestHappyNumbers:
    def test_known_happy(self):
        assert is_happy(1)
        assert is_happy(7)
        assert is_happy(19)
        assert is_happy(100)

    def test_known_sad(self):
        assert not is_happy(2)
        assert not is_happy(3)
        assert not is_happy(4)

    def test_generate(self):
        happy = happy_numbers(50)
        assert 1 in happy
        assert 7 in happy
        assert all(is_happy(n) for n in happy)


class TestNarcissisticNumbers:
    def test_known(self):
        assert is_narcissistic(1)
        assert is_narcissistic(153)  # 1^3+5^3+3^3=153
        assert is_narcissistic(370)
        assert is_narcissistic(371)
        assert is_narcissistic(407)

    def test_non(self):
        assert not is_narcissistic(10)
        assert not is_narcissistic(100)

    def test_generate(self):
        narcs = narcissistic_numbers(15)
        assert 153 in narcs
        assert all(is_narcissistic(n) for n in narcs)


class TestKaprekarNumbers:
    def test_known(self):
        assert is_kaprekar(9)
        assert is_kaprekar(45)
        assert is_kaprekar(99)
        assert is_kaprekar(297)

    def test_generate(self):
        # Only request first 5 Kaprekar numbers — more would scan very far
        kaps = kaprekar_numbers(5)
        assert 9 in kaps
        assert 45 in kaps


class TestHarshadNumbers:
    def test_known(self):
        assert is_harshad(1)
        assert is_harshad(12)  # 12/(1+2)=4
        assert is_harshad(18)
        assert is_harshad(21)

    def test_non(self):
        assert not is_harshad(19)
        assert not is_harshad(23)

    def test_generate(self):
        hs = harshad_numbers(50)
        assert 12 in hs
        assert all(is_harshad(n) for n in hs)


class TestPalindromicNumbers:
    def test_known(self):
        assert is_palindromic(121)
        assert is_palindromic(1221)
        assert is_palindromic(5)
        assert is_palindromic(11)

    def test_non(self):
        assert not is_palindromic(12)
        assert not is_palindromic(100)

    def test_generate(self):
        pals = palindromic_numbers(150)
        assert 121 in pals
        assert all(is_palindromic(n) for n in pals)


class TestAutomorphicNumbers:
    def test_known(self):
        assert is_automorphic(5)   # 5^2=25, ends in 5
        assert is_automorphic(6)   # 6^2=36, ends in 6
        assert is_automorphic(25)  # 25^2=625, ends in 25
        assert is_automorphic(76)  # 76^2=5776, ends in 76

    def test_non(self):
        assert not is_automorphic(7)
        assert not is_automorphic(10)

    def test_generate(self):
        autos = automorphic_numbers(4)
        assert 5 in autos
        assert 6 in autos


class TestVampireNumbers:
    def test_known(self):
        # 1260 = 21*60, 1395 = 15*93
        assert is_vampire(1260)
        assert is_vampire(1395)

    def test_non(self):
        assert not is_vampire(1234)
        assert not is_vampire(1000)

    def test_generate_nonempty(self):
        vs = vampire_numbers(5)
        assert len(vs) > 0
        assert all(is_vampire(v) for v in vs)


class TestArmstrongNumbers:
    def test_known(self):
        assert is_armstrong(1)
        assert is_armstrong(153)
        assert is_armstrong(370)
        assert is_armstrong(371)
        assert is_armstrong(407)

    def test_non(self):
        assert not is_armstrong(10)
        assert not is_armstrong(100)

    def test_generate(self):
        arms = armstrong_numbers(15)
        assert 153 in arms


class TestNumberTheoryFactory:
    def test_happy_via_factory(self):
        happy = NumberTheoryFactory.generate("happy", 10)
        assert 1 in happy

    def test_perfect_via_factory(self):
        ps = NumberTheoryFactory.generate("perfect", 2)
        assert 6 in ps

    def test_harshad_via_factory(self):
        hs = NumberTheoryFactory.generate("harshad", 10)
        assert len(hs) > 0

    def test_available_types(self):
        types = NumberTheoryFactory.available_types()
        assert "perfect" in types
        assert "happy" in types
