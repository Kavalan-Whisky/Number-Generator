"""Tests for chaos_generator module."""
import pytest
import math
from src.core.generators.chaos_generator import (
    LogisticMap, TentMap, HenonMap, LorenzSystem, ArnoldCatMap, ChaosPRNG
)


class TestLogisticMap:
    def test_initial_value(self):
        lm = LogisticMap(r=3.99, x0=0.5)
        v = lm.step()
        assert 0 < v < 1

    def test_iterate_count(self):
        lm = LogisticMap(r=3.5, x0=0.3)
        it = lm.iterate()
        vals = [next(it) for _ in range(20)]
        assert len(vals) == 20

    def test_all_in_range(self):
        lm = LogisticMap(r=3.99, x0=0.5)
        it = lm.iterate()
        for _ in range(100):
            v = next(it)
            assert 0 <= v <= 1, f"Out of range: {v}"

    def test_invalid_x0_zero(self):
        with pytest.raises((ValueError, Exception)):
            LogisticMap(r=3.5, x0=0.0)

    def test_invalid_r(self):
        with pytest.raises((ValueError, Exception)):
            LogisticMap(r=5.0, x0=0.5)

    def test_determinism(self):
        lm1 = LogisticMap(r=3.7, x0=0.4)
        lm2 = LogisticMap(r=3.7, x0=0.4)
        for _ in range(50):
            assert lm1.step() == lm2.step()

    def test_sensitivity(self):
        lm1 = LogisticMap(r=3.99, x0=0.5)
        lm2 = LogisticMap(r=3.99, x0=0.5001)
        for _ in range(50):
            lm1.step()
            lm2.step()
        v1, v2 = lm1.step(), lm2.step()
        # chaotic — should diverge
        assert abs(v1 - v2) > 1e-6

    def test_generate_returns_list(self):
        lm = LogisticMap(r=3.9, x0=0.4)
        vals = lm.generate(10)
        assert len(vals) == 10
        assert all(isinstance(v, float) for v in vals)


class TestTentMap:
    def test_basic(self):
        tm = TentMap(mu=1.9, x0=0.4)
        v = tm.step()
        assert 0 <= v <= 1

    def test_generate(self):
        tm = TentMap(mu=1.8, x0=0.3)
        vals = tm.generate(50)
        assert all(0 <= v <= 1 for v in vals)

    def test_determinism(self):
        t1 = TentMap(mu=1.9, x0=0.25)
        t2 = TentMap(mu=1.9, x0=0.25)
        for _ in range(20):
            assert t1.step() == t2.step()


class TestHenonMap:
    def test_basic(self):
        hm = HenonMap()
        pt = hm.step()
        assert len(pt) == 2

    def test_generate_more(self):
        hm = HenonMap()
        pts = hm.generate(30)
        assert len(pts) == 30
        for p in pts:
            assert len(p) == 2

    def test_determinism(self):
        hm1 = HenonMap(a=1.4, b=0.3, x0=0.0, y0=0.0)
        hm2 = HenonMap(a=1.4, b=0.3, x0=0.0, y0=0.0)
        for _ in range(10):
            assert hm1.step() == hm2.step()

    def test_generate(self):
        hm = HenonMap()
        pts = hm.generate(20)
        assert len(pts) == 20


class TestLorenzSystem:
    def test_step_returns_triple(self):
        ls = LorenzSystem()
        state = ls.step()
        assert len(state) == 3

    def test_generate(self):
        ls = LorenzSystem()
        pts = ls.generate(20)
        assert len(pts) == 20
        assert all(len(p) == 3 for p in pts)

    def test_generate_x(self):
        ls = LorenzSystem()
        xs = ls.generate_x(50)
        assert len(xs) == 50
        assert all(isinstance(x, float) for x in xs)

    def test_butterfly_effect(self):
        ls1 = LorenzSystem(x0=0.1, y0=0.0, z0=0.0)
        ls2 = LorenzSystem(x0=0.2, y0=0.0, z0=0.0)  # larger initial difference
        for _ in range(1000):
            ls1.step()
            ls2.step()
        s1 = ls1.step()
        s2 = ls2.step()
        assert any(abs(s1[i] - s2[i]) > 1e-6 for i in range(3))

    def test_discard_parameter(self):
        ls = LorenzSystem()
        pts = ls.generate(10, discard=100)
        assert len(pts) == 10


class TestArnoldCatMap:
    def test_stays_in_grid(self):
        acm = ArnoldCatMap(n=8)
        pts = acm.generate(50)
        for x, y in pts:
            assert 0 <= x < 8 and 0 <= y < 8

    def test_generate(self):
        acm = ArnoldCatMap(n=6)
        pts = acm.generate(30)
        assert len(pts) == 30

    def test_determinism(self):
        acm1 = ArnoldCatMap(n=6, x0=1, y0=2)
        acm2 = ArnoldCatMap(n=6, x0=1, y0=2)
        assert acm1.generate(10) == acm2.generate(10)

    def test_integer_outputs(self):
        acm = ArnoldCatMap(n=10)
        pts = acm.generate(20)
        for x, y in pts:
            assert isinstance(x, int) and isinstance(y, int)


class TestChaosPRNG:
    def test_generate_integers(self):
        prng = ChaosPRNG(seed=42)
        nums = prng.generate(50, min_val=0, max_val=100)
        assert len(nums) == 50
        assert all(0 <= n <= 100 for n in nums)

    def test_reproducible(self):
        p1 = ChaosPRNG(seed=7)
        p2 = ChaosPRNG(seed=7)
        assert p1.generate(20) == p2.generate(20)

    def test_different_seeds(self):
        p1 = ChaosPRNG(seed=1)
        p2 = ChaosPRNG(seed=2)
        assert p1.generate(20) != p2.generate(20)

    def test_floats(self):
        prng = ChaosPRNG(seed=99)
        floats = prng.generate_floats(30)
        assert all(0 <= f < 1 for f in floats)

    def test_random_method(self):
        prng = ChaosPRNG(seed=5)
        v = prng.random()
        assert 0 <= v < 1
