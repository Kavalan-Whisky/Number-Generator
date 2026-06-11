"""Tests for noise_generator module."""
import pytest
import math
from src.core.generators.noise_generator import (
    WhiteNoise, PinkNoise, BrownNoise,
    PerlinNoise1D, ValueNoise1D, RandomWalk, BrownianBridge,
    NoiseGeneratorFactory,
)


class TestWhiteNoise:
    def test_generate_length(self):
        wn = WhiteNoise(seed=1)
        vals = wn.generate(50)
        assert len(vals) == 50

    def test_in_range(self):
        wn = WhiteNoise(seed=42)
        vals = wn.generate(100)
        assert all(-1.0 <= v <= 1.0 for v in vals)

    def test_reproducible(self):
        wn1 = WhiteNoise(seed=5)
        wn2 = WhiteNoise(seed=5)
        assert wn1.generate(20) == wn2.generate(20)

    def test_different_seeds(self):
        wn1 = WhiteNoise(seed=1)
        wn2 = WhiteNoise(seed=2)
        assert wn1.generate(20) != wn2.generate(20)


class TestPinkNoise:
    def test_generate_length(self):
        pn = PinkNoise(seed=1)
        vals = pn.generate(50)
        assert len(vals) == 50

    def test_returns_floats(self):
        pn = PinkNoise(seed=7)
        vals = pn.generate(20)
        assert all(isinstance(v, float) for v in vals)

    def test_reproducible(self):
        pn1 = PinkNoise(seed=3)
        pn2 = PinkNoise(seed=3)
        assert pn1.generate(20) == pn2.generate(20)


class TestBrownNoise:
    def test_generate_length(self):
        bn = BrownNoise(seed=1)
        vals = bn.generate(50)
        assert len(vals) == 50

    def test_first_near_zero(self):
        bn = BrownNoise(seed=0)
        vals = bn.generate(100)
        assert all(isinstance(v, float) for v in vals)

    def test_bounded_step_scale(self):
        bn = BrownNoise(seed=0, step_scale=0.01)
        vals = bn.generate(200)
        # With small steps, should stay near starting value
        assert max(abs(v) for v in vals) < 10.0


class TestPerlinNoise:
    def test_single_value(self):
        p = PerlinNoise1D(seed=1)
        v = p.noise(0.5)
        assert isinstance(v, float)

    def test_generate_length(self):
        p = PerlinNoise1D(seed=42)
        vals = p.generate(30)
        assert len(vals) == 30

    def test_smooth(self):
        p = PerlinNoise1D(seed=3)
        vals = p.generate(50, step=0.05)
        # Perlin noise should be relatively smooth
        diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        avg_diff = sum(diffs) / len(diffs)
        assert avg_diff < 1.0  # not jumping wildly

    def test_reproducible(self):
        p1 = PerlinNoise1D(seed=10)
        p2 = PerlinNoise1D(seed=10)
        assert p1.generate(20) == p2.generate(20)


class TestValueNoise:
    def test_generate_length(self):
        vn = ValueNoise1D(seed=1)
        vals = vn.generate(30)
        assert len(vals) == 30

    def test_returns_floats(self):
        vn = ValueNoise1D(seed=2)
        vals = vn.generate(20)
        assert all(isinstance(v, float) for v in vals)


class TestRandomWalk:
    def test_generate_length(self):
        rw = RandomWalk(seed=1)
        vals = rw.generate(50)
        assert len(vals) == 50

    def test_returns_floats(self):
        rw = RandomWalk(seed=0)
        vals = rw.generate(20)
        assert all(isinstance(v, float) for v in vals)

    def test_step(self):
        rw = RandomWalk(seed=5)
        v = rw.step()
        assert isinstance(v, float)

    def test_reproducible(self):
        rw1 = RandomWalk(seed=7)
        rw2 = RandomWalk(seed=7)
        assert rw1.generate(20) == rw2.generate(20)

    def test_gaussian_walk(self):
        rw = RandomWalk(seed=1, gaussian=True)
        vals = rw.generate(50)
        assert len(vals) == 50


class TestBrownianBridge:
    def test_generate_length(self):
        bb = BrownianBridge(seed=1)
        vals = bb.generate(30)
        assert len(vals) == 30

    def test_endpoints(self):
        bb = BrownianBridge(seed=2, start=0.0, end=0.0)
        vals = bb.generate(50)
        # First value should be near 0 (start)
        assert abs(vals[0]) < 2.0


class TestNoiseGeneratorFactory:
    def test_white_noise(self):
        vals = NoiseGeneratorFactory.generate("white", 30, seed=1)
        assert len(vals) == 30

    def test_pink_noise(self):
        vals = NoiseGeneratorFactory.generate("pink", 30, seed=1)
        assert len(vals) == 30

    def test_brown_noise(self):
        vals = NoiseGeneratorFactory.generate("brown", 30, seed=1)
        assert len(vals) == 30

    def test_perlin_noise(self):
        vals = NoiseGeneratorFactory.generate("perlin", 30, seed=1)
        assert len(vals) == 30

    def test_unknown_type(self):
        with pytest.raises((ValueError, KeyError)):
            NoiseGeneratorFactory.generate("unknown_noise", 10)
