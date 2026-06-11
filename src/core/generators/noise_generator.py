"""
Noise sequence generators (pure Python).
1D Perlin noise, value noise, white/pink/brown noise, random walk, Brownian bridge.
"""

import math
import random
from typing import List


def _fade(t: float) -> float:
    """Perlin's quintic fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


class PerlinNoise1D:
    """Classic 1D Perlin (gradient) noise."""

    def __init__(self, seed: int = None):
        rng = random.Random(seed)
        perm = list(range(256))
        rng.shuffle(perm)
        self.perm = perm + perm  # avoid wrapping
        # Gradients in {-1..1} scaled
        self.grads = [rng.uniform(-1, 1) for _ in range(256)]

    def noise(self, x: float) -> float:
        """Noise value at x, roughly in [-1, 1]."""
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        g0 = self.grads[self.perm[xi] & 255]
        g1 = self.grads[self.perm[xi + 1] & 255]
        return _lerp(g0 * xf, g1 * (xf - 1.0), u)

    def generate(self, count: int, step: float = 0.1, octaves: int = 1,
                 persistence: float = 0.5) -> List[float]:
        """Sample fractal Perlin noise at regular intervals."""
        out = []
        for i in range(count):
            x = i * step
            total, amp, freq, max_amp = 0.0, 1.0, 1.0, 0.0
            for _ in range(max(1, octaves)):
                total += self.noise(x * freq) * amp
                max_amp += amp
                amp *= persistence
                freq *= 2.0
            out.append(total / max_amp)
        return out


class ValueNoise1D:
    """1D value noise: smooth interpolation between random lattice values."""

    def __init__(self, seed: int = None):
        rng = random.Random(seed)
        self.values = [rng.uniform(-1, 1) for _ in range(256)]

    def noise(self, x: float) -> float:
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        return _lerp(self.values[xi], self.values[(xi + 1) & 255], u)

    def generate(self, count: int, step: float = 0.1) -> List[float]:
        return [self.noise(i * step) for i in range(count)]


class WhiteNoise:
    """Uncorrelated uniform white noise in [-1, 1]."""

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def generate(self, count: int) -> List[float]:
        return [self.rng.uniform(-1, 1) for _ in range(count)]


class PinkNoise:
    """Pink (1/f) noise via the Voss-McCartney algorithm."""

    def __init__(self, seed: int = None, num_sources: int = 16):
        self.rng = random.Random(seed)
        self.num_sources = num_sources
        self.sources = [self.rng.uniform(-1, 1) for _ in range(num_sources)]
        self.counter = 0

    def next(self) -> float:
        self.counter += 1
        c = self.counter
        # Update source k when bit k flips (trailing zeros)
        k = 0
        while c % 2 == 0 and k < self.num_sources - 1:
            c //= 2
            k += 1
        self.sources[k] = self.rng.uniform(-1, 1)
        return sum(self.sources) / self.num_sources

    def generate(self, count: int) -> List[float]:
        return [self.next() for _ in range(count)]


class BrownNoise:
    """Brown (red) noise: integrated white noise with soft clipping."""

    def __init__(self, seed: int = None, step_scale: float = 0.1):
        self.rng = random.Random(seed)
        self.step_scale = step_scale
        self.value = 0.0

    def next(self) -> float:
        self.value += self.rng.uniform(-1, 1) * self.step_scale
        # Reflect to stay within [-1, 1]
        if self.value > 1.0:
            self.value = 2.0 - self.value
        elif self.value < -1.0:
            self.value = -2.0 - self.value
        return self.value

    def generate(self, count: int) -> List[float]:
        return [self.next() for _ in range(count)]


class RandomWalk:
    """Simple random walk with configurable step distribution."""

    def __init__(self, seed: int = None, start: float = 0.0,
                 step_size: float = 1.0, gaussian: bool = False):
        self.rng = random.Random(seed)
        self.position = start
        self.step_size = step_size
        self.gaussian = gaussian

    def step(self) -> float:
        if self.gaussian:
            self.position += self.rng.gauss(0, self.step_size)
        else:
            self.position += self.rng.choice((-1, 1)) * self.step_size
        return self.position

    def generate(self, count: int) -> List[float]:
        return [self.step() for _ in range(count)]


class BrownianBridge:
    """Brownian bridge from `start` to `end` over `count` points.

    Constructed by conditioning a Brownian motion: B(t) - t*B(1).
    """

    def __init__(self, seed: int = None, start: float = 0.0, end: float = 0.0,
                 sigma: float = 1.0):
        self.rng = random.Random(seed)
        self.start = start
        self.end = end
        self.sigma = sigma

    def generate(self, count: int) -> List[float]:
        if count <= 0:
            return []
        if count == 1:
            return [self.start]
        n = count - 1
        # Brownian motion path
        w = [0.0]
        dt = 1.0 / n
        for _ in range(n):
            w.append(w[-1] + self.rng.gauss(0, math.sqrt(dt)) * self.sigma)
        # Bridge: pin endpoints
        out = []
        for i in range(count):
            t = i / n
            bridge = w[i] - t * w[-1]
            out.append(self.start + (self.end - self.start) * t + bridge)
        return out


class NoiseGeneratorFactory:
    """Factory for noise generators by name."""

    TYPES = ("perlin", "value", "white", "pink", "brown", "walk", "bridge")

    @staticmethod
    def available_types() -> List[str]:
        return list(NoiseGeneratorFactory.TYPES)

    @staticmethod
    def generate(noise_type: str, count: int, seed: int = None, **kwargs) -> List[float]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if noise_type == "perlin":
            return PerlinNoise1D(seed).generate(count, **kwargs)
        if noise_type == "value":
            return ValueNoise1D(seed).generate(count, **kwargs)
        if noise_type == "white":
            return WhiteNoise(seed).generate(count)
        if noise_type == "pink":
            return PinkNoise(seed).generate(count)
        if noise_type == "brown":
            return BrownNoise(seed).generate(count)
        if noise_type == "walk":
            return RandomWalk(seed, **kwargs).generate(count)
        if noise_type == "bridge":
            return BrownianBridge(seed, **kwargs).generate(count)
        raise ValueError(f"Unknown noise type: {noise_type}")
