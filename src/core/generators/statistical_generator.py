"""
Statistical Distribution Generators using only the math module (no numpy).
Implements Box-Muller, Knuth Poisson, and other sampling algorithms from scratch.
"""

import math
import random
import time
from typing import List, Optional, Tuple


class _BaseDistribution:
    """Base class for all statistical distributions."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed if seed is not None else int(time.time() * 1000))

    def _uniform(self) -> float:
        """Generate a uniform [0, 1) random number."""
        return self._rng.random()

    def _uniform_nonzero(self) -> float:
        """Generate a uniform (0, 1) random number."""
        x = 0.0
        while x == 0.0:
            x = self._rng.random()
        return x

    def sample(self) -> float:
        raise NotImplementedError

    def generate(self, count: int) -> List[float]:
        return [self.sample() for _ in range(count)]

    def mean(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def std_dev(self) -> float:
        return math.sqrt(self.variance())


class NormalDistribution(_BaseDistribution):
    """
    Normal (Gaussian) distribution using Box-Muller transform.
    N(mu, sigma^2)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._spare: Optional[float] = None

    def sample(self) -> float:
        """Box-Muller transform to generate normal samples."""
        if self._spare is not None:
            val = self._spare
            self._spare = None
            return self.mu + self.sigma * val

        u1 = self._uniform_nonzero()
        u2 = self._uniform()
        mag = math.sqrt(-2.0 * math.log(u1))
        z0 = mag * math.cos(2.0 * math.pi * u2)
        z1 = mag * math.sin(2.0 * math.pi * u2)
        self._spare = z1
        return self.mu + self.sigma * z0

    def pdf(self, x: float) -> float:
        """Probability density function."""
        return math.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (
            self.sigma * math.sqrt(2 * math.pi)
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        return 0.5 * (1 + math.erf((x - self.mu) / (self.sigma * math.sqrt(2))))

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma**2


class PoissonDistribution(_BaseDistribution):
    """
    Poisson distribution using Knuth's algorithm.
    Models count of events in a fixed interval.
    """

    def __init__(self, lam: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if lam <= 0:
            raise ValueError("lambda must be positive")
        self.lam = lam

    def sample(self) -> int:
        """Knuth's algorithm for Poisson sampling."""
        L = math.exp(-self.lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self._uniform_nonzero()
        return k - 1

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        return (self.lam**k * math.exp(-self.lam)) / math.factorial(k)

    def mean(self) -> float:
        return self.lam

    def variance(self) -> float:
        return self.lam


class ExponentialDistribution(_BaseDistribution):
    """
    Exponential distribution using inverse transform sampling.
    Models time between events in Poisson process.
    """

    def __init__(self, rate: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if rate <= 0:
            raise ValueError("rate (lambda) must be positive")
        self.rate = rate

    def sample(self) -> float:
        """Inverse transform: -ln(U) / lambda."""
        return -math.log(self._uniform_nonzero()) / self.rate

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return self.rate * math.exp(-self.rate * x)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.rate * x)

    def mean(self) -> float:
        return 1.0 / self.rate

    def variance(self) -> float:
        return 1.0 / (self.rate**2)


class GammaDistribution(_BaseDistribution):
    """
    Gamma distribution using Marsaglia-Tsang method.
    Gamma(shape=alpha, rate=beta)
    """

    def __init__(self, shape: float = 1.0, rate: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if shape <= 0 or rate <= 0:
            raise ValueError("shape and rate must be positive")
        self.shape = shape
        self.rate = rate
        self._normal = NormalDistribution(0, 1, seed)

    def _sample_gamma_marsaglia(self, alpha: float) -> float:
        """Marsaglia-Tsang algorithm for alpha >= 1."""
        d = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self._normal.sample()
            v = 1.0 + c * x
            if v <= 0:
                continue
            v = v**3
            u = self._uniform_nonzero()
            x2 = x**2
            if u < 1 - 0.0331 * (x2**2):
                return d * v
            if math.log(u) < 0.5 * x2 + d * (1 - v + math.log(v)):
                return d * v

    def sample(self) -> float:
        """Sample from Gamma distribution."""
        if self.shape < 1:
            # Boost method: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
            g = self._sample_gamma_marsaglia(self.shape + 1)
            u = self._uniform_nonzero()
            return g * (u ** (1.0 / self.shape)) / self.rate
        return self._sample_gamma_marsaglia(self.shape) / self.rate

    def mean(self) -> float:
        return self.shape / self.rate

    def variance(self) -> float:
        return self.shape / (self.rate**2)


class BetaDistribution(_BaseDistribution):
    """
    Beta distribution: Beta(alpha, beta).
    Uses ratio of Gamma samples.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta_param = beta
        self._gamma_a = GammaDistribution(alpha, 1.0, seed)
        self._gamma_b = GammaDistribution(beta, 1.0, seed)

    def sample(self) -> float:
        """Beta = Gamma(alpha) / (Gamma(alpha) + Gamma(beta))."""
        x = self._gamma_a.sample()
        y = self._gamma_b.sample()
        return x / (x + y)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    def variance(self) -> float:
        a, b = self.alpha, self.beta_param
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


class UniformDistribution(_BaseDistribution):
    """Uniform distribution over [low, high]."""

    def __init__(self, low: float = 0.0, high: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if low >= high:
            raise ValueError("low must be less than high")
        self.low = low
        self.high = high

    def sample(self) -> float:
        return self.low + self._uniform() * (self.high - self.low)

    def sample_int(self) -> int:
        return int(self.sample())

    def mean(self) -> float:
        return (self.low + self.high) / 2

    def variance(self) -> float:
        return (self.high - self.low) ** 2 / 12


class BinomialDistribution(_BaseDistribution):
    """
    Binomial distribution B(n, p).
    Uses direct Bernoulli sampling.
    """

    def __init__(self, n: int = 10, p: float = 0.5, seed: Optional[int] = None):
        super().__init__(seed)
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        if n < 1:
            raise ValueError("n must be positive")
        self.n = n
        self.p = p

    def sample(self) -> int:
        """Sum of n Bernoulli trials."""
        return sum(1 for _ in range(self.n) if self._uniform() < self.p)

    def pmf(self, k: int) -> float:
        return math.comb(self.n, k) * (self.p**k) * ((1 - self.p) ** (self.n - k))

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)


class ChiSquaredDistribution(_BaseDistribution):
    """Chi-squared distribution with k degrees of freedom."""

    def __init__(self, k: int = 1, seed: Optional[int] = None):
        super().__init__(seed)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._normal = NormalDistribution(0, 1, seed)

    def sample(self) -> float:
        """Sum of k squared standard normals."""
        return sum(self._normal.sample() ** 2 for _ in range(self.k))

    def mean(self) -> float:
        return float(self.k)

    def variance(self) -> float:
        return float(2 * self.k)


class StudentTDistribution(_BaseDistribution):
    """Student's t-distribution with nu degrees of freedom."""

    def __init__(self, nu: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        if nu <= 0:
            raise ValueError("nu must be positive")
        self.nu = nu
        self._normal = NormalDistribution(0, 1, seed)
        self._chi2 = ChiSquaredDistribution(int(max(1, nu)), seed)

    def sample(self) -> float:
        """Z / sqrt(V/nu) where Z~N(0,1), V~Chi2(nu)."""
        z = self._normal.sample()
        v = self._chi2.sample()
        return z / math.sqrt(v / self.nu)

    def mean(self) -> float:
        if self.nu > 1:
            return 0.0
        return float("nan")

    def variance(self) -> float:
        if self.nu > 2:
            return self.nu / (self.nu - 2)
        if self.nu > 1:
            return float("inf")
        return float("nan")


class MultivariateNormal:
    """
    Multivariate normal distribution sampler using Cholesky decomposition.
    """

    def __init__(self, mean: List[float], cov: List[List[float]], seed: Optional[int] = None):
        self.mean = mean
        self.dim = len(mean)
        self.cov = cov
        self._chol = self._cholesky(cov)
        self._normal = NormalDistribution(0, 1, seed)

    def _cholesky(self, A: List[List[float]]) -> List[List[float]]:
        """Cholesky decomposition L such that A = L * L^T."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    if val < 0:
                        val = 0.0
                    L[i][j] = math.sqrt(val)
                else:
                    if L[j][j] == 0:
                        L[i][j] = 0.0
                    else:
                        L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    def sample(self) -> List[float]:
        """Generate a sample from the multivariate normal."""
        z = [self._normal.sample() for _ in range(self.dim)]
        x = []
        for i in range(self.dim):
            val = self.mean[i] + sum(self._chol[i][j] * z[j] for j in range(i + 1))
            x.append(val)
        return x

    def generate(self, count: int) -> List[List[float]]:
        return [self.sample() for _ in range(count)]


class StatisticalGeneratorFactory:
    """Factory for creating statistical distribution generators."""

    DISTRIBUTIONS = {
        "normal": NormalDistribution,
        "gaussian": NormalDistribution,
        "poisson": PoissonDistribution,
        "exponential": ExponentialDistribution,
        "gamma": GammaDistribution,
        "beta": BetaDistribution,
        "uniform": UniformDistribution,
        "binomial": BinomialDistribution,
        "chi_squared": ChiSquaredDistribution,
        "student_t": StudentTDistribution,
    }

    @classmethod
    def create(cls, distribution: str, seed: Optional[int] = None, **params):
        """Create a distribution generator."""
        dist = distribution.lower().replace("-", "_")
        if dist not in cls.DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {distribution}. Available: {list(cls.DISTRIBUTIONS.keys())}")
        dist_class = cls.DISTRIBUTIONS[dist]
        return dist_class(seed=seed, **params)

    @classmethod
    def available_distributions(cls) -> List[str]:
        return list(cls.DISTRIBUTIONS.keys())
