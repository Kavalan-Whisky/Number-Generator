"""
Benchmark script for comparing performance of all number generators.
Measures throughput (numbers/second) and timing statistics.
"""

import time
import math
import statistics
from typing import Callable, Dict, List, Tuple


def benchmark_function(func: Callable, n_calls: int = 10000) -> Dict:
    """Benchmark a callable and return timing statistics."""
    # Warmup
    for _ in range(100):
        func()

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(n_calls):
        func()
    elapsed = time.perf_counter() - start

    return {
        "total_time": elapsed,
        "calls": n_calls,
        "throughput": n_calls / elapsed,
        "avg_ns": elapsed / n_calls * 1e9,
    }


def benchmark_random_generators(n: int = 100000):
    """Benchmark all random number generators."""
    print("\n" + "=" * 70)
    print("RANDOM NUMBER GENERATOR BENCHMARKS")
    print("=" * 70)
    print(f"{'Generator':<25} {'Throughput (M/s)':>18} {'Avg (ns)':>12} {'Status'}")
    print("-" * 70)

    from src.core.generators.random_generator import (
        LinearCongruentialGenerator, Xorshift32Generator, Xorshift64Generator,
        PCGGenerator, LFSRGenerator, BlumBlumShubGenerator,
        MiddleSquareGenerator, MersenneTwisterGenerator,
    )

    generators = {
        "LCG": LinearCongruentialGenerator(seed=42),
        "Xorshift-32": Xorshift32Generator(seed=42),
        "Xorshift-64": Xorshift64Generator(seed=42),
        "PCG": PCGGenerator(seed=42),
        "LFSR-32": LFSRGenerator(seed=42, bits=32),
        "Blum-Blum-Shub": BlumBlumShubGenerator(p=11, q=23, seed=42),
        "Middle-Square": MiddleSquareGenerator(seed=1234, digits=8),
        "Mersenne Twister": MersenneTwisterGenerator(seed=42),
    }

    results = {}
    for name, gen in generators.items():
        try:
            stats = benchmark_function(gen.next_int, n)
            throughput_m = stats["throughput"] / 1e6
            avg_ns = stats["avg_ns"]
            print(f"{name:<25} {throughput_m:>15.2f} M/s {avg_ns:>10.1f} ns  OK")
            results[name] = stats
        except Exception as e:
            print(f"{name:<25} {'ERROR':>18}  {str(e)[:20]}")

    return results


def benchmark_prime_generators(n: int = 1000):
    """Benchmark prime number generation."""
    print("\n" + "=" * 70)
    print("PRIME GENERATOR BENCHMARKS")
    print("=" * 70)
    print(f"{'Method':<30} {'Time (ms)':>12} {'Primes/s':>12}")
    print("-" * 70)

    from src.core.generators.prime_generator import (
        SieveOfEratosthenes, SegmentedSieve, SundaramSieve
    )

    # Sieve of Eratosthenes
    for limit in [1000, 10000, 100000]:
        start = time.perf_counter()
        sieve = SieveOfEratosthenes(limit)
        _ = sieve.primes
        elapsed = time.perf_counter() - start
        count = len(sieve.primes)
        print(f"{'Sieve(limit='+str(limit)+')':<30} {elapsed*1000:>10.2f} ms {count/elapsed:>10.0f}/s")

    # Segmented Sieve
    for limit in [10000, 100000]:
        start = time.perf_counter()
        sieve = SegmentedSieve()
        primes = sieve.primes_up_to(limit)
        elapsed = time.perf_counter() - start
        print(f"{'SegmentedSieve('+str(limit)+')':<30} {elapsed*1000:>10.2f} ms {len(primes)/elapsed:>10.0f}/s")

    # Miller-Rabin test
    from src.core.generators.prime_generator import miller_rabin_is_prime
    candidates = [random_candidate() for _ in range(100)]
    start = time.perf_counter()
    for c in candidates:
        miller_rabin_is_prime(c)
    elapsed = time.perf_counter() - start
    print(f"{'Miller-Rabin (100 tests)':<30} {elapsed*1000:>10.2f} ms {100/elapsed:>10.0f}/s")


def random_candidate():
    """Generate a random candidate for primality testing."""
    import secrets
    return secrets.randbits(64) | 1  # Force odd


def benchmark_fibonacci(n: int = 1000):
    """Benchmark Fibonacci computation methods."""
    print("\n" + "=" * 70)
    print("FIBONACCI BENCHMARKS")
    print("=" * 70)
    print(f"{'Method':<30} {'F(n=100)':<15} {'Time (ms)':>10} {'N/s':>12}")
    print("-" * 70)

    from src.core.generators.fibonacci_generator import FibonacciGenerator
    gen = FibonacciGenerator()

    for method in ["iterative", "matrix", "closed_form"]:
        start = time.perf_counter()
        for _ in range(n):
            gen.generate(100, method)
        elapsed = time.perf_counter() - start
        f_100 = gen.generate(1, method, 100)[0]
        print(f"{method:<30} {str(f_100)[:14]:<15} {elapsed*1000:>8.2f} ms {n/elapsed:>10.0f}/s")


def benchmark_sequence_generators():
    """Benchmark various sequence generators."""
    print("\n" + "=" * 70)
    print("SEQUENCE GENERATOR BENCHMARKS")
    print("=" * 70)
    print(f"{'Sequence':<25} {'Count':>8} {'Time (ms)':>12} {'Throughput':>12}")
    print("-" * 70)

    from src.core.generators.sequence_generator import (
        CatalanNumbers, BellNumbers, CollatzSequence, RecamanSequence,
        TriangularNumbers, SequenceGeneratorFactory
    )

    benchmarks = [
        ("Catalan(n=20)", lambda: CatalanNumbers.generate(20)),
        ("Bell(n=15)", lambda: BellNumbers.generate(15)),
        ("Triangular(n=100)", lambda: TriangularNumbers.generate(100)),
        ("Recaman(n=100)", lambda: RecamanSequence.generate(100)),
        ("Collatz(n=27)", lambda: CollatzSequence.sequence(27)),
    ]

    for name, func in benchmarks:
        n_runs = 1000
        start = time.perf_counter()
        for _ in range(n_runs):
            result = func()
        elapsed = time.perf_counter() - start
        count = len(result)
        print(f"{name:<25} {count:>8} {elapsed*1000:>10.2f} ms {n_runs/elapsed:>10.0f}/s")


def benchmark_statistical_generators(n: int = 10000):
    """Benchmark statistical distribution samplers."""
    print("\n" + "=" * 70)
    print("STATISTICAL DISTRIBUTION BENCHMARKS")
    print("=" * 70)
    print(f"{'Distribution':<25} {'Throughput (K/s)':>18} {'Avg (µs)':>12}")
    print("-" * 70)

    from src.core.generators.statistical_generator import (
        NormalDistribution, PoissonDistribution, ExponentialDistribution,
        GammaDistribution, BetaDistribution, UniformDistribution,
    )

    distributions = {
        "Normal(0, 1)": NormalDistribution(0, 1, seed=42),
        "Poisson(5)": PoissonDistribution(5, seed=42),
        "Exponential(1)": ExponentialDistribution(1, seed=42),
        "Gamma(2, 1)": GammaDistribution(2, 1, seed=42),
        "Beta(2, 3)": BetaDistribution(2, 3, seed=42),
        "Uniform(0, 1)": UniformDistribution(0, 1, seed=42),
    }

    for name, dist in distributions.items():
        stats = benchmark_function(dist.sample, n)
        throughput_k = stats["throughput"] / 1e3
        avg_us = stats["avg_ns"] / 1000
        print(f"{name:<25} {throughput_k:>15.2f} K/s {avg_us:>10.2f} µs")


def benchmark_analyzers(data_size: int = 1000):
    """Benchmark statistical analyzers."""
    print("\n" + "=" * 70)
    print("ANALYZER BENCHMARKS")
    print("=" * 70)

    import random
    rng = random.Random(42)
    data = [rng.gauss(0, 1) for _ in range(data_size)]

    # Statistical analysis
    from src.core.analyzers.statistical_analyzer import describe
    start = time.perf_counter()
    for _ in range(100):
        describe(data)
    elapsed = time.perf_counter() - start
    print(f"Statistical describe (n={data_size}): {elapsed/100*1000:.2f} ms/call")

    # Pattern detection
    from src.core.analyzers.pattern_detector import PatternDetector
    detector = PatternDetector()
    start = time.perf_counter()
    for _ in range(10):
        detector.analyze(data[:100])
    elapsed = time.perf_counter() - start
    print(f"Pattern detection (n=100): {elapsed/10*1000:.2f} ms/call")


def run_all_benchmarks():
    """Run all benchmarks and produce a summary report."""
    print("\n" + "█" * 70)
    print("█ NUMBER GENERATOR PERFORMANCE BENCHMARK")
    print("█" * 70)
    print(f"Python implementation benchmarks\n")

    import sys
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}\n")

    benchmark_random_generators(100000)
    benchmark_prime_generators()
    benchmark_fibonacci(500)
    benchmark_sequence_generators()
    benchmark_statistical_generators(10000)
    benchmark_analyzers(1000)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
