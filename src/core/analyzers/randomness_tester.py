"""
Randomness Testing Suite - NIST SP 800-22 inspired tests.
Tests statistical randomness properties of number sequences.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TestResult:
    """Result of a single randomness test."""
    name: str
    passed: bool
    p_value: float
    statistic: float
    details: Dict = field(default_factory=dict)
    significance_level: float = 0.01

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name}: {status} (p={self.p_value:.4f}, stat={self.statistic:.4f})"


@dataclass
class TestReport:
    """Full randomness test report for a sequence."""
    results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    def summary(self) -> str:
        lines = ["=== Randomness Test Report ==="]
        for r in self.results:
            lines.append(str(r))
        lines.append(f"\nPassed: {self.passed_tests}/{self.total_tests} ({self.pass_rate:.1%})")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.pass_rate,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "p_value": r.p_value,
                    "statistic": r.statistic,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


def _to_bits(data: List[int]) -> List[int]:
    """Convert integer sequence to bit sequence."""
    bits = []
    for x in data:
        x = abs(x)
        for i in range(8):
            bits.append((x >> i) & 1)
    return bits


def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _erfc(x: float) -> float:
    return math.erfc(x)


class FrequencyTest:
    """
    NIST SP 800-22 Test 1: Frequency (Monobit) Test.
    Tests whether number of 0s and 1s is approximately equal.
    """

    def run(self, data: List[int], normalize_range: Optional[Tuple[int, int]] = None) -> TestResult:
        """
        Run frequency test on integer sequence.
        If normalize_range provided, maps values to bits based on median split.
        """
        if normalize_range:
            lo, hi = normalize_range
            mid = (lo + hi) / 2
            bits = [1 if x >= mid else 0 for x in data]
        else:
            bits = _to_bits(data)

        n = len(bits)
        if n == 0:
            return TestResult("FrequencyTest", False, 0.0, 0.0)

        s_n = sum(2 * b - 1 for b in bits)  # +1 or -1
        s_obs = abs(s_n) / math.sqrt(n)
        p_value = math.erfc(s_obs / math.sqrt(2))

        return TestResult(
            name="FrequencyTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=s_obs,
            details={
                "n_bits": n,
                "sum": s_n,
                "ones": sum(bits),
                "zeros": n - sum(bits),
            },
        )


class RunsTest:
    """
    NIST SP 800-22 Test 3: Runs Test.
    Tests whether number of runs is appropriate for random sequence.
    A run is a maximal sequence of identical consecutive bits.
    """

    def run(self, data: List[int], normalize_range: Optional[Tuple[int, int]] = None) -> TestResult:
        if normalize_range:
            lo, hi = normalize_range
            mid = (lo + hi) / 2
            bits = [1 if x >= mid else 0 for x in data]
        else:
            bits = _to_bits(data)

        n = len(bits)
        if n < 2:
            return TestResult("RunsTest", False, 0.0, 0.0)

        # Pre-test: proportion of ones
        pi = sum(bits) / n
        tau = 2 / math.sqrt(n)
        if abs(pi - 0.5) >= tau:
            return TestResult(
                "RunsTest",
                False,
                0.0,
                0.0,
                details={"reason": "Proportion test failed", "pi": pi},
            )

        # Count runs
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                runs += 1

        # Statistic
        numerator = abs(runs - 2 * n * pi * (1 - pi))
        denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
        if denominator == 0:
            return TestResult("RunsTest", False, 0.0, 0.0)

        p_value = math.erfc(numerator / denominator)

        return TestResult(
            name="RunsTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=runs,
            details={"n_bits": n, "runs": runs, "pi": pi},
        )


class LongestRunTest:
    """
    NIST SP 800-22 Test 4: Longest Run of Ones Test.
    Tests whether the length of longest run of ones is consistent with random.
    """

    def run(self, data: List[int], normalize_range: Optional[Tuple[int, int]] = None) -> TestResult:
        if normalize_range:
            lo, hi = normalize_range
            mid = (lo + hi) / 2
            bits = [1 if x >= mid else 0 for x in data]
        else:
            bits = _to_bits(data)

        n = len(bits)
        if n < 128:
            return TestResult("LongestRunTest", False, 0.0, 0.0,
                              details={"reason": "Need at least 128 bits"})

        # Use M=8 blocks for n >= 128
        M = 8
        K = 3
        pi = [0.2148, 0.3672, 0.2305, 0.1875]  # theoretical probabilities

        blocks = [bits[i * M:(i + 1) * M] for i in range(n // M)]
        nu = [0, 0, 0, 0]
        for block in blocks:
            # Find longest run of ones
            max_run = 0
            current_run = 0
            for b in block:
                if b == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            if max_run <= 1:
                nu[0] += 1
            elif max_run == 2:
                nu[1] += 1
            elif max_run == 3:
                nu[2] += 1
            else:
                nu[3] += 1

        N_blocks = n // M
        chi2 = sum((nu[i] - N_blocks * pi[i]) ** 2 / (N_blocks * pi[i]) for i in range(K + 1))
        p_value = _igamc(K / 2.0, chi2 / 2.0)

        return TestResult(
            name="LongestRunTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=chi2,
            details={"n_bits": n, "nu": nu},
        )


def _igamc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x)."""
    try:
        from scipy.special import gammaincc
        return float(gammaincc(a, x))
    except ImportError:
        # Simple approximation
        if x < 0:
            return 1.0
        if x == 0:
            return 1.0
        return math.exp(-x)  # rough approximation


class SerialTest:
    """
    Serial Test: Tests frequency of overlapping patterns of length m.
    """

    def run(self, data: List[int], m: int = 2) -> TestResult:
        """Run serial test on integer data."""
        bits = _to_bits(data)
        n = len(bits)
        if n < 4 * m:
            return TestResult("SerialTest", False, 0.0, 0.0,
                              details={"reason": "Insufficient data"})

        def count_patterns(bits, length):
            counts = {}
            for i in range(n):
                pattern = tuple(bits[(i + j) % n] for j in range(length))
                counts[pattern] = counts.get(pattern, 0) + 1
            return counts

        psi_m = self._psi(bits, n, m)
        psi_m1 = self._psi(bits, n, m - 1) if m > 1 else 0
        psi_m2 = self._psi(bits, n, m - 2) if m > 2 else 0

        delta1 = psi_m - psi_m1
        delta2 = psi_m - 2 * psi_m1 + psi_m2

        p_value1 = _igamc(2 ** (m - 2), delta1 / 2)
        p_value2 = _igamc(2 ** (m - 3), delta2 / 2) if m > 2 else 1.0

        p_value = min(p_value1, p_value2)
        return TestResult(
            name="SerialTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=delta1,
            details={"m": m, "psi_m": psi_m, "delta1": delta1, "delta2": delta2},
        )

    def _psi(self, bits: List[int], n: int, m: int) -> float:
        if m <= 0:
            return 0.0
        counts = {}
        for i in range(n):
            pattern = tuple(bits[(i + j) % n] for j in range(m))
            counts[pattern] = counts.get(pattern, 0) + 1
        return (2 ** m / n) * sum(c * c for c in counts.values()) - n


class PokerTest:
    """
    Poker Test: Tests frequency of patterns (like poker hands).
    Groups bits into non-overlapping groups and counts distinct patterns.
    """

    def run(self, data: List[int], group_size: int = 4) -> TestResult:
        """Run poker test."""
        bits = _to_bits(data)
        n = len(bits)
        m = group_size
        k = n // m  # number of groups

        if k < 5 * (2**m):
            return TestResult("PokerTest", False, 0.0, 0.0,
                              details={"reason": "Insufficient data for group_size"})

        counts = {}
        for i in range(k):
            group = tuple(bits[i * m:(i + 1) * m])
            counts[group] = counts.get(group, 0) + 1

        # Chi-square statistic
        expected = k / (2**m)
        chi2 = (2**m / k) * sum(c**2 for c in counts.values()) - k

        p_value = _igamc((2**m - 1) / 2.0, chi2 / 2.0)

        return TestResult(
            name="PokerTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=chi2,
            details={"group_size": m, "k": k, "distinct_patterns": len(counts)},
        )


class AutocorrelationTest:
    """
    Autocorrelation Test: Tests correlation between sequence and lagged version.
    """

    def run(self, data: List[int], lag: int = 1) -> TestResult:
        """Run autocorrelation test."""
        bits = _to_bits(data)
        n = len(bits)

        if lag >= n:
            return TestResult("AutocorrelationTest", False, 0.0, 0.0)

        # Count agreements
        A = sum(bits[i] ^ bits[i + lag] for i in range(n - lag))
        d = 2 * A - (n - lag)
        stat = abs(d) / math.sqrt(n - lag)
        p_value = math.erfc(stat / math.sqrt(2))

        return TestResult(
            name="AutocorrelationTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=stat,
            details={"lag": lag, "n_bits": n, "agreements": A},
        )


class UniformityTest:
    """
    Test that values are uniformly distributed across the range.
    Uses chi-square test with equal-width bins.
    """

    def run(self, data: List[float], bins: int = 10) -> TestResult:
        if not data:
            return TestResult("UniformityTest", False, 0.0, 0.0)

        n = len(data)
        min_v, max_v = min(data), max(data)
        if min_v == max_v:
            return TestResult("UniformityTest", False, 0.0, 0.0,
                              details={"reason": "All values equal"})

        counts = [0] * bins
        bin_width = (max_v - min_v) / bins
        for x in data:
            idx = int((x - min_v) / bin_width)
            idx = min(idx, bins - 1)
            counts[idx] += 1

        expected = n / bins
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        dof = bins - 1

        # p-value
        p_value = _igamc(dof / 2.0, chi2 / 2.0)

        return TestResult(
            name="UniformityTest",
            passed=p_value >= 0.01,
            p_value=p_value,
            statistic=chi2,
            details={"bins": bins, "counts": counts, "expected": expected},
        )


class RandomnessTester:
    """
    Run all randomness tests on a sequence and produce a comprehensive report.
    """

    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level
        self.tests = {
            "frequency": FrequencyTest(),
            "runs": RunsTest(),
            "longest_run": LongestRunTest(),
            "serial": SerialTest(),
            "poker": PokerTest(),
            "autocorrelation": AutocorrelationTest(),
            "uniformity": UniformityTest(),
        }

    def run_all(self, data: List[float], data_range: Optional[Tuple[int, int]] = None) -> TestReport:
        """Run all tests and return a comprehensive report."""
        report = TestReport()

        int_data = [int(x) for x in data]
        float_data = [float(x) for x in data]
        r = data_range or (int(min(data)), int(max(data)))

        # Frequency test
        result = self.tests["frequency"].run(int_data, r)
        report.add_result(result)

        # Runs test
        result = self.tests["runs"].run(int_data, r)
        report.add_result(result)

        # Longest run test
        result = self.tests["longest_run"].run(int_data, r)
        report.add_result(result)

        # Serial test
        result = self.tests["serial"].run(int_data)
        report.add_result(result)

        # Poker test
        result = self.tests["poker"].run(int_data)
        report.add_result(result)

        # Autocorrelation
        result = self.tests["autocorrelation"].run(int_data)
        report.add_result(result)

        # Uniformity
        result = self.tests["uniformity"].run(float_data)
        report.add_result(result)

        return report

    def run_test(self, test_name: str, data: List[float], **kwargs) -> TestResult:
        """Run a single named test."""
        if test_name not in self.tests:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(self.tests.keys())}")
        int_data = [int(x) for x in data]
        return self.tests[test_name].run(int_data, **kwargs)
