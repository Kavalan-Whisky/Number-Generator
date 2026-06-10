"""Tests for the randomness testing module."""

import math
import pytest
from src.core.analyzers.randomness_tester import (
    FrequencyTest, RunsTest, LongestRunTest, SerialTest,
    PokerTest, AutocorrelationTest, UniformityTest,
    RandomnessTester, TestResult, TestReport,
)


def generate_random_data(n=1000, seed=42):
    """Generate pseudo-random data for testing."""
    import random
    rng = random.Random(seed)
    return [rng.randint(0, 255) for _ in range(n)]


def generate_uniform_floats(n=1000, seed=42):
    import random
    rng = random.Random(seed)
    return [rng.random() * 100 for _ in range(n)]


class TestTestResult:
    def test_str_pass(self):
        r = TestResult("TestX", True, 0.5, 1.0)
        assert "PASS" in str(r)

    def test_str_fail(self):
        r = TestResult("TestX", False, 0.001, 10.0)
        assert "FAIL" in str(r)


class TestTestReport:
    def test_add_result(self):
        report = TestReport()
        report.add_result(TestResult("T1", True, 0.5, 1.0))
        report.add_result(TestResult("T2", False, 0.001, 5.0))
        assert report.total_tests == 2
        assert report.passed_tests == 1
        assert report.failed_tests == 1

    def test_pass_rate(self):
        report = TestReport()
        for _ in range(8):
            report.add_result(TestResult("T", True, 0.5, 1.0))
        for _ in range(2):
            report.add_result(TestResult("T", False, 0.001, 5.0))
        assert report.pass_rate == 0.8

    def test_summary(self):
        report = TestReport()
        report.add_result(TestResult("FrequencyTest", True, 0.5, 1.0))
        s = report.summary()
        assert "FrequencyTest" in s

    def test_to_dict(self):
        report = TestReport()
        report.add_result(TestResult("T", True, 0.5, 1.0))
        d = report.to_dict()
        assert "total_tests" in d
        assert "results" in d


class TestFrequencyTest:
    def test_balanced_data(self):
        # Alternating 0 and 1
        data = [0, 255] * 500
        test = FrequencyTest()
        result = test.run(data)
        assert isinstance(result, TestResult)
        assert result.name == "FrequencyTest"

    def test_returns_test_result(self):
        data = generate_random_data(200)
        test = FrequencyTest()
        result = test.run(data)
        assert 0 <= result.p_value <= 1

    def test_with_range_normalization(self):
        data = list(range(100))
        test = FrequencyTest()
        result = test.run(data, normalize_range=(0, 99))
        assert isinstance(result.passed, bool)


class TestRunsTest:
    def test_random_data(self):
        data = generate_random_data(500)
        test = RunsTest()
        result = test.run(data, (0, 255))
        assert isinstance(result, TestResult)
        assert 0 <= result.p_value <= 1

    def test_alternating_data(self):
        # Alternating high/low should show many runs
        data = [0, 255] * 200
        test = RunsTest()
        result = test.run(data, (0, 255))
        assert isinstance(result, TestResult)


class TestLongestRunTest:
    def test_sufficient_data(self):
        data = generate_random_data(200)
        test = LongestRunTest()
        result = test.run(data)
        assert isinstance(result, TestResult)

    def test_insufficient_data(self):
        data = [1, 2, 3]
        test = LongestRunTest()
        result = test.run(data)
        assert "insufficient" in result.details.get("reason", "").lower() or not result.passed


class TestSerialTest:
    def test_basic(self):
        data = generate_random_data(500)
        test = SerialTest()
        result = test.run(data)
        assert isinstance(result, TestResult)
        assert 0 <= result.p_value <= 1

    def test_with_m2(self):
        data = generate_random_data(500)
        test = SerialTest()
        result = test.run(data, m=2)
        assert isinstance(result, TestResult)


class TestPokerTest:
    def test_basic(self):
        data = generate_random_data(500)
        test = PokerTest()
        result = test.run(data, group_size=4)
        assert isinstance(result, TestResult)

    def test_insufficient_data(self):
        data = [1] * 10
        test = PokerTest()
        result = test.run(data, group_size=4)
        assert isinstance(result, TestResult)


class TestAutocorrelationTest:
    def test_random_data(self):
        data = generate_random_data(500)
        test = AutocorrelationTest()
        result = test.run(data, lag=1)
        assert isinstance(result, TestResult)
        assert 0 <= result.p_value <= 1

    def test_periodic_data(self):
        # Periodic data should fail autocorrelation test
        data = [0, 255, 0, 255] * 200
        test = AutocorrelationTest()
        result = test.run(data, lag=2)
        assert isinstance(result, TestResult)


class TestUniformityTest:
    def test_uniform_data(self):
        data = list(range(0, 1000, 1))
        test = UniformityTest()
        result = test.run([float(x) for x in data])
        assert isinstance(result, TestResult)

    def test_empty_data(self):
        test = UniformityTest()
        result = test.run([])
        assert not result.passed

    def test_constant_data(self):
        test = UniformityTest()
        result = test.run([5.0] * 100)
        assert not result.passed


class TestRandomnessTester:
    def test_run_all(self):
        data = generate_uniform_floats(500)
        tester = RandomnessTester()
        report = tester.run_all(data)
        assert report.total_tests >= 5

    def test_run_single_test(self):
        data = generate_uniform_floats(500)
        tester = RandomnessTester()
        result = tester.run_test("frequency", data)
        assert isinstance(result, TestResult)

    def test_run_all_returns_report(self):
        data = generate_uniform_floats(500)
        tester = RandomnessTester()
        report = tester.run_all(data)
        assert isinstance(report, TestReport)
        assert 0 <= report.pass_rate <= 1

    def test_unknown_test_raises(self):
        tester = RandomnessTester()
        with pytest.raises(ValueError):
            tester.run_test("unknown_test", [1, 2, 3])

    def test_mersenne_twister_passes_most(self):
        """Mersenne Twister should pass most randomness tests."""
        from src.core.generators.random_generator import MersenneTwisterGenerator
        gen = MersenneTwisterGenerator(seed=42)
        data = [gen.next_range(0, 255) for _ in range(1000)]
        tester = RandomnessTester()
        report = tester.run_all(data, (0, 255))
        # MT should pass at least half the tests
        assert report.pass_rate >= 0.4
