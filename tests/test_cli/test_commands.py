"""Tests for CLI commands."""

import json
import pytest
from click.testing import CliRunner
from src.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestGenerateRandom:
    def test_basic(self, runner):
        result = runner.invoke(cli, ["generate", "random", "--count", "5", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 5

    def test_with_seed(self, runner):
        result1 = runner.invoke(cli, ["generate", "random", "--seed", "42", "--count", "5", "--format", "csv"])
        result2 = runner.invoke(cli, ["generate", "random", "--seed", "42", "--count", "5", "--format", "csv"])
        assert result1.output == result2.output

    def test_range_respected(self, runner):
        result = runner.invoke(cli, ["generate", "random", "--count", "50", "--min", "10", "--max", "20", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(10 <= n <= 20 for n in data)

    def test_algorithm_lcg(self, runner):
        result = runner.invoke(cli, ["generate", "random", "--algorithm", "lcg", "--count", "5", "--format", "json"])
        assert result.exit_code == 0

    def test_csv_output(self, runner):
        result = runner.invoke(cli, ["generate", "random", "--count", "5", "--format", "csv"])
        assert result.exit_code == 0
        values = result.output.strip().split(",")
        assert len(values) == 5


class TestGeneratePrime:
    def test_basic(self, runner):
        result = runner.invoke(cli, ["generate", "prime", "--count", "5", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == [2, 3, 5, 7, 11]

    def test_with_start(self, runner):
        result = runner.invoke(cli, ["generate", "prime", "--count", "5", "--start", "10", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0] >= 10
        assert all(d > 0 for d in data)


class TestGenerateFibonacci:
    def test_basic(self, runner):
        result = runner.invoke(cli, ["generate", "fibonacci", "--count", "8", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[:5] == [0, 1, 1, 2, 3]

    def test_matrix_variant(self, runner):
        result = runner.invoke(cli, ["generate", "fibonacci", "--count", "8", "--variant", "matrix", "--format", "json"])
        assert result.exit_code == 0

    def test_lucas_type(self, runner):
        result = runner.invoke(cli, ["generate", "fibonacci", "--count", "5", "--type", "lucas", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0] == 2  # Lucas starts with 2


class TestGenerateSequence:
    def test_catalan(self, runner):
        result = runner.invoke(cli, ["generate", "sequence", "--type", "catalan", "--count", "5", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == [1, 1, 2, 5, 14]

    def test_triangular(self, runner):
        result = runner.invoke(cli, ["generate", "sequence", "--type", "triangular", "--count", "4", "--format", "json"])
        assert result.exit_code == 0

    def test_arithmetic(self, runner):
        result = runner.invoke(cli, ["generate", "sequence", "--type", "arithmetic", "--count", "5",
                                      "--start", "1", "--step", "3", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == [1.0, 4.0, 7.0, 10.0, 13.0]


class TestGenerateStats:
    def test_normal(self, runner):
        result = runner.invoke(cli, ["generate", "stats", "--distribution", "normal",
                                      "--count", "100", "--seed", "42", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 100

    def test_uniform(self, runner):
        result = runner.invoke(cli, ["generate", "stats", "--distribution", "uniform",
                                      "--count", "50", "--format", "json"])
        assert result.exit_code == 0


class TestAnalyze:
    def test_statistical(self, runner):
        result = runner.invoke(cli, ["analyze", "statistical",
                                      "--data", "1,2,3,4,5,6,7,8,9,10"])
        assert result.exit_code == 0
        assert "Mean" in result.output or "Count" in result.output

    def test_patterns(self, runner):
        result = runner.invoke(cli, ["analyze", "patterns",
                                      "--data", "1,2,3,4,5,6,7,8,9,10"])
        assert result.exit_code == 0

    def test_fit(self, runner):
        result = runner.invoke(cli, ["analyze", "fit",
                                      "--data", "1,2,3,4,5,6,7,8,9,10"])
        assert result.exit_code == 0

    def test_randomness(self, runner):
        data = ",".join(str(i * 37 % 251) for i in range(100))
        result = runner.invoke(cli, ["analyze", "randomness", "--data", data])
        assert result.exit_code == 0


class TestTransform:
    def test_normalize(self, runner):
        result = runner.invoke(cli, ["transform", "apply",
                                      "--data", "1,2,3,4,5",
                                      "--operation", "normalize",
                                      "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0] == 0.0
        assert data[-1] == 1.0

    def test_sort(self, runner):
        result = runner.invoke(cli, ["transform", "apply",
                                      "--data", "5,3,1,4,2",
                                      "--operation", "sort",
                                      "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == sorted(data)

    def test_format_binary(self, runner):
        result = runner.invoke(cli, ["transform", "format-numbers",
                                      "--data", "1,2,4,8",
                                      "--format", "binary"])
        assert result.exit_code == 0
        assert "1" in result.output


class TestCLIVersion:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
