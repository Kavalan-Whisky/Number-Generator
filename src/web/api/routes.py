"""
REST API routes for the Number Generator web application.
"""

import json
import time
from typing import Any, Dict

from flask import Blueprint, jsonify, request

api_bp = Blueprint("api", __name__)


def _success_response(data: Any, count: int = None) -> Dict:
    resp = {"status": "success", "data": data, "timestamp": time.time()}
    if count is not None:
        resp["count"] = count
    return resp


def _error_response(message: str, code: int = 400) -> tuple:
    return jsonify({"status": "error", "message": message}), code


# ---- Random Number Generation ----

@api_bp.route("/generate/random", methods=["GET", "POST"])
def generate_random():
    """
    Generate random numbers using various PRNG algorithms.

    Query/body params:
      - algorithm: lcg|xorshift32|xorshift64|pcg|lfsr|mersenne (default: mersenne)
      - count: number of values (default: 10)
      - min: minimum value (default: 0)
      - max: maximum value (default: 1000)
      - seed: optional integer seed
    """
    from src.core.generators.random_generator import RandomGeneratorFactory

    params = request.get_json(silent=True) or request.args
    algorithm = params.get("algorithm", "mersenne")
    count = int(params.get("count", 10))
    min_val = int(params.get("min", 0))
    max_val = int(params.get("max", 1000))
    seed = params.get("seed")
    seed = int(seed) if seed is not None else None

    if count > 10000:
        return _error_response("count exceeds maximum of 10000")
    if min_val >= max_val:
        return _error_response("min must be less than max")

    try:
        gen = RandomGeneratorFactory.create(algorithm, seed=seed)
        numbers = gen.generate(count, min_val, max_val)
        return jsonify(_success_response(
            {"numbers": numbers, "algorithm": algorithm, "range": [min_val, max_val]},
            count=count
        ))
    except Exception as e:
        return _error_response(str(e))


# ---- Prime Numbers ----

@api_bp.route("/generate/prime", methods=["GET", "POST"])
def generate_prime():
    """
    Generate prime numbers.

    Params:
      - count: number of primes (default: 20)
      - start: starting value (default: 2)
    """
    from src.core.generators.prime_generator import PrimeGeneratorFactory

    params = request.get_json(silent=True) or request.args
    count = int(params.get("count", 20))
    start = int(params.get("start", 2))

    if count > 10000:
        return _error_response("count exceeds maximum of 10000")

    try:
        primes = PrimeGeneratorFactory.generate_primes(count, start)
        return jsonify(_success_response(
            {"primes": primes, "start": start},
            count=len(primes)
        ))
    except Exception as e:
        return _error_response(str(e))


@api_bp.route("/generate/prime/check", methods=["GET", "POST"])
def check_prime():
    """Check if a number is prime."""
    from src.core.generators.prime_generator import miller_rabin_is_prime

    params = request.get_json(silent=True) or request.args
    n = params.get("n")
    if n is None:
        return _error_response("Parameter 'n' is required")

    try:
        n = int(n)
        is_p = miller_rabin_is_prime(n)
        return jsonify(_success_response({"n": n, "is_prime": is_p}))
    except ValueError:
        return _error_response("n must be an integer")


# ---- Fibonacci ----

@api_bp.route("/generate/fibonacci", methods=["GET", "POST"])
def generate_fibonacci():
    """
    Generate Fibonacci sequence.

    Params:
      - count: number of values (default: 20)
      - variant: iterative|matrix|recursive (default: iterative)
      - type: fibonacci|lucas|tribonacci (default: fibonacci)
    """
    from src.core.generators.fibonacci_generator import (
        FibonacciGenerator, LucasSequence, GeneralizedFibonacci
    )

    params = request.get_json(silent=True) or request.args
    count = int(params.get("count", 20))
    variant = params.get("variant", "iterative")
    fib_type = params.get("type", "fibonacci")

    if count > 1000:
        return _error_response("count exceeds maximum of 1000")

    try:
        if fib_type == "fibonacci":
            gen = FibonacciGenerator()
            numbers = gen.generate(count, variant)
        elif fib_type == "lucas":
            numbers = LucasSequence.generate(count)
        elif fib_type == "tribonacci":
            gen = GeneralizedFibonacci.tribonacci()
            numbers = gen.generate(count)
        else:
            return _error_response(f"Unknown type: {fib_type}")

        return jsonify(_success_response(
            {"sequence": numbers, "type": fib_type, "variant": variant},
            count=count
        ))
    except Exception as e:
        return _error_response(str(e))


# ---- Mathematical Sequences ----

@api_bp.route("/generate/sequence", methods=["GET", "POST"])
def generate_sequence():
    """
    Generate mathematical sequences (Catalan, Bell, etc.)

    Params:
      - type: arithmetic|geometric|catalan|bell|triangular|recaman|collatz|padovan|perrin
      - count: number of values (default: 15)
      - start/difference/ratio: for arithmetic/geometric
    """
    from src.core.generators.sequence_generator import SequenceGeneratorFactory

    params = request.get_json(silent=True) or request.args
    seq_type = params.get("type", "arithmetic")
    count = int(params.get("count", 15))

    if count > 1000:
        return _error_response("count exceeds maximum of 1000")

    kwargs = {}
    if "start" in params:
        kwargs["start"] = float(params["start"])
    if "difference" in params:
        kwargs["difference"] = float(params["difference"])
    if "ratio" in params:
        kwargs["ratio"] = float(params["ratio"])
    if "n" in params:
        kwargs["n"] = int(params["n"])

    try:
        sequence = SequenceGeneratorFactory.generate(seq_type, count, **kwargs)
        return jsonify(_success_response(
            {"sequence": sequence, "type": seq_type},
            count=len(sequence)
        ))
    except Exception as e:
        return _error_response(str(e))


# ---- Statistical Distributions ----

@api_bp.route("/generate/statistical", methods=["GET", "POST"])
def generate_statistical():
    """Generate numbers from statistical distributions."""
    from src.core.generators.statistical_generator import StatisticalGeneratorFactory

    params = request.get_json(silent=True) or request.args
    distribution = params.get("distribution", "normal")
    count = int(params.get("count", 100))
    seed = params.get("seed")
    seed = int(seed) if seed is not None else None

    if count > 100000:
        return _error_response("count exceeds maximum of 100000")

    # Build distribution params
    dist_params = {}
    if distribution in ("normal", "gaussian"):
        dist_params["mu"] = float(params.get("mean", 0))
        dist_params["sigma"] = float(params.get("std", 1))
    elif distribution == "poisson":
        dist_params["lam"] = float(params.get("lambda", 1))
    elif distribution == "exponential":
        dist_params["rate"] = float(params.get("rate", 1))
    elif distribution == "uniform":
        dist_params["low"] = float(params.get("low", 0))
        dist_params["high"] = float(params.get("high", 1))

    try:
        gen = StatisticalGeneratorFactory.create(distribution, seed=seed, **dist_params)
        numbers = gen.generate(count)
        return jsonify(_success_response(
            {"numbers": [round(x, 6) for x in numbers], "distribution": distribution},
            count=count
        ))
    except Exception as e:
        return _error_response(str(e))


# ---- Analysis ----

@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Perform statistical analysis on a sequence.

    Body:
      - data: list of numbers
      - tests: statistical|randomness|patterns|distributions
    """
    body = request.get_json(silent=True)
    if not body:
        return _error_response("Request body required")

    data = body.get("data", [])
    tests = body.get("tests", "statistical")

    if not data:
        return _error_response("'data' field is required")
    if len(data) > 100000:
        return _error_response("data exceeds maximum of 100000 elements")

    try:
        numbers = [float(x) for x in data]
    except (TypeError, ValueError):
        return _error_response("All values must be numeric")

    results = {}

    if tests in ("statistical", "all"):
        from src.core.analyzers.statistical_analyzer import describe
        results["statistics"] = describe(numbers)

    if tests in ("randomness", "all") and len(numbers) >= 10:
        from src.core.analyzers.randomness_tester import RandomnessTester
        tester = RandomnessTester()
        report = tester.run_all(numbers)
        results["randomness"] = report.to_dict()

    if tests in ("patterns", "all"):
        from src.core.analyzers.pattern_detector import PatternDetector
        detector = PatternDetector()
        report = detector.analyze(numbers)
        results["patterns"] = report.to_dict()

    if tests in ("distributions", "all") and len(numbers) >= 5:
        from src.core.analyzers.distribution_fitter import DistributionFitter
        fitter = DistributionFitter()
        fit_results = fitter.fit_all(numbers)
        results["distribution_fits"] = [
            {
                "distribution": r.distribution,
                "parameters": r.parameters,
                "ks_statistic": r.ks_statistic,
                "ks_p_value": r.ks_p_value,
                "aic": r.aic,
                "is_good_fit": r.is_good_fit,
            }
            for r in fit_results
        ]

    return jsonify(_success_response(results))


# ---- Transform ----

@api_bp.route("/transform", methods=["POST"])
def transform():
    """
    Apply transformations to a sequence.

    Body:
      - data: list of numbers
      - operation: normalize|standardize|differences|moving_average|delta_encode|etc.
      - params: additional parameters
    """
    from src.core.transformers.sequence_transformer import (
        normalize, standardize, sort_sequence, reverse, unique,
        running_sum, running_product, differences, delta_encode,
        moving_average, exponential_moving_average
    )

    body = request.get_json(silent=True)
    if not body:
        return _error_response("Request body required")

    data = body.get("data", [])
    operation = body.get("operation", "normalize")
    params = body.get("params", {})

    if not data:
        return _error_response("'data' field is required")

    try:
        numbers = [float(x) for x in data]
    except (TypeError, ValueError):
        return _error_response("All values must be numeric")

    op_map = {
        "normalize": lambda d: normalize(d),
        "standardize": lambda d: standardize(d),
        "sort": lambda d: sort_sequence(d),
        "reverse": lambda d: reverse(d),
        "unique": lambda d: unique(d),
        "running_sum": running_sum,
        "running_product": running_product,
        "differences": lambda d: differences(d, int(params.get("order", 1))),
        "delta_encode": delta_encode,
        "moving_average": lambda d: moving_average(d, int(params.get("window", 3))),
        "ema": lambda d: exponential_moving_average(d, float(params.get("alpha", 0.3))),
    }

    if operation not in op_map:
        return _error_response(f"Unknown operation: {operation}. Available: {list(op_map.keys())}")

    try:
        result = op_map[operation](numbers)
        return jsonify(_success_response(
            {"result": result, "operation": operation},
            count=len(result)
        ))
    except Exception as e:
        return _error_response(str(e))


# ---- History (requires DB) ----

@api_bp.route("/history", methods=["GET"])
def history():
    """Get generation history (placeholder - requires database setup)."""
    limit = int(request.args.get("limit", 20))
    return jsonify(_success_response(
        {"sessions": [], "message": "Database not configured"},
        count=0
    ))


# ---- Utility endpoints ----

@api_bp.route("/algorithms", methods=["GET"])
def algorithms():
    """List available algorithms and distributions."""
    from src.core.generators.random_generator import RandomGeneratorFactory
    from src.core.generators.statistical_generator import StatisticalGeneratorFactory
    from src.core.generators.sequence_generator import SequenceGeneratorFactory

    return jsonify(_success_response({
        "random_algorithms": RandomGeneratorFactory.available_algorithms(),
        "distributions": StatisticalGeneratorFactory.available_distributions(),
        "sequence_types": SequenceGeneratorFactory.available_types(),
    }))


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "number-generator"})
