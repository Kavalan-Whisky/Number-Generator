# Number Generator

A comprehensive Python-based toolkit for generating, analyzing, and transforming number sequences. Provides a CLI, REST API, and web dashboard.

## Features

- **8 Generator Modules**: Random PRNGs, prime numbers, Fibonacci, mathematical sequences, statistical distributions, cryptographic generators, fractals, combinatorics
- **4 Analyzer Modules**: Statistical analysis, randomness testing (NIST-inspired), pattern detection, distribution fitting
- **3 Transformer Modules**: Number formatting, sequence transformations, encoders
- **Click CLI**: Rich colored terminal output with progress bars
- **Flask REST API**: Full REST API with JSON responses
- **Web Dashboard**: Interactive browser-based dashboard
- **SQLAlchemy Database**: Generation history and results persistence
- **100+ Tests**: Comprehensive test suite

## Installation

```bash
# Clone / enter the project
cd /home/user/Number-Generator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### CLI

```bash
# Generate random numbers
numgen generate random --algorithm mersenne --count 20 --min 0 --max 1000

# Generate prime numbers
numgen generate prime --count 50

# Fibonacci sequence
numgen generate fibonacci --count 20 --variant matrix

# Mathematical sequences
numgen generate sequence --type catalan --count 10

# Statistical distributions
numgen generate stats --distribution normal --mean 0 --std 1 --count 100

# Analyze sequence
numgen analyze statistical --data "1,2,3,4,5,6,7,8,9,10"

# Test randomness
numgen analyze randomness --data "42,17,88,33,71,..."

# Transform
numgen transform apply --data "5,3,1,4,2" --operation normalize
```

### Web API

```bash
# Start the server
python -m src.web.app

# Or with Flask
FLASK_APP=src/web/app.py flask run
```

Then visit:
- Dashboard: http://localhost:5000
- API: http://localhost:5000/api/v1/generate/random?count=10

### Python API

```python
from src.core.generators.random_generator import RandomGeneratorFactory
from src.core.generators.prime_generator import PrimeGeneratorFactory
from src.core.generators.fibonacci_generator import FibonacciGenerator
from src.core.analyzers.statistical_analyzer import StatisticalAnalyzer

# Random numbers with PCG
gen = RandomGeneratorFactory.create("pcg", seed=42)
numbers = gen.generate(100, 0, 1000)

# Prime numbers
primes = PrimeGeneratorFactory.generate_primes(50, start=2)

# Fibonacci
fib = FibonacciGenerator()
sequence = fib.generate(30, "matrix")

# Statistical analysis
analyzer = StatisticalAnalyzer(numbers)
stats = analyzer.describe()
print(f"Mean: {stats['mean']:.2f}, Std: {stats['std_dev']:.2f}")
```

## Architecture

```
Number-Generator/
├── src/
│   ├── core/
│   │   ├── generators/    # Number generation algorithms
│   │   ├── analyzers/     # Statistical analysis tools
│   │   ├── transformers/  # Sequence transformation tools
│   │   └── utils/         # Shared utilities
│   ├── cli/               # Click-based CLI
│   ├── web/               # Flask web app + REST API
│   └── database/          # SQLAlchemy models
├── tests/                 # 100+ unit tests
├── scripts/               # Benchmark scripts
├── config/                # Configuration files
└── docs/                  # API documentation
```

## Generators

### Random PRNGs
| Algorithm | Period | Speed | Quality |
|-----------|--------|-------|---------|
| LCG | 2^32 | Very Fast | Basic |
| Xorshift-32 | 2^32-1 | Very Fast | Good |
| Xorshift-64 | 2^64-1 | Very Fast | Good |
| PCG | 2^64 | Fast | Excellent |
| LFSR | 2^n-1 | Fast | Good |
| Mersenne Twister | 2^19937-1 | Fast | Excellent |
| Middle Square | Short | Moderate | Poor |
| Blum-Blum-Shub | - | Slow | Cryptographic |

### Sequences
- Arithmetic, Geometric, Harmonic
- Triangular, Square, Pentagonal, Hexagonal
- Catalan, Bell, Euler, Bernoulli numbers
- Collatz, Recamán, Look-and-Say sequences
- Padovan, Perrin sequences

### Statistical Distributions
- Normal (Box-Muller transform)
- Poisson (Knuth algorithm)
- Exponential, Gamma, Beta
- Uniform, Binomial, Chi-squared, Student-t

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_generators/test_random.py -v
```

## Benchmarks

```bash
python scripts/benchmark.py
```

## API Documentation

See [docs/API.md](docs/API.md) for full REST API reference.

## License

MIT License
