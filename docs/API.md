# Number Generator API Documentation

## Base URL

```
http://localhost:5000/api/v1
```

## Endpoints

### Health Check

```
GET /health
```

Returns service status.

**Response:**
```json
{"status": "ok", "service": "number-generator"}
```

---

### Generate Random Numbers

```
GET /generate/random
POST /generate/random
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| algorithm | string | mersenne | PRNG algorithm (lcg, xorshift32, xorshift64, pcg, lfsr, mersenne) |
| count | int | 10 | Number of values (max: 10000) |
| min | int | 0 | Minimum value |
| max | int | 1000 | Maximum value |
| seed | int | null | Random seed for reproducibility |

**Example:**
```bash
curl "http://localhost:5000/api/v1/generate/random?algorithm=lcg&count=10&min=1&max=100"
```

**Response:**
```json
{
  "status": "success",
  "count": 10,
  "data": {
    "numbers": [42, 17, 93, 55, ...],
    "algorithm": "lcg",
    "range": [1, 100]
  }
}
```

---

### Generate Prime Numbers

```
GET /generate/prime
POST /generate/prime
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| count | int | 20 | Number of primes |
| start | int | 2 | Starting value |

**Example:**
```bash
curl "http://localhost:5000/api/v1/generate/prime?count=10"
```

---

### Generate Fibonacci Sequence

```
GET /generate/fibonacci
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| count | int | 20 | Number of values |
| variant | string | iterative | Computation method (iterative, matrix, recursive) |
| type | string | fibonacci | Sequence type (fibonacci, lucas, tribonacci, tetranacci) |

---

### Generate Mathematical Sequences

```
GET /generate/sequence
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| type | string | Sequence type (arithmetic, geometric, catalan, bell, triangular, recaman, ...) |
| count | int | Number of values |
| start | float | Starting value (arithmetic/geometric) |
| difference | float | Common difference (arithmetic) |
| ratio | float | Common ratio (geometric) |

---

### Generate Statistical Distribution Samples

```
GET /generate/statistical
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| distribution | string | Distribution name (normal, poisson, exponential, gamma, beta, uniform) |
| count | int | Number of samples |
| mean / std | float | Normal distribution parameters |
| rate / lambda | float | Poisson/Exponential parameters |

---

### Analyze a Sequence

```
POST /analyze
```

**Request Body:**
```json
{
  "data": [1.0, 2.0, 3.0, ...],
  "tests": "all"
}
```

**Test options:** `statistical`, `randomness`, `patterns`, `distributions`, `all`

**Response includes:**
- Descriptive statistics (mean, median, std, skewness, kurtosis, entropy, ...)
- Randomness test results (frequency, runs, poker, autocorrelation, ...)
- Pattern detection results (arithmetic, geometric, Fibonacci-like, periodic, ...)
- Distribution fit results (KS test, AIC, parameters)

---

### Transform a Sequence

```
POST /transform
```

**Request Body:**
```json
{
  "data": [1.0, 2.0, 3.0, ...],
  "operation": "normalize",
  "params": {}
}
```

**Available operations:**
- `normalize` - Min-max normalization to [0, 1]
- `standardize` - Z-score standardization
- `sort` - Sort ascending
- `reverse` - Reverse the sequence
- `unique` - Remove duplicates
- `running_sum` - Cumulative sum
- `running_product` - Cumulative product
- `differences` - Finite differences (params: `order`)
- `delta_encode` - Delta encoding
- `moving_average` - Simple moving average (params: `window`)
- `ema` - Exponential moving average (params: `alpha`)

---

### List Available Algorithms

```
GET /algorithms
```

Returns all available generator algorithms, distributions, and sequence types.

---

## CLI Usage

```bash
# Random numbers
numgen generate random --algorithm lcg --count 100 --min 0 --max 1000 --seed 42

# Prime numbers
numgen generate prime --count 50 --start 2

# Fibonacci
numgen generate fibonacci --count 30 --variant matrix --type fibonacci

# Sequences
numgen generate sequence --type catalan --count 10
numgen generate sequence --type arithmetic --count 10 --start 0 --step 5

# Statistical distributions
numgen generate stats --distribution normal --mean 0 --std 1 --count 1000

# Analyze
numgen analyze statistical --data "1,2,3,4,5,6,7,8,9,10"
numgen analyze randomness --input data.csv
numgen analyze patterns --data "1,1,2,3,5,8,13,21"

# Transform
numgen transform apply --data "5,3,1,4,2" --operation normalize
numgen transform format-numbers --data "1,2,3,4" --format binary

# Export
numgen export save --data "1,2,3" --format json --output output.json
```

## Error Responses

```json
{
  "status": "error",
  "message": "Description of the error"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found
- `500` - Internal server error
