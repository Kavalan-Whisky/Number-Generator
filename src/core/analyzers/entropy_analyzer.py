"""
Entropy and complexity analysis for number sequences.
Shannon entropy, min-entropy, collision entropy, approximate entropy,
sample entropy, Lempel-Ziv complexity, and a compression-ratio estimate.
"""

import math
import zlib
from collections import Counter
from typing import Dict, List, Sequence


def _probabilities(data: Sequence) -> List[float]:
    if not data:
        return []
    counts = Counter(data)
    n = len(data)
    return [c / n for c in counts.values()]


def shannon_entropy(data: Sequence, base: float = 2.0) -> float:
    """Shannon entropy H = -sum p log_b p (in bits by default)."""
    probs = _probabilities(data)
    if not probs:
        return 0.0
    return -sum(p * math.log(p, base) for p in probs if p > 0)


def min_entropy(data: Sequence, base: float = 2.0) -> float:
    """Min-entropy H_inf = -log_b max(p)."""
    probs = _probabilities(data)
    if not probs:
        return 0.0
    return -math.log(max(probs), base)


def collision_entropy(data: Sequence, base: float = 2.0) -> float:
    """Renyi collision entropy H_2 = -log_b sum p_i^2."""
    probs = _probabilities(data)
    if not probs:
        return 0.0
    return -math.log(sum(p * p for p in probs), base)


def approximate_entropy(data: Sequence[float], m: int = 2, r: float = None) -> float:
    """Approximate entropy ApEn(m, r) of a numeric sequence."""
    data = [float(x) for x in data]
    n = len(data)
    if n < m + 1:
        return 0.0
    if r is None:
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / n)
        r = 0.2 * std if std > 0 else 0.2

    def _phi(m_len: int) -> float:
        templates = [data[i:i + m_len] for i in range(n - m_len + 1)]
        count = len(templates)
        total = 0.0
        for t1 in templates:
            matches = sum(
                1 for t2 in templates
                if max(abs(a - b) for a, b in zip(t1, t2)) <= r
            )
            total += math.log(matches / count)
        return total / count

    return _phi(m) - _phi(m + 1)


def sample_entropy(data: Sequence[float], m: int = 2, r: float = None) -> float:
    """Sample entropy SampEn(m, r); excludes self-matches."""
    data = [float(x) for x in data]
    n = len(data)
    if n < m + 2:
        return 0.0
    if r is None:
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / n)
        r = 0.2 * std if std > 0 else 0.2

    def _count(m_len: int) -> int:
        templates = [data[i:i + m_len] for i in range(n - m_len + 1)]
        total = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if max(abs(a - b) for a, b in zip(templates[i], templates[j])) <= r:
                    total += 1
        return total

    b = _count(m)
    a = _count(m + 1)
    if a == 0 or b == 0:
        return float("inf")
    return -math.log(a / b)


def lempel_ziv_complexity(data: Sequence) -> int:
    """Lempel-Ziv complexity: number of distinct phrases in an LZ78-style parse.

    The sequence is scanned left to right; each time the current phrase has not
    been seen before, it is added to the dictionary and a new phrase begins.
    """
    symbols = tuple(str(x) for x in data)
    n = len(symbols)
    if n == 0:
        return 0
    phrases = set()
    count = 0
    start = 0
    end = 1
    while end <= n:
        phrase = symbols[start:end]
        if phrase not in phrases:
            phrases.add(phrase)
            count += 1
            start = end
        end += 1
    if start < n:
        # Trailing phrase that was already in the dictionary
        count += 1
    return count


def normalized_lz_complexity(data: Sequence) -> float:
    """LZ complexity normalized by n / log2(n) (approaches 1 for random data)."""
    n = len(data)
    if n < 2:
        return 0.0
    alphabet = max(2, len(set(str(x) for x in data)))
    return lempel_ziv_complexity(data) * math.log(n, alphabet) / n


def compression_ratio(data: Sequence) -> float:
    """Estimate compressibility via zlib: compressed_size / original_size.

    Lower ratios indicate more structure (less entropy).
    """
    raw = ",".join(str(x) for x in data).encode("utf-8")
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, 9)
    return len(compressed) / len(raw)


class EntropyAnalyzer:
    """Aggregate entropy analysis over a sequence."""

    def __init__(self, m: int = 2, r: float = None):
        self.m = m
        self.r = r

    def analyze(self, data: Sequence) -> Dict[str, float]:
        """Run all entropy measures and return a report dictionary."""
        result = {
            "n": len(data),
            "shannon_entropy": shannon_entropy(data),
            "min_entropy": min_entropy(data),
            "collision_entropy": collision_entropy(data),
            "lempel_ziv_complexity": lempel_ziv_complexity(data),
            "normalized_lz_complexity": normalized_lz_complexity(data),
            "compression_ratio": compression_ratio(data),
        }
        # ApEn/SampEn need numeric data
        try:
            numeric = [float(x) for x in data]
            if len(numeric) >= self.m + 2:
                result["approximate_entropy"] = approximate_entropy(numeric, self.m, self.r)
                samp = sample_entropy(numeric, self.m, self.r)
                result["sample_entropy"] = samp if samp != float("inf") else None
        except (TypeError, ValueError):
            pass
        return result
