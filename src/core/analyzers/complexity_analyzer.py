"""
Complexity and information-theoretic analysis of number sequences.

Provides Kolmogorov complexity estimates (via compression), Lempel-Ziv (LZ76)
complexity, approximate entropy (ApEn), sample entropy (SampEn), permutation
entropy, recurrence-plot statistics, and multiscale entropy.
"""

import math
import zlib
import bz2
import lzma
import struct
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ComplexityReport:
    sequence_length: int
    lz76_complexity: int
    lz76_normalized: float
    approximate_entropy: Optional[float]
    sample_entropy: Optional[float]
    permutation_entropy: float
    compression_ratios: Dict[str, float] = field(default_factory=dict)
    recurrence_rate: Optional[float] = None
    determinism: Optional[float] = None


# ---------------------------------------------------------------------------
# Lempel-Ziv (LZ76) complexity
# ---------------------------------------------------------------------------

def lz76_complexity(seq: List[int]) -> int:
    """Compute LZ76 complexity (number of distinct substrings in LZ parsing)."""
    if not seq:
        return 0
    vocab = set()
    i, step, phrases = 0, 1, 0
    while i + step <= len(seq):
        sub = tuple(seq[i:i + step])
        if sub in vocab:
            step += 1
        else:
            vocab.add(sub)
            phrases += 1
            i += step
            step = 1
    return phrases


def lz76_normalized(seq: List[int]) -> float:
    """Normalised LZ76 complexity (0=simple, 1=maximally complex)."""
    n = len(seq)
    if n < 2:
        return 0.0
    c = lz76_complexity(seq)
    # Normalisation: c / (n / log2(n))
    return c / (n / math.log2(n))


# ---------------------------------------------------------------------------
# Approximate entropy (ApEn)
# ---------------------------------------------------------------------------

def _phi(seq: List[float], m: int, r: float) -> float:
    n = len(seq)
    count = 0
    for i in range(n - m):
        template = seq[i:i + m]
        for j in range(n - m):
            if max(abs(template[k] - seq[j + k]) for k in range(m)) <= r:
                count += 1
    return math.log(count / (n - m)) if count > 0 else float("-inf")


def approximate_entropy(seq: List[float], m: int = 2, r: Optional[float] = None) -> float:
    """ApEn(m, r) — smaller → more regular, larger → more irregular."""
    if r is None:
        std = _std(seq)
        r = 0.2 * std if std > 0 else 0.01
    return _phi(seq, m, r) - _phi(seq, m + 1, r)


# ---------------------------------------------------------------------------
# Sample entropy (SampEn)
# ---------------------------------------------------------------------------

def sample_entropy(seq: List[float], m: int = 2, r: Optional[float] = None) -> float:
    """SampEn — more robust than ApEn for short sequences."""
    if r is None:
        std = _std(seq)
        r = 0.2 * std if std > 0 else 0.01
    n = len(seq)

    def _count(template_len: int) -> int:
        total = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if max(abs(seq[i + k] - seq[j + k]) for k in range(template_len)) <= r:
                    total += 1
        return total

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return float("inf")
    return -math.log(A / B) if A > 0 else float("inf")


# ---------------------------------------------------------------------------
# Permutation entropy
# ---------------------------------------------------------------------------

def permutation_entropy(seq: List[float], order: int = 3,
                         normalise: bool = True) -> float:
    """Permutation entropy of a sequence (Bandt-Pompe)."""
    n = len(seq)
    if n < order:
        return 0.0
    from itertools import permutations as _perms

    perm_counts: Dict[Tuple[int, ...], int] = {}
    for i in range(n - order + 1):
        window = seq[i:i + order]
        rank = tuple(sorted(range(order), key=lambda k: window[k]))
        perm_counts[rank] = perm_counts.get(rank, 0) + 1

    total = sum(perm_counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in perm_counts.values())
    if normalise:
        entropy /= math.log2(math.factorial(order))
    return entropy


# ---------------------------------------------------------------------------
# Compression-ratio complexity estimate
# ---------------------------------------------------------------------------

def compression_ratio(seq: List[int], codec: str = "zlib") -> float:
    """Estimate complexity via compression ratio (higher = more complex)."""
    raw = struct.pack(f">{len(seq)}i", *seq)
    if codec == "zlib":
        compressed = zlib.compress(raw, level=9)
    elif codec == "bz2":
        compressed = bz2.compress(raw)
    elif codec == "lzma":
        compressed = lzma.compress(raw)
    else:
        raise ValueError(f"Unknown codec: {codec}")
    return len(compressed) / len(raw)


def compression_ratios_all(seq: List[int]) -> Dict[str, float]:
    return {
        "zlib": compression_ratio(seq, "zlib"),
        "bz2": compression_ratio(seq, "bz2"),
        "lzma": compression_ratio(seq, "lzma"),
    }


# ---------------------------------------------------------------------------
# Recurrence quantification analysis (RQA)
# ---------------------------------------------------------------------------

def recurrence_matrix(seq: List[float], threshold: float,
                        norm: str = "max") -> List[List[int]]:
    """Build a binary recurrence matrix."""
    n = len(seq)
    R = []
    for i in range(n):
        row = []
        for j in range(n):
            diff = abs(seq[i] - seq[j])
            row.append(1 if diff <= threshold else 0)
        R.append(row)
    return R


def recurrence_rate(R: List[List[int]]) -> float:
    """RR = fraction of recurrence points (excluding diagonal)."""
    n = len(R)
    if n < 2:
        return 0.0
    count = sum(R[i][j] for i in range(n) for j in range(n) if i != j)
    return count / (n * (n - 1))


def determinism_rqa(R: List[List[int]], min_line: int = 2) -> float:
    """DET = fraction of recurrence points in diagonal lines of length >= min_line."""
    n = len(R)
    diag_points = 0
    total_rec = 0
    for k in range(-(n - 1), n):
        diag = [R[i][i - k] for i in range(n) if 0 <= i - k < n]
        run = 0
        for bit in diag:
            if bit:
                run += 1
                total_rec += 1
            else:
                if run >= min_line:
                    diag_points += run
                run = 0
        if run >= min_line:
            diag_points += run
    return diag_points / total_rec if total_rec > 0 else 0.0


# ---------------------------------------------------------------------------
# Multiscale entropy
# ---------------------------------------------------------------------------

def coarse_grain(seq: List[float], scale: int) -> List[float]:
    n = len(seq) // scale
    return [sum(seq[i * scale:(i + 1) * scale]) / scale for i in range(n)]


def multiscale_entropy(seq: List[float], max_scale: int = 5,
                        m: int = 2) -> List[float]:
    """Compute sample entropy at multiple coarse-graining scales."""
    results = []
    for s in range(1, max_scale + 1):
        cg = coarse_grain(seq, s)
        if len(cg) < m + 2:
            results.append(float("nan"))
        else:
            results.append(sample_entropy(cg, m))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std(seq: List[float]) -> float:
    n = len(seq)
    if n < 2:
        return 0.0
    mean = sum(seq) / n
    return math.sqrt(sum((x - mean) ** 2 for x in seq) / (n - 1))


# ---------------------------------------------------------------------------
# Unified complexity analyzer
# ---------------------------------------------------------------------------

class ComplexityAnalyzer:
    """All-in-one sequence complexity analysis."""

    def analyze(self, seq: List[float], rqa_threshold: Optional[float] = None,
                fast: bool = False) -> ComplexityReport:
        int_seq = [int(round(x)) for x in seq]
        lz = lz76_complexity(int_seq)
        lzn = lz76_normalized(int_seq)
        pe = permutation_entropy(seq)

        if fast or len(seq) > 500:
            apen = None
            sampen = None
        else:
            apen = approximate_entropy(seq)
            sampen = sample_entropy(seq)

        cr = compression_ratios_all(int_seq)

        rr = None
        det = None
        if rqa_threshold is not None and len(seq) <= 200:
            R = recurrence_matrix(seq, rqa_threshold)
            rr = recurrence_rate(R)
            det = determinism_rqa(R)

        return ComplexityReport(
            sequence_length=len(seq),
            lz76_complexity=lz,
            lz76_normalized=lzn,
            approximate_entropy=apen,
            sample_entropy=sampen,
            permutation_entropy=pe,
            compression_ratios=cr,
            recurrence_rate=rr,
            determinism=det,
        )

    def lz76(self, seq: List[int]) -> int:
        return lz76_complexity(seq)

    def permutation_entropy(self, seq: List[float], order: int = 3) -> float:
        return permutation_entropy(seq, order)

    def multiscale(self, seq: List[float], max_scale: int = 5) -> List[float]:
        return multiscale_entropy(seq, max_scale)
