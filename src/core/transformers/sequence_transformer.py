"""
Sequence Transformers - Operations on numeric sequences.
Running aggregates, moving averages, differences, normalization, etc.
"""

import math
from typing import Callable, Iterator, List, Optional, Tuple, Union


def running_sum(data: List[float]) -> List[float]:
    """Cumulative sum."""
    result = []
    total = 0.0
    for x in data:
        total += x
        result.append(total)
    return result


def running_product(data: List[float]) -> List[float]:
    """Cumulative product."""
    result = []
    prod = 1.0
    for x in data:
        prod *= x
        result.append(prod)
    return result


def running_max(data: List[float]) -> List[float]:
    """Running maximum."""
    result = []
    max_val = float("-inf")
    for x in data:
        max_val = max(max_val, x)
        result.append(max_val)
    return result


def running_min(data: List[float]) -> List[float]:
    """Running minimum."""
    result = []
    min_val = float("inf")
    for x in data:
        min_val = min(min_val, x)
        result.append(min_val)
    return result


def moving_average(data: List[float], window: int = 3) -> List[float]:
    """Simple moving average with given window size."""
    if window > len(data):
        raise ValueError("Window size exceeds data length")
    result = []
    for i in range(len(data) - window + 1):
        result.append(sum(data[i:i + window]) / window)
    return result


def exponential_moving_average(data: List[float], alpha: float = 0.3) -> List[float]:
    """
    Exponential moving average (EMA).
    alpha: smoothing factor (0 < alpha <= 1)
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    result = [data[0]]
    for x in data[1:]:
        result.append(alpha * x + (1 - alpha) * result[-1])
    return result


def weighted_moving_average(data: List[float], window: int = 3) -> List[float]:
    """Weighted moving average - more recent values weighted higher."""
    weights = list(range(1, window + 1))
    weight_sum = sum(weights)
    result = []
    for i in range(len(data) - window + 1):
        wma = sum(data[i + j] * weights[j] for j in range(window)) / weight_sum
        result.append(wma)
    return result


def differences(data: List[float], order: int = 1) -> List[float]:
    """Compute nth-order finite differences."""
    result = list(data)
    for _ in range(order):
        result = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        if not result:
            return []
    return result


def normalize(data: List[float], low: float = 0.0, high: float = 1.0) -> List[float]:
    """Min-max normalization to [low, high]."""
    if not data:
        return []
    min_v = min(data)
    max_v = max(data)
    if min_v == max_v:
        return [(low + high) / 2] * len(data)
    span = max_v - min_v
    return [low + (x - min_v) / span * (high - low) for x in data]


def standardize(data: List[float]) -> List[float]:
    """Z-score standardization: zero mean, unit variance."""
    n = len(data)
    if n == 0:
        return []
    mean_v = sum(data) / n
    variance = sum((x - mean_v) ** 2 for x in data) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(x - mean_v) / std for x in data]


def interleave(*sequences: List) -> List:
    """Interleave multiple sequences element by element."""
    if not sequences:
        return []
    min_len = min(len(s) for s in sequences)
    result = []
    for i in range(min_len):
        for seq in sequences:
            result.append(seq[i])
    return result


def chunk(data: List, size: int) -> List[List]:
    """Split sequence into chunks of given size."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    return [data[i:i + size] for i in range(0, len(data), size)]


def window_slide(data: List, window: int, step: int = 1) -> List[List]:
    """Generate sliding windows over the sequence."""
    if window > len(data):
        return []
    return [data[i:i + window] for i in range(0, len(data) - window + 1, step)]


def delta_encode(data: List[float]) -> List[float]:
    """Delta encoding: store differences from previous value."""
    if not data:
        return []
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(data[i] - data[i - 1])
    return result


def delta_decode(encoded: List[float]) -> List[float]:
    """Decode delta-encoded sequence."""
    if not encoded:
        return []
    result = [encoded[0]]
    for i in range(1, len(encoded)):
        result.append(result[-1] + encoded[i])
    return result


def flatten(nested: List) -> List:
    """Recursively flatten a nested list."""
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def zip_sequences(*sequences: List) -> List[Tuple]:
    """Zip multiple sequences together."""
    return list(zip(*sequences))


def apply_map(data: List[float], func: Callable[[float], float]) -> List[float]:
    """Apply a function to each element."""
    return [func(x) for x in data]


def apply_filter(data: List[float], func: Callable[[float], bool]) -> List[float]:
    """Filter elements by predicate."""
    return [x for x in data if func(x)]


def apply_reduce(data: List[float], func: Callable[[float, float], float], initial: Optional[float] = None) -> float:
    """Reduce sequence to single value."""
    if not data:
        raise ValueError("Empty sequence")
    result = initial if initial is not None else data[0]
    start = 0 if initial is not None else 1
    for x in data[start:]:
        result = func(result, x)
    return result


def sort_sequence(data: List[float], reverse: bool = False) -> List[float]:
    """Sort sequence."""
    return sorted(data, reverse=reverse)


def unique(data: List[float]) -> List[float]:
    """Return unique values preserving order."""
    seen = set()
    result = []
    for x in data:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def pad(data: List[float], target_length: int, value: float = 0.0, align: str = "right") -> List[float]:
    """Pad sequence to target length."""
    if len(data) >= target_length:
        return data[:target_length]
    padding = [value] * (target_length - len(data))
    if align == "right":
        return data + padding
    elif align == "left":
        return padding + data
    raise ValueError("align must be 'left' or 'right'")


def scale(data: List[float], factor: float) -> List[float]:
    """Scale all values by a factor."""
    return [x * factor for x in data]


def shift(data: List[float], offset: float) -> List[float]:
    """Add offset to all values."""
    return [x + offset for x in data]


def clip(data: List[float], low: float, high: float) -> List[float]:
    """Clip values to [low, high]."""
    return [max(low, min(high, x)) for x in data]


def rotate(data: List, n: int) -> List:
    """Rotate sequence by n positions."""
    if not data:
        return []
    n = n % len(data)
    return data[n:] + data[:n]


def reverse(data: List) -> List:
    """Reverse sequence."""
    return list(reversed(data))


def cumulative_max_drawdown(data: List[float]) -> float:
    """Compute maximum drawdown of a sequence (financial metric)."""
    if not data:
        return 0.0
    peak = data[0]
    max_dd = 0.0
    for x in data:
        if x > peak:
            peak = x
        drawdown = (peak - x) / peak if peak != 0 else 0
        max_dd = max(max_dd, drawdown)
    return max_dd


def autocorrelate(data: List[float], max_lag: int = 10) -> List[float]:
    """Compute autocorrelation for lags 1 through max_lag."""
    n = len(data)
    mean_v = sum(data) / n
    variance = sum((x - mean_v) ** 2 for x in data) / n
    if variance == 0:
        return [0.0] * max_lag

    result = []
    for lag in range(1, min(max_lag + 1, n)):
        cov = sum((data[i] - mean_v) * (data[i + lag] - mean_v) for i in range(n - lag)) / (n - lag)
        result.append(cov / variance)
    return result


class SequenceTransformer:
    """Class-based interface for sequence transformations."""

    def __init__(self, data: List[float]):
        self.data = list(data)

    def running_sum(self) -> List[float]:
        return running_sum(self.data)

    def running_product(self) -> List[float]:
        return running_product(self.data)

    def running_max(self) -> List[float]:
        return running_max(self.data)

    def running_min(self) -> List[float]:
        return running_min(self.data)

    def moving_average(self, window: int = 3) -> List[float]:
        return moving_average(self.data, window)

    def ema(self, alpha: float = 0.3) -> List[float]:
        return exponential_moving_average(self.data, alpha)

    def differences(self, order: int = 1) -> List[float]:
        return differences(self.data, order)

    def normalize(self) -> List[float]:
        return normalize(self.data)

    def standardize(self) -> List[float]:
        return standardize(self.data)

    def delta_encode(self) -> List[float]:
        return delta_encode(self.data)

    def sort(self, reverse: bool = False) -> List[float]:
        return sort_sequence(self.data, reverse)

    def unique(self) -> List[float]:
        return unique(self.data)

    def chunk(self, size: int) -> List[List[float]]:
        return chunk(self.data, size)

    def window(self, window: int, step: int = 1) -> List[List[float]]:
        return window_slide(self.data, window, step)

    def scale(self, factor: float) -> List[float]:
        return scale(self.data, factor)

    def shift(self, offset: float) -> List[float]:
        return shift(self.data, offset)

    def clip(self, low: float, high: float) -> List[float]:
        return clip(self.data, low, high)

    def reverse(self) -> List[float]:
        return reverse(self.data)

    def rotate(self, n: int) -> List[float]:
        return rotate(self.data, n)

    def autocorrelate(self, max_lag: int = 10) -> List[float]:
        return autocorrelate(self.data, max_lag)
