"""
Signal processing transformations for number sequences.

Includes: convolution, cross-correlation, DFT/IDFT, windowing functions
(Hann, Hamming, Blackman, Kaiser), filtering (FIR low/high/band-pass),
hilbert transform, and phase vocoder utilities.
"""

import math
import cmath
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def rectangular_window(n: int) -> List[float]:
    return [1.0] * n


def hann_window(n: int) -> List[float]:
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]


def hamming_window(n: int) -> List[float]:
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def blackman_window(n: int) -> List[float]:
    return [
        0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) +
        0.08 * math.cos(4 * math.pi * i / (n - 1))
        for i in range(n)
    ]


def bartlett_window(n: int) -> List[float]:
    half = (n - 1) / 2
    return [1 - abs(i - half) / half for i in range(n)]


def apply_window(seq: List[float], window_fn) -> List[float]:
    w = window_fn(len(seq))
    return [seq[i] * w[i] for i in range(len(seq))]


# ---------------------------------------------------------------------------
# DFT / IDFT (pure Python)
# ---------------------------------------------------------------------------

def dft(seq: List[float]) -> List[complex]:
    n = len(seq)
    return [
        sum(seq[k] * cmath.exp(-2j * math.pi * k * m / n) for k in range(n))
        for m in range(n)
    ]


def idft(spec: List[complex]) -> List[float]:
    n = len(spec)
    return [
        (sum(spec[k] * cmath.exp(2j * math.pi * k * m / n) for k in range(n)) / n).real
        for m in range(n)
    ]


def magnitude_spectrum(seq: List[float]) -> List[float]:
    return [abs(c) for c in dft(seq)]


def phase_spectrum(seq: List[float]) -> List[float]:
    return [cmath.phase(c) for c in dft(seq)]


# ---------------------------------------------------------------------------
# Convolution and cross-correlation
# ---------------------------------------------------------------------------

def convolve(signal: List[float], kernel: List[float]) -> List[float]:
    """Full linear convolution."""
    n, m = len(signal), len(kernel)
    out_len = n + m - 1
    out = [0.0] * out_len
    for i, s in enumerate(signal):
        for j, k in enumerate(kernel):
            out[i + j] += s * k
    return out


def cross_correlate(x: List[float], y: List[float]) -> List[float]:
    """Cross-correlation of x and y."""
    n, m = len(x), len(y)
    out_len = n + m - 1
    out = [0.0] * out_len
    for i in range(n):
        for j in range(m):
            out[i + j] += x[i] * y[j]
    return out


def circular_convolve(x: List[float], h: List[float]) -> List[float]:
    """Circular convolution (length must match or h is zero-padded)."""
    n = len(x)
    h_pad = h + [0.0] * (n - len(h)) if len(h) < n else h[:n]
    out = [0.0] * n
    for i in range(n):
        for j in range(n):
            out[i] += x[j] * h_pad[(i - j) % n]
    return out


# ---------------------------------------------------------------------------
# FIR filter design (window method)
# ---------------------------------------------------------------------------

def _sinc(x: float) -> float:
    if x == 0:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def fir_lowpass(cutoff: float, num_taps: int) -> List[float]:
    """Design a low-pass FIR filter using a Hamming window."""
    mid = (num_taps - 1) / 2
    ideal = [2 * cutoff * _sinc(2 * cutoff * (i - mid)) for i in range(num_taps)]
    window = hamming_window(num_taps)
    kernel = [ideal[i] * window[i] for i in range(num_taps)]
    # Normalise
    s = sum(kernel)
    return [k / s for k in kernel]


def fir_highpass(cutoff: float, num_taps: int) -> List[float]:
    lp = fir_lowpass(cutoff, num_taps)
    mid = (num_taps - 1) // 2
    hp = [-k for k in lp]
    hp[mid] += 1.0
    return hp


def fir_bandpass(low: float, high: float, num_taps: int) -> List[float]:
    lp_low = fir_lowpass(low, num_taps)
    lp_high = fir_lowpass(high, num_taps)
    bp = [lp_high[i] - lp_low[i] for i in range(num_taps)]
    s = sum(bp)
    return [k / s if s else k for k in bp]


def apply_fir(signal: List[float], kernel: List[float]) -> List[float]:
    """Apply an FIR filter (linear convolution, cropped to signal length)."""
    full = convolve(signal, kernel)
    half = len(kernel) // 2
    return full[half:half + len(signal)]


# ---------------------------------------------------------------------------
# Hilbert transform (via DFT)
# ---------------------------------------------------------------------------

def hilbert_transform(seq: List[float]) -> List[float]:
    """Compute the discrete Hilbert transform (imaginary part)."""
    n = len(seq)
    spec = dft(seq)
    h_spec = [0.0 + 0.0j] * n
    for k in range(n):
        if k == 0 or (n % 2 == 0 and k == n // 2):
            h_spec[k] = spec[k]
        elif k < n // 2:
            h_spec[k] = 2 * spec[k]
        else:
            h_spec[k] = 0.0 + 0.0j
    analytic = idft(h_spec)
    return analytic


def instantaneous_amplitude(seq: List[float]) -> List[float]:
    ht = hilbert_transform(seq)
    return [math.sqrt(seq[i] ** 2 + ht[i] ** 2) for i in range(len(seq))]


def instantaneous_phase(seq: List[float]) -> List[float]:
    ht = hilbert_transform(seq)
    return [math.atan2(ht[i], seq[i]) for i in range(len(seq))]


# ---------------------------------------------------------------------------
# Differencing and integration
# ---------------------------------------------------------------------------

def diff(seq: List[float], order: int = 1) -> List[float]:
    result = list(seq)
    for _ in range(order):
        result = [result[i] - result[i - 1] for i in range(1, len(result))]
    return result


def integrate(seq: List[float], initial: float = 0.0) -> List[float]:
    out = [initial]
    for x in seq:
        out.append(out[-1] + x)
    return out


def fractional_diff(seq: List[float], d: float, max_lag: int = 10) -> List[float]:
    """Fractional differencing with parameter d (0 < d < 1)."""
    weights = [1.0]
    for k in range(1, max_lag + 1):
        weights.append(-weights[-1] * (d - k + 1) / k)
    out = []
    for i in range(len(seq)):
        val = sum(
            weights[k] * seq[i - k]
            for k in range(min(i + 1, len(weights)))
        )
        out.append(val)
    return out


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def downsample(seq: List[float], factor: int) -> List[float]:
    return seq[::factor]


def upsample(seq: List[float], factor: int) -> List[float]:
    out = []
    for x in seq:
        out.append(x)
        out.extend([0.0] * (factor - 1))
    return out


def linear_interpolate(seq: List[float], factor: int) -> List[float]:
    """Upsample by factor using linear interpolation."""
    if len(seq) < 2:
        return list(seq) * factor
    out = []
    for i in range(len(seq) - 1):
        out.append(seq[i])
        for k in range(1, factor):
            t = k / factor
            out.append(seq[i] * (1 - t) + seq[i + 1] * t)
    out.append(seq[-1])
    return out


# ---------------------------------------------------------------------------
# Normalisation and scaling
# ---------------------------------------------------------------------------

def min_max_normalize(seq: List[float], lo: float = 0.0,
                       hi: float = 1.0) -> List[float]:
    mn, mx = min(seq), max(seq)
    if mx == mn:
        return [lo] * len(seq)
    return [lo + (x - mn) / (mx - mn) * (hi - lo) for x in seq]


def z_score_normalize(seq: List[float]) -> List[float]:
    n = len(seq)
    m = sum(seq) / n
    s = math.sqrt(sum((x - m) ** 2 for x in seq) / (n - 1)) if n > 1 else 1.0
    return [(x - m) / s if s else 0.0 for x in seq]


def quantize(seq: List[float], levels: int) -> List[int]:
    """Uniform quantization to `levels` bins."""
    mn, mx = min(seq), max(seq)
    if mx == mn:
        return [0] * len(seq)
    step = (mx - mn) / levels
    return [min(int((x - mn) / step), levels - 1) for x in seq]


# ---------------------------------------------------------------------------
# Unified transformer class
# ---------------------------------------------------------------------------

class SignalTransformer:
    """Apply signal processing transforms to numeric sequences."""

    def lowpass_filter(self, seq: List[float], cutoff: float = 0.1,
                        taps: int = 31) -> List[float]:
        kernel = fir_lowpass(cutoff, taps)
        return apply_fir(seq, kernel)

    def highpass_filter(self, seq: List[float], cutoff: float = 0.1,
                         taps: int = 31) -> List[float]:
        kernel = fir_highpass(cutoff, taps)
        return apply_fir(seq, kernel)

    def bandpass_filter(self, seq: List[float], low: float = 0.1,
                         high: float = 0.4, taps: int = 31) -> List[float]:
        kernel = fir_bandpass(low, high, taps)
        return apply_fir(seq, kernel)

    def spectrum(self, seq: List[float]) -> List[float]:
        return magnitude_spectrum(seq)

    def apply_window(self, seq: List[float], name: str = "hann") -> List[float]:
        fns = {"hann": hann_window, "hamming": hamming_window,
               "blackman": blackman_window, "bartlett": bartlett_window,
               "rectangular": rectangular_window}
        if name not in fns:
            raise ValueError(f"Unknown window: {name}")
        return apply_window(seq, fns[name])

    def diff(self, seq: List[float], order: int = 1) -> List[float]:
        return diff(seq, order)

    def normalize(self, seq: List[float], method: str = "minmax") -> List[float]:
        if method == "minmax":
            return min_max_normalize(seq)
        elif method == "zscore":
            return z_score_normalize(seq)
        raise ValueError(f"Unknown method: {method}")
