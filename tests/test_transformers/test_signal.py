"""Tests for signal_transformer module."""
import pytest
import math
from src.core.transformers.signal_transformer import (
    rectangular_window, hann_window, hamming_window, blackman_window, bartlett_window,
    apply_window, dft, idft, magnitude_spectrum, convolve, cross_correlate,
    fir_lowpass, fir_highpass, apply_fir,
    diff, integrate, fractional_diff,
    downsample, upsample, linear_interpolate,
    min_max_normalize, z_score_normalize, quantize,
    SignalTransformer,
)


class TestWindows:
    def test_rectangular_ones(self):
        w = rectangular_window(8)
        assert w == [1.0] * 8

    def test_hann_ends(self):
        w = hann_window(8)
        assert w[0] == pytest.approx(0.0, abs=1e-9)
        assert w[-1] == pytest.approx(0.0, abs=0.05)

    def test_hamming_range(self):
        w = hamming_window(16)
        assert all(0 <= v <= 1 for v in w)

    def test_blackman_length(self):
        assert len(blackman_window(32)) == 32

    def test_bartlett_center(self):
        w = bartlett_window(9)
        assert w[4] == pytest.approx(1.0)

    def test_apply_window(self):
        seq = [1.0] * 8
        windowed = apply_window(seq, hann_window)
        assert len(windowed) == 8


class TestDFT:
    def test_constant(self):
        seq = [1.0] * 4
        spec = dft(seq)
        assert abs(spec[0]) == pytest.approx(4.0)
        assert abs(spec[1]) < 1e-9

    def test_roundtrip(self):
        seq = [1.0, 2.0, 3.0, 4.0]
        recovered = idft(dft(seq))
        for a, b in zip(seq, recovered):
            assert a == pytest.approx(b, abs=1e-9)

    def test_magnitude_length(self):
        seq = [math.sin(i) for i in range(8)]
        ms = magnitude_spectrum(seq)
        assert len(ms) == 8


class TestConvolution:
    def test_impulse_response(self):
        signal = [1.0, 0.0, 0.0, 0.0]
        kernel = [1.0, 2.0, 3.0]
        result = convolve(signal, kernel)
        assert result[:3] == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]

    def test_length(self):
        s = [1.0] * 5
        k = [1.0] * 3
        assert len(convolve(s, k)) == 7  # 5+3-1

    def test_cross_correlate_length(self):
        x = [1.0] * 4
        y = [1.0] * 3
        cc = cross_correlate(x, y)
        assert len(cc) == 6  # 4+3-1


class TestFIRFilter:
    def test_lowpass_length(self):
        kernel = fir_lowpass(0.1, 31)
        assert len(kernel) == 31

    def test_lowpass_sum_approx_1(self):
        kernel = fir_lowpass(0.5, 31)
        assert sum(kernel) == pytest.approx(1.0, abs=0.01)

    def test_highpass_blocks_dc(self):
        kernel = fir_highpass(0.1, 31)
        dc_signal = [1.0] * 100
        filtered = apply_fir(dc_signal, kernel)
        assert abs(sum(filtered) / len(filtered)) < 0.1

    def test_apply_fir_length(self):
        kernel = fir_lowpass(0.2, 11)
        signal = [float(i) for i in range(50)]
        out = apply_fir(signal, kernel)
        assert len(out) == 50


class TestDifferencing:
    def test_diff_first_order(self):
        seq = [1.0, 3.0, 6.0, 10.0]
        d = diff(seq, 1)
        assert d == [pytest.approx(2.0), pytest.approx(3.0), pytest.approx(4.0)]

    def test_diff_second_order(self):
        seq = [1.0, 4.0, 9.0, 16.0]
        d = diff(seq, 2)
        assert d == [pytest.approx(2.0), pytest.approx(2.0)]

    def test_integrate(self):
        seq = [1.0] * 5
        integrated = integrate(seq, initial=0.0)
        assert integrated == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_fractional_diff_length(self):
        seq = [float(i) for i in range(20)]
        fd = fractional_diff(seq, d=0.5)
        assert len(fd) == 20


class TestResampling:
    def test_downsample(self):
        seq = list(range(10))
        ds = downsample(seq, 2)
        assert ds == [0, 2, 4, 6, 8]

    def test_upsample(self):
        seq = [1, 2, 3]
        us = upsample(seq, 2)
        assert us == [1, 0, 2, 0, 3, 0]

    def test_linear_interpolate_length(self):
        seq = [0.0, 1.0, 2.0]
        result = linear_interpolate(seq, 2)
        assert len(result) == 5  # 3 + 2 intermediate

    def test_linear_interpolate_midpoint(self):
        seq = [0.0, 2.0]
        result = linear_interpolate(seq, 2)
        assert result[1] == pytest.approx(1.0)


class TestNormalization:
    def test_minmax_range(self):
        seq = [1.0, 3.0, 5.0, 7.0, 9.0]
        normalized = min_max_normalize(seq)
        assert min(normalized) == pytest.approx(0.0)
        assert max(normalized) == pytest.approx(1.0)

    def test_zscore_mean(self):
        seq = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = z_score_normalize(seq)
        mean = sum(z) / len(z)
        assert mean == pytest.approx(0.0, abs=1e-9)

    def test_zscore_std(self):
        seq = [float(i) for i in range(20)]
        z = z_score_normalize(seq)
        std = math.sqrt(sum(v ** 2 for v in z) / (len(z) - 1))
        assert std == pytest.approx(1.0, rel=0.01)

    def test_quantize_levels(self):
        seq = [0.0, 0.25, 0.5, 0.75, 1.0]
        q = quantize(seq, 4)
        assert all(0 <= v < 4 for v in q)

    def test_constant_minmax(self):
        seq = [5.0] * 10
        n = min_max_normalize(seq)
        assert n == [0.0] * 10


class TestSignalTransformer:
    def setup_method(self):
        self.t = SignalTransformer()
        self.seq = [math.sin(2 * math.pi * i / 16) for i in range(64)]

    def test_lowpass(self):
        out = self.t.lowpass_filter(self.seq)
        assert len(out) == len(self.seq)

    def test_highpass(self):
        out = self.t.highpass_filter(self.seq)
        assert len(out) == len(self.seq)

    def test_bandpass(self):
        out = self.t.bandpass_filter(self.seq)
        assert len(out) == len(self.seq)

    def test_spectrum(self):
        ps = self.t.spectrum(self.seq)
        assert len(ps) == len(self.seq)

    def test_window_hann(self):
        out = self.t.apply_window(self.seq, "hann")
        assert len(out) == len(self.seq)

    def test_window_invalid(self):
        with pytest.raises(ValueError):
            self.t.apply_window(self.seq, "nonexistent")

    def test_diff(self):
        d = self.t.diff(self.seq, 1)
        assert len(d) == len(self.seq) - 1

    def test_normalize_minmax(self):
        n = self.t.normalize(self.seq, "minmax")
        assert min(n) == pytest.approx(0.0, abs=0.01)
        assert max(n) == pytest.approx(1.0, abs=0.01)

    def test_normalize_zscore(self):
        n = self.t.normalize(self.seq, "zscore")
        mean = sum(n) / len(n)
        assert mean == pytest.approx(0.0, abs=0.01)
