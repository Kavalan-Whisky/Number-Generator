"""Tests for time_series_analyzer module."""
import pytest
import math
from src.core.analyzers.time_series_analyzer import (
    linear_trend, detrend, trend_values,
    simple_moving_average, exponential_moving_average,
    dft, power_spectrum, dominant_frequency,
    autocorrelation_full, detect_seasonal_period,
    zero_crossing_rate, cusum_change_points,
    yule_walker, ar_forecast, TimeSeriesAnalyzer,
)


class TestLinearTrend:
    def test_constant(self):
        seq = [5.0] * 20
        slope, intercept = linear_trend(seq)
        assert slope == pytest.approx(0.0, abs=1e-10)
        assert intercept == pytest.approx(5.0)

    def test_increasing(self):
        seq = [float(i) for i in range(20)]
        slope, _ = linear_trend(seq)
        assert slope == pytest.approx(1.0, rel=0.01)

    def test_decreasing(self):
        seq = [float(20 - i) for i in range(20)]
        slope, _ = linear_trend(seq)
        assert slope < 0

    def test_detrend_removes_slope(self):
        seq = [2.0 * i + 10 for i in range(30)]
        dt = detrend(seq)
        s, _ = linear_trend(dt)
        assert abs(s) < 1e-8

    def test_trend_values_length(self):
        seq = [float(i) for i in range(25)]
        tv = trend_values(seq)
        assert len(tv) == 25


class TestSmoothing:
    def test_sma_length(self):
        seq = [float(i) for i in range(20)]
        sma = simple_moving_average(seq, 5)
        assert len(sma) == 16  # 20-5+1

    def test_sma_constant(self):
        seq = [3.0] * 20
        sma = simple_moving_average(seq, 5)
        assert all(v == pytest.approx(3.0) for v in sma)

    def test_ema_length(self):
        seq = list(range(20, dtype=float) if False else [float(i) for i in range(20)])
        ema = exponential_moving_average(seq, alpha=0.3)
        assert len(ema) == 20

    def test_ema_starts_at_first(self):
        seq = [7.0] + [float(i) for i in range(19)]
        ema = exponential_moving_average(seq, alpha=0.3)
        assert ema[0] == pytest.approx(7.0)

    def test_ema_smooths(self):
        seq = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        ema = exponential_moving_average(seq, alpha=0.5)
        # Values should be between 0 and 10
        assert all(0 <= v <= 10 for v in ema)


class TestDFT:
    def test_dc_component(self):
        seq = [3.0] * 8
        spec = dft(seq)
        assert abs(spec[0]) == pytest.approx(24.0)
        assert all(abs(spec[k]) < 1e-8 for k in range(1, 8))

    def test_power_spectrum_length(self):
        seq = [math.sin(2 * math.pi * i / 8) for i in range(8)]
        ps = power_spectrum(seq)
        assert len(ps) == 8

    def test_dominant_frequency_pure_sine(self):
        n = 32
        # frequency index 2 out of 32
        seq = [math.sin(2 * math.pi * 2 * i / n) for i in range(n)]
        freq, period = dominant_frequency(seq)
        assert freq is not None
        assert period is not None
        assert period == pytest.approx(n / 2, abs=1.0)

    def test_dominant_frequency_none_short(self):
        freq, period = dominant_frequency([1.0, 2.0])
        assert freq is None


class TestAutocorrelation:
    def test_lag0_is_one(self):
        seq = [math.sin(0.5 * i) for i in range(30)]
        ac = autocorrelation_full(seq)
        assert ac[0] == pytest.approx(1.0, abs=1e-9)

    def test_seasonal_detection(self):
        # Periodic signal with period 10
        seq = [math.sin(2 * math.pi * i / 10) for i in range(100)]
        period = detect_seasonal_period(seq, max_period=20)
        assert period is not None
        assert abs(period - 10) <= 2


class TestZeroCrossing:
    def test_alternating(self):
        seq = [(-1) ** i for i in range(20)]
        zcr = zero_crossing_rate(seq)
        assert zcr > 0.8

    def test_constant_positive(self):
        seq = [5.0] * 20
        assert zero_crossing_rate(seq) == pytest.approx(0.0)

    def test_range(self):
        seq = [math.sin(i * 0.5) for i in range(50)]
        zcr = zero_crossing_rate(seq)
        assert 0 <= zcr <= 1


class TestChangePointDetection:
    def test_no_change(self):
        seq = [5.0] * 50
        cps = cusum_change_points(seq)
        assert cps == []

    def test_detects_jump(self):
        seq = [1.0] * 25 + [100.0] * 25
        cps = cusum_change_points(seq, threshold=2.0)
        assert len(cps) > 0
        # Change point should be near index 25
        assert any(20 <= cp <= 30 for cp in cps)


class TestYuleWalker:
    def test_ar1_coefficient(self):
        # AR(1) with phi=0.9: x[n] = 0.9*x[n-1] + noise
        # Build a near-perfect AR(1) series
        seq = [0.0] * 200
        for i in range(1, 200):
            seq[i] = 0.9 * seq[i - 1]
        seq[50] = 5.0  # impulse
        coeffs = yule_walker(seq, 1)
        assert len(coeffs) == 1

    def test_ar_forecast_length(self):
        seq = [math.sin(i * 0.3) for i in range(50)]
        forecast = ar_forecast(seq, order=3, steps=10)
        assert len(forecast) == 10


class TestTimeSeriesAnalyzer:
    def setup_method(self):
        self.analyzer = TimeSeriesAnalyzer()
        self.seq = [math.sin(0.2 * i) for i in range(60)]

    def test_analyze_returns_report(self):
        report = self.analyzer.analyze(self.seq)
        assert report.length == 60
        assert isinstance(report.trend_slope, float)

    def test_smooth_ema(self):
        smoothed = self.analyzer.smooth(self.seq, method="ema")
        assert len(smoothed) == len(self.seq)

    def test_smooth_sma(self):
        smoothed = self.analyzer.smooth(self.seq, method="sma", window=5)
        assert len(smoothed) == len(self.seq) - 4

    def test_forecast(self):
        forecast = self.analyzer.forecast(self.seq, order=3, steps=5)
        assert len(forecast) == 5

    def test_spectrum(self):
        ps = self.analyzer.spectrum(self.seq)
        assert len(ps) == len(self.seq)

    def test_detrend(self):
        linear_seq = [float(i) for i in range(40)]
        dt = self.analyzer.detrend(linear_seq)
        slope, _ = linear_trend(dt)
        assert abs(slope) < 1e-8
