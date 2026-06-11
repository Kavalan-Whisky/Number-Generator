"""Tests for quasirandom_generator module."""
import pytest
from src.core.generators.quasirandom_generator import (
    HaltonSequence, SobolSequence, GoldenRatioSequence,
    van_der_corput, van_der_corput_sequence, latin_hypercube,
    QuasiRandomFactory,
)


class TestVanDerCorput:
    def test_base2_known(self):
        # VdC base 2: 1->0.5, 2->0.25, 3->0.75, 4->0.125
        assert van_der_corput(1, 2) == pytest.approx(0.5)
        assert van_der_corput(2, 2) == pytest.approx(0.25)
        assert van_der_corput(3, 2) == pytest.approx(0.75)
        assert van_der_corput(4, 2) == pytest.approx(0.125)

    def test_base3_known(self):
        assert van_der_corput(1, 3) == pytest.approx(1 / 3)
        assert van_der_corput(2, 3) == pytest.approx(2 / 3)
        assert van_der_corput(3, 3) == pytest.approx(1 / 9)

    def test_sequence_length(self):
        seq = van_der_corput_sequence(20, 2)
        assert len(seq) == 20

    def test_all_in_unit_interval(self):
        for base in [2, 3, 5]:
            seq = van_der_corput_sequence(50, base)
            assert all(0 < v < 1 for v in seq)

    def test_no_duplicates(self):
        seq = van_der_corput_sequence(16, 2)
        assert len(set(seq)) == 16


class TestHaltonSequence:
    def test_1d(self):
        h = HaltonSequence(dimensions=1)
        pts = h.generate_1d(8)
        assert len(pts) == 8
        assert all(0 < v < 1 for v in pts)

    def test_2d(self):
        h = HaltonSequence(dimensions=2)
        pts = h.generate(10)
        assert len(pts) == 10
        for pt in pts:
            assert len(pt) == 2
            assert all(0 < v < 1 for v in pt)

    def test_known_first_value_base2(self):
        h = HaltonSequence(dimensions=1)
        first = h.next()[0]
        assert first == pytest.approx(0.5)

    def test_next_increments(self):
        h = HaltonSequence(dimensions=1)
        a = h.next()
        b = h.next()
        assert a != b

    def test_generate_1d_length(self):
        h = HaltonSequence(dimensions=1)
        vals = h.generate_1d(20)
        assert len(vals) == 20


class TestSobolSequence:
    def test_1d_range(self):
        s = SobolSequence(dimensions=1)
        pts = s.generate_1d(16)
        assert all(0 <= v < 1 for v in pts)

    def test_2d_range(self):
        s = SobolSequence(dimensions=2)
        pts = s.generate(16)
        for pt in pts:
            assert len(pt) == 2
            assert all(0 <= v < 1 for v in pt)

    def test_low_discrepancy(self):
        # Sobol should cover [0,1) more evenly than random
        s = SobolSequence(dimensions=1)
        vals = s.generate_1d(8)
        # All 8 values should be distinct
        assert len(set(round(v, 6) for v in vals)) == 8

    def test_generate_1d_length(self):
        s = SobolSequence(dimensions=1)
        assert len(s.generate_1d(32)) == 32


class TestGoldenRatioSequence:
    def test_range(self):
        gr = GoldenRatioSequence()
        vals = gr.generate(50)
        assert all(0 <= v < 1 for v in vals)

    def test_no_exact_repeats(self):
        gr = GoldenRatioSequence()
        vals = gr.generate(100)
        assert len(set(round(v, 8) for v in vals)) == 100

    def test_length(self):
        gr = GoldenRatioSequence()
        assert len(gr.generate(30)) == 30


class TestLatinHypercube:
    def test_shape(self):
        lhs = latin_hypercube(5, 3)
        assert len(lhs) == 5
        assert all(len(row) == 3 for row in lhs)

    def test_range(self):
        lhs = latin_hypercube(8, 2)
        for row in lhs:
            assert all(0 <= v <= 1 for v in row)

    def test_stratified(self):
        n = 10
        lhs = latin_hypercube(n, 1)
        col = sorted(row[0] for row in lhs)
        # Each stratum [i/n, (i+1)/n] should contain exactly one sample
        for i in range(n):
            assert any(i / n <= v < (i + 1) / n for v in col), \
                f"Stratum {i} not covered"


class TestQuasiRandomFactory:
    def test_halton(self):
        pts = QuasiRandomFactory.generate("halton", 10)
        assert len(pts) == 10

    def test_sobol(self):
        pts = QuasiRandomFactory.generate("sobol", 8)
        assert len(pts) == 8

    def test_van_der_corput(self):
        vals = QuasiRandomFactory.generate("van_der_corput", 16, base=2)
        assert len(vals) == 16

    def test_golden(self):
        vals = QuasiRandomFactory.generate("golden", 20)
        assert len(vals) == 20

    def test_lhs(self):
        vals = QuasiRandomFactory.generate("lhs", 10, dimensions=2)
        assert len(vals) == 10
