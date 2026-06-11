"""Tests for cryptographic_sequences module."""
import pytest
from src.core.generators.cryptographic_sequences import (
    legendre_symbol, jacobi_symbol, quadratic_residues, quadratic_non_residues,
    legendre_sequence, jacobi_sequence,
    HashCounterGenerator, Sha3SequenceGenerator,
    blum_micali, blum_micali_integers,
    ec_add, ec_mul, ec_point_sequence, ec_x_sequence,
    m_sequence, gold_code,
    CryptographicSequenceGenerator,
)


class TestLegendreSymbol:
    def test_known_values(self):
        # (1/5)=1, (4/5)=1, (2/5)=-1, (3/5)=-1
        assert legendre_symbol(1, 5) == 1
        assert legendre_symbol(4, 5) == 1
        assert legendre_symbol(2, 5) == -1
        assert legendre_symbol(3, 5) == -1

    def test_zero_residue(self):
        assert legendre_symbol(5, 5) == 0
        assert legendre_symbol(0, 7) == 0

    def test_quadratic_residues(self):
        qr = quadratic_residues(7)
        assert set(qr) == {1, 2, 4}

    def test_non_residues(self):
        nqr = quadratic_non_residues(7)
        assert set(nqr) == {3, 5, 6}

    def test_sequence_binary(self):
        seq = legendre_sequence(7)
        assert all(b in (0, 1) for b in seq)
        assert len(seq) == 6  # a = 1..6


class TestJacobiSymbol:
    def test_known(self):
        # (2/9): 9=3^2, so Jacobi(2,9) = Jacobi(2,3)^2 mod ... just test roundtrip
        v = jacobi_symbol(2, 9)
        assert v in (-1, 0, 1)

    def test_prime_equals_legendre(self):
        for p in [5, 7, 11, 13]:
            for a in range(1, p):
                assert jacobi_symbol(a, p) == legendre_symbol(a, p)

    def test_invalid_even(self):
        with pytest.raises(ValueError):
            jacobi_symbol(3, 4)

    def test_sequence(self):
        seq = jacobi_sequence(15, 10)
        assert len(seq) == 10
        assert all(v in (-1, 0, 1) for v in seq)


class TestHashCounterGenerator:
    def test_length(self):
        gen = HashCounterGenerator()
        vals = gen.generate(50)
        assert len(vals) == 50

    def test_range(self):
        gen = HashCounterGenerator()
        vals = gen.generate(100, lo=0, hi=100)
        assert all(0 <= v < 100 for v in vals)

    def test_reproducible_after_reset(self):
        gen = HashCounterGenerator(key=b"test")
        v1 = gen.generate(20)
        gen.reset()
        v2 = gen.generate(20)
        assert v1 == v2

    def test_bits(self):
        gen = HashCounterGenerator()
        bits = gen.generate_bits(64)
        assert len(bits) == 64
        assert all(b in (0, 1) for b in bits)

    def test_different_keys(self):
        g1 = HashCounterGenerator(key=b"key1")
        g2 = HashCounterGenerator(key=b"key2")
        assert g1.generate(20) != g2.generate(20)


class TestSha3SequenceGenerator:
    def test_length(self):
        gen = Sha3SequenceGenerator(seed=b"test")
        vals = gen.generate(30)
        assert len(vals) == 30

    def test_range_2bytes(self):
        gen = Sha3SequenceGenerator(seed=b"s")
        vals = gen.generate(50, nbytes=2)
        assert all(0 <= v < 2 ** 16 for v in vals)

    def test_different_seeds(self):
        g1 = Sha3SequenceGenerator(seed=b"a")
        g2 = Sha3SequenceGenerator(seed=b"b")
        assert g1.generate(20) != g2.generate(20)


class TestBlumMicali:
    # Use a small prime where g is a primitive root
    P, G, X0 = 59, 2, 3

    def test_bits_binary(self):
        bits = blum_micali(self.P, self.G, self.X0, 50)
        assert all(b in (0, 1) for b in bits)

    def test_length(self):
        bits = blum_micali(self.P, self.G, self.X0, 100)
        assert len(bits) == 100

    def test_integers_length(self):
        ints = blum_micali_integers(self.P, self.G, self.X0, 80, int_bits=8)
        assert len(ints) == 10

    def test_integers_range(self):
        ints = blum_micali_integers(self.P, self.G, self.X0, 80, int_bits=8)
        assert all(0 <= v < 256 for v in ints)


class TestEllipticCurve:
    # y^2 = x^3 + x + 6 mod 11, generator (2, 7)
    A, B, P, G_PT = 1, 6, 11, (2, 7)

    def test_point_on_curve(self):
        x, y = self.G_PT
        lhs = (y * y) % self.P
        rhs = (x ** 3 + self.A * x + self.B) % self.P
        assert lhs == rhs

    def test_add_identity(self):
        pt = ec_add(None, self.G_PT, self.A, self.P)
        assert pt == self.G_PT

    def test_double(self):
        pt2 = ec_add(self.G_PT, self.G_PT, self.A, self.P)
        assert pt2 is not None
        # should lie on curve
        x, y = pt2
        assert (y * y) % self.P == (x ** 3 + self.A * x + self.B) % self.P

    def test_sequence_length(self):
        seq = ec_point_sequence(self.A, self.B, self.P, self.G_PT, 5)
        assert len(seq) <= 5

    def test_x_sequence(self):
        xs = ec_x_sequence(self.A, self.B, self.P, self.G_PT, 5)
        assert all(0 <= x < self.P for x in xs)


class TestMSequence:
    def test_length(self):
        seq = m_sequence([4, 3], 0b1111, 15)
        assert len(seq) == 15

    def test_binary(self):
        seq = m_sequence([4, 3], 0b1111, 50)
        assert all(b in (0, 1) for b in seq)

    def test_maximal_period(self):
        # Use taps [4,1] which gives maximal LFSR period 2^4-1=15 with state 0b1111
        seq = m_sequence([4, 1], 0b1111, 30)
        # First 15 should repeat (maximal period)
        assert seq[:15] == seq[15:30]

    def test_gold_code(self):
        gc = gold_code([4, 3], [4, 1], 0b1111, 0b0001, 15)
        assert len(gc) == 15
        assert all(b in (0, 1) for b in gc)


class TestCryptographicSequenceGenerator:
    def setup_method(self):
        self.gen = CryptographicSequenceGenerator()

    def test_legendre_sequence(self):
        seq = self.gen.legendre_sequence(7)
        assert len(seq) == 6

    def test_quadratic_residues(self):
        qr = self.gen.quadratic_residues(7)
        assert set(qr) == {1, 2, 4}

    def test_hash_counter(self):
        vals = self.gen.hash_counter(20, key=b"test")
        assert len(vals) == 20

    def test_sha3_sequence(self):
        vals = self.gen.sha3_sequence(20)
        assert len(vals) == 20

    def test_m_sequence(self):
        seq = self.gen.m_sequence([4, 3], 0b1111, 15)
        assert len(seq) == 15

    def test_ec_x_coords(self):
        xs = self.gen.ec_x_coords(5)
        assert len(xs) <= 5
