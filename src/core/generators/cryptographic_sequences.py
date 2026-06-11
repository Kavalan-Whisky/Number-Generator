"""
Advanced cryptographic and hash-derived number sequences.

Includes: hash-based sequence generation (SHA-256 counter mode), Blum-Micali
generator, Naor-Reingold PRF, Legendre symbol sequences, Jacobi symbol sequences,
quadratic residue sequences, and elliptic-curve point sequences over small fields.
"""

import hashlib
import hmac
import struct
import math
from typing import List, Iterator, Tuple, Optional


# ---------------------------------------------------------------------------
# Modular arithmetic helpers
# ---------------------------------------------------------------------------

def mod_pow(base: int, exp: int, mod: int) -> int:
    return pow(base, exp, mod)


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p): 0, 1, or -1. p must be an odd prime."""
    if a % p == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def jacobi_symbol(a: int, n: int) -> int:
    """Jacobi symbol (a/n): generalises Legendre to odd n."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def quadratic_residues(p: int) -> List[int]:
    """All quadratic residues mod p (p prime)."""
    return sorted({pow(x, 2, p) for x in range(1, p)})


def quadratic_non_residues(p: int) -> List[int]:
    qr = set(quadratic_residues(p))
    return [x for x in range(1, p) if x not in qr]


# ---------------------------------------------------------------------------
# Legendre / Jacobi symbol sequences
# ---------------------------------------------------------------------------

def legendre_sequence(p: int) -> List[int]:
    """Binary sequence derived from Legendre symbols for a=1..p-1."""
    return [(legendre_symbol(a, p) + 1) // 2 for a in range(1, p)]


def jacobi_sequence(n: int, count: int) -> List[int]:
    """Jacobi symbol sequence (a/n) for a=1,3,5,...  (n odd)."""
    out = []
    a = 1
    while len(out) < count:
        if math.gcd(a, n) == 1:
            out.append(jacobi_symbol(a, n))
        a += 2
    return out


# ---------------------------------------------------------------------------
# Hash-counter mode generator (deterministic, seeded)
# ---------------------------------------------------------------------------

class HashCounterGenerator:
    """Generate integers deterministically from HMAC-SHA256 in counter mode."""

    def __init__(self, key: bytes = b"numgen-default-key", bits: int = 64):
        self.key = key
        self.bits = bits
        self._counter = 0

    def _next_bytes(self) -> bytes:
        msg = struct.pack(">Q", self._counter)
        self._counter += 1
        return hmac.new(self.key, msg, hashlib.sha256).digest()

    def next_int(self, lo: int = 0, hi: int = 2 ** 32) -> int:
        raw = int.from_bytes(self._next_bytes()[:8], "big")
        span = hi - lo
        return lo + (raw % span)

    def generate(self, count: int, lo: int = 0, hi: int = 2 ** 32) -> List[int]:
        return [self.next_int(lo, hi) for _ in range(count)]

    def generate_bits(self, count: int) -> List[int]:
        out = []
        while len(out) < count:
            raw = int.from_bytes(self._next_bytes(), "big")
            for shift in range(255, -1, -1):
                out.append((raw >> shift) & 1)
                if len(out) == count:
                    break
        return out

    def reset(self) -> None:
        self._counter = 0


# ---------------------------------------------------------------------------
# SHA-3 sponge-based generator
# ---------------------------------------------------------------------------

class Sha3SequenceGenerator:
    """Absorb a seed, squeeze out a stream of integers using SHA3-256."""

    def __init__(self, seed: bytes = b"sha3-seed"):
        self._state = hashlib.sha3_256(seed).digest()
        self._buf: List[int] = []

    def _squeeze(self) -> None:
        self._state = hashlib.sha3_256(self._state).digest()
        self._buf = list(self._state)

    def next_byte(self) -> int:
        if not self._buf:
            self._squeeze()
        return self._buf.pop(0)

    def next_int(self, nbytes: int = 4) -> int:
        b = bytes(self.next_byte() for _ in range(nbytes))
        return int.from_bytes(b, "big")

    def generate(self, count: int, nbytes: int = 4) -> List[int]:
        return [self.next_int(nbytes) for _ in range(count)]


# ---------------------------------------------------------------------------
# Blum-Micali generator
# ---------------------------------------------------------------------------

def blum_micali(p: int, g: int, x0: int, count: int) -> List[int]:
    """Blum-Micali PRNG: bit_i = 1 if x_i < (p-1)/2.

    g must be a primitive root mod p, p must be prime.
    """
    half = (p - 1) // 2
    bits: List[int] = []
    x = x0
    for _ in range(count):
        x = pow(g, x, p)
        bits.append(1 if x < half else 0)
    return bits


def blum_micali_integers(p: int, g: int, x0: int, bit_count: int,
                          int_bits: int = 8) -> List[int]:
    """Generate integers from Blum-Micali bit stream."""
    bits = blum_micali(p, g, x0, bit_count)
    ints = []
    for i in range(0, len(bits) - int_bits + 1, int_bits):
        val = 0
        for b in bits[i:i + int_bits]:
            val = (val << 1) | b
        ints.append(val)
    return ints


# ---------------------------------------------------------------------------
# Elliptic curve point sequences over small prime fields
# ---------------------------------------------------------------------------

def ec_add(P: Optional[Tuple[int, int]], Q: Optional[Tuple[int, int]],
           a: int, p: int) -> Optional[Tuple[int, int]]:
    """Add two points on y^2 = x^3 + ax + b (mod p)."""
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2:
        if (y1 + y2) % p == 0:
            return None
        # Point doubling
        num = (3 * x1 * x1 + a) % p
        den = pow(2 * y1, p - 2, p)
        lam = num * den % p
    else:
        num = (y2 - y1) % p
        den = pow((x2 - x1) % p, p - 2, p)
        lam = num * den % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)


def ec_mul(k: int, P: Optional[Tuple[int, int]], a: int, p: int) -> Optional[Tuple[int, int]]:
    """Scalar multiplication on elliptic curve."""
    result = None
    addend = P
    while k:
        if k & 1:
            result = ec_add(result, addend, a, p)
        addend = ec_add(addend, addend, a, p)
        k >>= 1
    return result


def ec_point_sequence(a: int, b: int, p: int,
                       generator_point: Tuple[int, int],
                       count: int) -> List[Tuple[int, int]]:
    """Generate the sequence P, 2P, 3P, ... on curve y^2=x^3+ax+b mod p."""
    seq = []
    current = generator_point
    for _ in range(count):
        if current is None:
            break
        seq.append(current)
        current = ec_add(current, generator_point, a, p)
    return seq


def ec_x_sequence(a: int, b: int, p: int,
                   generator_point: Tuple[int, int],
                   count: int) -> List[int]:
    """x-coordinates of the elliptic-curve point sequence."""
    return [pt[0] for pt in ec_point_sequence(a, b, p, generator_point, count)]


# ---------------------------------------------------------------------------
# Pseudo-noise sequences (m-sequences via LFSR)
# ---------------------------------------------------------------------------

def m_sequence(taps: List[int], state: int, length: int) -> List[int]:
    """Generate an m-sequence (maximal-length LFSR sequence).

    taps: feedback tap positions (1-based).
    state: initial state (non-zero integer).
    """
    seq = []
    n = max(taps)
    mask = (1 << n) - 1
    for _ in range(length):
        bit = state & 1
        seq.append(bit)
        feedback = 0
        for t in taps:
            feedback ^= (state >> (t - 1)) & 1
        state = ((state >> 1) | (feedback << (n - 1))) & mask
    return seq


def gold_code(taps1: List[int], taps2: List[int],
              state1: int, state2: int, length: int) -> List[int]:
    """Gold code: XOR of two m-sequences."""
    s1 = m_sequence(taps1, state1, length)
    s2 = m_sequence(taps2, state2, length)
    return [a ^ b for a, b in zip(s1, s2)]


# ---------------------------------------------------------------------------
# Unified crypto-sequence generator class
# ---------------------------------------------------------------------------

class CryptographicSequenceGenerator:
    """High-level interface for all cryptographic integer sequences."""

    def legendre_sequence(self, p: int) -> List[int]:
        return legendre_sequence(p)

    def quadratic_residues(self, p: int) -> List[int]:
        return quadratic_residues(p)

    def hash_counter(self, count: int, key: bytes = b"seed",
                     lo: int = 0, hi: int = 1000) -> List[int]:
        gen = HashCounterGenerator(key=key)
        return gen.generate(count, lo, hi)

    def sha3_sequence(self, count: int, seed: bytes = b"seed",
                      nbytes: int = 4) -> List[int]:
        gen = Sha3SequenceGenerator(seed)
        return gen.generate(count, nbytes)

    def blum_micali(self, p: int, g: int, x0: int,
                    count: int, int_bits: int = 8) -> List[int]:
        return blum_micali_integers(p, g, x0, count * int_bits, int_bits)

    def m_sequence(self, taps: List[int], state: int, length: int) -> List[int]:
        return m_sequence(taps, state, length)

    def ec_x_coords(self, count: int = 10) -> List[int]:
        # Small toy curve: y^2 = x^3 + x + 6  mod 11
        return ec_x_sequence(1, 6, 11, (2, 7), count)
