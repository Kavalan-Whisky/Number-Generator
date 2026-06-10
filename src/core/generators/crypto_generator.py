"""
Cryptographically Secure Number and Token Generators.
Uses Python's secrets module for secure randomness.
"""

import hashlib
import hmac
import math
import os
import secrets
import struct
import time
import uuid
from typing import List, Optional, Tuple


class SecureRandomGenerator:
    """
    Cryptographically secure random number generator.
    Uses os.urandom / secrets module.
    """

    def __init__(self):
        pass

    def next_int(self, bits: int = 32) -> int:
        """Generate a cryptographically secure random integer."""
        return secrets.randbits(bits)

    def next_float(self) -> float:
        """Generate a secure float in [0, 1)."""
        return secrets.randbits(53) / (1 << 53)

    def next_range(self, low: int, high: int) -> int:
        """Generate a secure integer in [low, high]."""
        return secrets.randbelow(high - low + 1) + low

    def generate(self, count: int, low: int = 0, high: int = 2**32 - 1) -> List[int]:
        """Generate a list of secure random integers."""
        return [self.next_range(low, high) for _ in range(count)]

    def random_bytes(self, n: int) -> bytes:
        """Generate n cryptographically secure random bytes."""
        return secrets.token_bytes(n)

    def random_hex(self, n: int = 16) -> str:
        """Generate n-byte hex string."""
        return secrets.token_hex(n)

    def random_urlsafe(self, n: int = 16) -> str:
        """Generate URL-safe random string."""
        return secrets.token_urlsafe(n)


class UUIDGenerator:
    """
    UUID Generation: v1 (time-based), v4 (random), v5 (namespace+name SHA1).
    """

    @staticmethod
    def v1() -> str:
        """Generate UUID v1 (time-based)."""
        return str(uuid.uuid1())

    @staticmethod
    def v4() -> str:
        """Generate UUID v4 (random)."""
        return str(uuid.uuid4())

    @staticmethod
    def v5(namespace: str, name: str) -> str:
        """Generate UUID v5 (namespace + name, SHA1)."""
        ns_map = {
            "dns": uuid.NAMESPACE_DNS,
            "url": uuid.NAMESPACE_URL,
            "oid": uuid.NAMESPACE_OID,
            "x500": uuid.NAMESPACE_X500,
        }
        ns = ns_map.get(namespace.lower(), uuid.NAMESPACE_URL)
        return str(uuid.uuid5(ns, name))

    @staticmethod
    def generate(count: int, version: int = 4) -> List[str]:
        """Generate count UUIDs of given version."""
        if version == 1:
            return [UUIDGenerator.v1() for _ in range(count)]
        elif version == 4:
            return [UUIDGenerator.v4() for _ in range(count)]
        else:
            raise ValueError(f"Unsupported UUID version: {version}")

    @staticmethod
    def uuid_to_int(uuid_str: str) -> int:
        """Convert UUID string to integer."""
        return uuid.UUID(uuid_str).int

    @staticmethod
    def uuid_from_int(n: int) -> str:
        """Convert integer to UUID."""
        return str(uuid.UUID(int=n))


class TokenGenerator:
    """Generates secure tokens in various formats."""

    def __init__(self, length: int = 32):
        self.length = length

    def hex_token(self, length: Optional[int] = None) -> str:
        """Generate hex token."""
        n = length or self.length
        return secrets.token_hex(n // 2)

    def urlsafe_token(self, length: Optional[int] = None) -> str:
        """Generate URL-safe base64 token."""
        n = length or self.length
        return secrets.token_urlsafe(n)

    def bytes_token(self, length: Optional[int] = None) -> bytes:
        """Generate raw bytes token."""
        n = length or self.length
        return secrets.token_bytes(n)

    def numeric_token(self, digits: int = 6) -> str:
        """Generate numeric token (like OTP)."""
        return str(secrets.randbelow(10**digits)).zfill(digits)

    def alphanumeric_token(self, length: int = 16) -> str:
        """Generate alphanumeric token."""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def generate_tokens(self, count: int, token_type: str = "hex") -> List[str]:
        """Generate count tokens of given type."""
        methods = {
            "hex": self.hex_token,
            "urlsafe": self.urlsafe_token,
            "numeric": self.numeric_token,
            "alphanumeric": self.alphanumeric_token,
        }
        if token_type not in methods:
            raise ValueError(f"Unknown token type: {token_type}")
        return [methods[token_type]() for _ in range(count)]


class OTPGenerator:
    """One-Time Password and one-time pad number generators."""

    @staticmethod
    def totp(secret: bytes, time_step: int = 30, digits: int = 6) -> str:
        """
        Time-based OTP (TOTP) as per RFC 6238.
        Returns a digits-length numeric OTP.
        """
        t = int(time.time()) // time_step
        msg = struct.pack(">Q", t)
        h = hmac.new(secret, msg, hashlib.sha1).digest()
        offset = h[19] & 0xF
        code = (
            (h[offset] & 0x7F) << 24
            | (h[offset + 1] & 0xFF) << 16
            | (h[offset + 2] & 0xFF) << 8
            | (h[offset + 3] & 0xFF)
        )
        return str(code % (10**digits)).zfill(digits)

    @staticmethod
    def hotp(secret: bytes, counter: int, digits: int = 6) -> str:
        """
        HMAC-based OTP (HOTP) as per RFC 4226.
        """
        msg = struct.pack(">Q", counter)
        h = hmac.new(secret, msg, hashlib.sha1).digest()
        offset = h[19] & 0xF
        code = (
            (h[offset] & 0x7F) << 24
            | (h[offset + 1] & 0xFF) << 16
            | (h[offset + 2] & 0xFF) << 8
            | (h[offset + 3] & 0xFF)
        )
        return str(code % (10**digits)).zfill(digits)

    @staticmethod
    def otp_pad(length: int) -> List[int]:
        """Generate a one-time pad as a list of random integers [0, 255]."""
        return [secrets.randbelow(256) for _ in range(length)]

    @staticmethod
    def encrypt_otp(message: List[int], pad: List[int]) -> List[int]:
        """XOR encrypt message with pad."""
        if len(message) != len(pad):
            raise ValueError("Message and pad must have same length")
        return [m ^ p for m, p in zip(message, pad)]

    @staticmethod
    def decrypt_otp(ciphertext: List[int], pad: List[int]) -> List[int]:
        """XOR decrypt (same as encrypt)."""
        return OTPGenerator.encrypt_otp(ciphertext, pad)

    @staticmethod
    def generate_numeric_otp(length: int = 6) -> str:
        """Generate a secure numeric OTP."""
        return str(secrets.randbelow(10**length)).zfill(length)


class PasswordNumberGenerator:
    """
    Generates numbers suitable for password components.
    Includes PIN, secure numeric sequences, etc.
    """

    @staticmethod
    def pin(digits: int = 4) -> str:
        """Generate a secure PIN of given length."""
        return "".join(str(secrets.randbelow(10)) for _ in range(digits))

    @staticmethod
    def dice_roll(sides: int = 6, count: int = 1) -> List[int]:
        """Simulate dice rolls using secure randomness."""
        return [secrets.randbelow(sides) + 1 for _ in range(count)]

    @staticmethod
    def lottery_numbers(pool: int = 49, count: int = 6) -> List[int]:
        """Generate lottery numbers (unique, sorted)."""
        if count > pool:
            raise ValueError("count cannot exceed pool size")
        numbers = list(range(1, pool + 1))
        selected = []
        for _ in range(count):
            idx = secrets.randbelow(len(numbers))
            selected.append(numbers.pop(idx))
        return sorted(selected)

    @staticmethod
    def random_prime_near(n: int) -> int:
        """Find a random prime near n using secure randomness."""
        from src.core.generators.prime_generator import is_prime, next_prime
        offset = secrets.randbelow(100)
        candidate = n + offset
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime(candidate):
            candidate += 2
        return candidate

    @staticmethod
    def generate_rsa_primes(bits: int = 512) -> Tuple[int, int]:
        """
        Generate two random primes suitable for RSA key generation.
        Uses Miller-Rabin primality testing.
        """
        from src.core.generators.prime_generator import miller_rabin_is_prime

        def gen_prime(bits: int) -> int:
            while True:
                n = secrets.randbits(bits)
                n |= (1 << (bits - 1))  # Set high bit
                n |= 1  # Set low bit (make odd)
                if miller_rabin_is_prime(n):
                    return n

        p = gen_prime(bits)
        q = gen_prime(bits)
        while q == p:
            q = gen_prime(bits)
        return p, q


class HashBasedGenerator:
    """
    Hash-based deterministic random number generation.
    Suitable for reproducible sequences from a seed.
    """

    def __init__(self, seed: bytes, algorithm: str = "sha256"):
        self.seed = seed
        self.algorithm = algorithm
        self._counter = 0

    def _hash(self, data: bytes) -> bytes:
        """Compute hash of data."""
        h = hashlib.new(self.algorithm)
        h.update(data)
        return h.digest()

    def next_bytes(self, n: int) -> bytes:
        """Generate n pseudorandom bytes."""
        result = b""
        while len(result) < n:
            counter_bytes = self._counter.to_bytes(8, "big")
            block = self._hash(self.seed + counter_bytes)
            result += block
            self._counter += 1
        return result[:n]

    def next_int(self, bits: int = 32) -> int:
        """Generate a pseudorandom integer."""
        byte_count = (bits + 7) // 8
        data = self.next_bytes(byte_count)
        return int.from_bytes(data, "big") & ((1 << bits) - 1)

    def next_float(self) -> float:
        return self.next_int(53) / (1 << 53)

    def next_range(self, low: int, high: int) -> int:
        span = high - low + 1
        bits = span.bit_length() + 1
        while True:
            r = self.next_int(bits)
            if r < span:
                return low + r

    def generate(self, count: int, low: int = 0, high: int = 2**32 - 1) -> List[int]:
        return [self.next_range(low, high) for _ in range(count)]


class CryptoGeneratorFactory:
    """Factory for cryptographic generators."""

    @staticmethod
    def secure_random() -> SecureRandomGenerator:
        return SecureRandomGenerator()

    @staticmethod
    def uuid_generator() -> UUIDGenerator:
        return UUIDGenerator()

    @staticmethod
    def token_generator(length: int = 32) -> TokenGenerator:
        return TokenGenerator(length)

    @staticmethod
    def otp() -> OTPGenerator:
        return OTPGenerator()

    @staticmethod
    def hash_based(seed: Optional[bytes] = None) -> HashBasedGenerator:
        s = seed or secrets.token_bytes(32)
        return HashBasedGenerator(s)

    @staticmethod
    def generate_secure_numbers(count: int, low: int = 0, high: int = 1000) -> List[int]:
        """Convenience method for secure number generation."""
        gen = SecureRandomGenerator()
        return gen.generate(count, low, high)
