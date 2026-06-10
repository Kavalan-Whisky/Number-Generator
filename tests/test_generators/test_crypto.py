"""Tests for cryptographic number generators."""

import pytest
from src.core.generators.crypto_generator import (
    SecureRandomGenerator, UUIDGenerator, TokenGenerator,
    OTPGenerator, PasswordNumberGenerator, HashBasedGenerator,
    CryptoGeneratorFactory,
)


class TestSecureRandomGenerator:
    def test_next_int(self):
        gen = SecureRandomGenerator()
        n = gen.next_int(32)
        assert 0 <= n < 2**32

    def test_next_float(self):
        gen = SecureRandomGenerator()
        f = gen.next_float()
        assert 0.0 <= f < 1.0

    def test_next_range(self):
        gen = SecureRandomGenerator()
        for _ in range(100):
            n = gen.next_range(10, 20)
            assert 10 <= n <= 20

    def test_generate(self):
        gen = SecureRandomGenerator()
        nums = gen.generate(20, 0, 100)
        assert len(nums) == 20
        assert all(0 <= n <= 100 for n in nums)

    def test_random_bytes(self):
        gen = SecureRandomGenerator()
        b = gen.random_bytes(16)
        assert len(b) == 16

    def test_random_hex(self):
        gen = SecureRandomGenerator()
        h = gen.random_hex(16)
        assert len(h) == 32  # 16 bytes = 32 hex chars
        assert all(c in '0123456789abcdef' for c in h)

    def test_random_urlsafe(self):
        gen = SecureRandomGenerator()
        s = gen.random_urlsafe(16)
        assert len(s) > 0
        # URL-safe characters
        allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=')
        assert all(c in allowed for c in s)


class TestUUIDGenerator:
    def test_v4_format(self):
        uuid = UUIDGenerator.v4()
        parts = uuid.split('-')
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[2]) == 4

    def test_v4_unique(self):
        uuids = [UUIDGenerator.v4() for _ in range(20)]
        assert len(set(uuids)) == 20

    def test_v1_format(self):
        uuid = UUIDGenerator.v1()
        assert len(uuid) == 36
        assert uuid.count('-') == 4

    def test_v5(self):
        uuid = UUIDGenerator.v5("url", "https://example.com")
        assert len(uuid) == 36
        # Same input should give same UUID
        assert uuid == UUIDGenerator.v5("url", "https://example.com")

    def test_generate_multiple(self):
        uuids = UUIDGenerator.generate(5, version=4)
        assert len(uuids) == 5

    def test_uuid_to_int_and_back(self):
        uuid_str = UUIDGenerator.v4()
        n = UUIDGenerator.uuid_to_int(uuid_str)
        back = UUIDGenerator.uuid_from_int(n)
        assert back == uuid_str


class TestTokenGenerator:
    def test_hex_token(self):
        gen = TokenGenerator(32)
        token = gen.hex_token()
        assert len(token) == 32
        assert all(c in '0123456789abcdef' for c in token)

    def test_numeric_token_digits(self):
        gen = TokenGenerator()
        token = gen.numeric_token(6)
        assert len(token) == 6
        assert token.isdigit()

    def test_alphanumeric_token(self):
        gen = TokenGenerator()
        token = gen.alphanumeric_token(16)
        assert len(token) == 16
        assert token.isalnum()

    def test_generate_tokens(self):
        gen = TokenGenerator()
        tokens = gen.generate_tokens(5, "hex")
        assert len(tokens) == 5

    def test_unknown_type_raises(self):
        gen = TokenGenerator()
        with pytest.raises(ValueError):
            gen.generate_tokens(5, "unknown_type")

    def test_unique_tokens(self):
        gen = TokenGenerator(32)
        tokens = [gen.hex_token() for _ in range(20)]
        assert len(set(tokens)) == 20


class TestOTPGenerator:
    def test_generate_numeric_otp(self):
        otp = OTPGenerator.generate_numeric_otp(6)
        assert len(otp) == 6
        assert otp.isdigit()

    def test_otp_pad_length(self):
        pad = OTPGenerator.otp_pad(100)
        assert len(pad) == 100
        assert all(0 <= x <= 255 for x in pad)

    def test_encrypt_decrypt_otp(self):
        message = [72, 101, 108, 108, 111]  # "Hello" in ASCII
        pad = OTPGenerator.otp_pad(len(message))
        encrypted = OTPGenerator.encrypt_otp(message, pad)
        decrypted = OTPGenerator.decrypt_otp(encrypted, pad)
        assert decrypted == message

    def test_otp_length_mismatch(self):
        with pytest.raises(ValueError):
            OTPGenerator.encrypt_otp([1, 2, 3], [4, 5])

    def test_hotp_format(self):
        secret = b"test_secret_key"
        otp = OTPGenerator.hotp(secret, counter=0)
        assert len(otp) == 6
        assert otp.isdigit()


class TestPasswordNumberGenerator:
    def test_pin_length(self):
        pin = PasswordNumberGenerator.pin(4)
        assert len(pin) == 4
        assert pin.isdigit()

    def test_pin_secure(self):
        pins = [PasswordNumberGenerator.pin(4) for _ in range(20)]
        # Should have variety
        assert len(set(pins)) > 5

    def test_dice_roll(self):
        rolls = PasswordNumberGenerator.dice_roll(6, 100)
        assert len(rolls) == 100
        assert all(1 <= r <= 6 for r in rolls)

    def test_lottery_numbers(self):
        nums = PasswordNumberGenerator.lottery_numbers(49, 6)
        assert len(nums) == 6
        assert len(set(nums)) == 6
        assert all(1 <= n <= 49 for n in nums)
        assert nums == sorted(nums)

    def test_lottery_invalid(self):
        with pytest.raises(ValueError):
            PasswordNumberGenerator.lottery_numbers(5, 10)


class TestHashBasedGenerator:
    def test_reproducible(self):
        gen1 = HashBasedGenerator(seed=b"test_seed")
        gen2 = HashBasedGenerator(seed=b"test_seed")
        assert [gen1.next_int() for _ in range(10)] == [gen2.next_int() for _ in range(10)]

    def test_different_seeds(self):
        gen1 = HashBasedGenerator(seed=b"seed1")
        gen2 = HashBasedGenerator(seed=b"seed2")
        assert gen1.next_int() != gen2.next_int()

    def test_generate_range(self):
        gen = HashBasedGenerator(seed=b"test")
        nums = gen.generate(100, 0, 1000)
        assert all(0 <= n <= 1000 for n in nums)


class TestCryptoGeneratorFactory:
    def test_secure_random(self):
        gen = CryptoGeneratorFactory.secure_random()
        assert isinstance(gen, SecureRandomGenerator)

    def test_token_generator(self):
        gen = CryptoGeneratorFactory.token_generator()
        assert isinstance(gen, TokenGenerator)

    def test_uuid_generator(self):
        gen = CryptoGeneratorFactory.uuid_generator()
        assert isinstance(gen, UUIDGenerator)

    def test_generate_secure_numbers(self):
        nums = CryptoGeneratorFactory.generate_secure_numbers(20, 0, 100)
        assert len(nums) == 20
        assert all(0 <= n <= 100 for n in nums)
