"""
Encoder module - Various encoding formats for number sequences.
Base64, Morse, Custom Base, URL encoding, etc.
"""

import base64
import math
import struct
from typing import List, Union


def to_base64(data: List[Union[int, float]]) -> str:
    """Encode list of numbers to base64 string."""
    # Pack integers as 4-byte little-endian
    int_data = [int(x) & 0xFFFFFFFF for x in data]
    packed = struct.pack(f"<{len(int_data)}I", *int_data)
    return base64.b64encode(packed).decode("utf-8")


def from_base64(encoded: str) -> List[int]:
    """Decode base64 string back to list of integers."""
    packed = base64.b64decode(encoded.encode("utf-8"))
    count = len(packed) // 4
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}I", packed[:count * 4]))


def to_base64_urlsafe(data: List[Union[int, float]]) -> str:
    """Encode to URL-safe base64."""
    int_data = [int(x) & 0xFFFFFFFF for x in data]
    packed = struct.pack(f"<{len(int_data)}I", *int_data)
    return base64.urlsafe_b64encode(packed).decode("utf-8")


def from_base64_urlsafe(encoded: str) -> List[int]:
    """Decode URL-safe base64."""
    packed = base64.urlsafe_b64decode(encoded.encode("utf-8"))
    count = len(packed) // 4
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}I", packed[:count * 4]))


MORSE_DIGITS = {
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}
MORSE_REVERSE = {v: k for k, v in MORSE_DIGITS.items()}


def numbers_to_morse(data: List[int]) -> str:
    """Encode list of integers as morse code."""
    groups = []
    for n in data:
        digit_morse = [MORSE_DIGITS[d] for d in str(abs(n))]
        groups.append(" ".join(digit_morse))
    return " / ".join(groups)


def morse_to_numbers(morse: str) -> List[int]:
    """Decode morse code to list of integers."""
    result = []
    groups = morse.split(" / ")
    for group in groups:
        codes = group.split(" ")
        digits = "".join(MORSE_REVERSE.get(c, "?") for c in codes if c)
        try:
            result.append(int(digits))
        except ValueError:
            result.append(0)
    return result


def to_custom_base(data: List[int], base: int, alphabet: str) -> List[str]:
    """Encode list of integers using a custom base and alphabet."""
    if len(alphabet) < base:
        raise ValueError(f"Alphabet must have at least {base} characters")

    def encode_one(n: int) -> str:
        if n == 0:
            return alphabet[0]
        negative = n < 0
        n = abs(n)
        digits = []
        while n:
            digits.append(alphabet[n % base])
            n //= base
        if negative:
            digits.append("-")
        return "".join(reversed(digits))

    return [encode_one(n) for n in data]


def from_custom_base(encoded: List[str], base: int, alphabet: str) -> List[int]:
    """Decode custom base encoded strings."""
    def decode_one(s: str) -> int:
        negative = s.startswith("-")
        s = s.lstrip("-")
        result = 0
        for char in s:
            result = result * base + alphabet.index(char)
        return -result if negative else result

    return [decode_one(s) for s in encoded]


def run_length_encode(data: List) -> List[tuple]:
    """Run-length encoding: compress runs of identical values."""
    if not data:
        return []
    result = []
    current = data[0]
    count = 1
    for x in data[1:]:
        if x == current:
            count += 1
        else:
            result.append((current, count))
            current = x
            count = 1
    result.append((current, count))
    return result


def run_length_decode(encoded: List[tuple]) -> List:
    """Decode run-length encoded data."""
    result = []
    for value, count in encoded:
        result.extend([value] * count)
    return result


def elias_gamma_encode(data: List[int]) -> str:
    """Elias gamma coding for positive integers."""
    result = []
    for n in data:
        if n <= 0:
            raise ValueError("Elias gamma coding only works for positive integers")
        k = int(math.floor(math.log2(n)))
        result.append("0" * k + bin(n)[2:])
    return "".join(result)


def elias_delta_encode(data: List[int]) -> str:
    """Elias delta coding for positive integers."""
    result = []
    for n in data:
        if n <= 0:
            raise ValueError("Elias delta coding only works for positive integers")
        k = int(math.floor(math.log2(n)))
        # Encode k+1 in Elias gamma, then the remaining bits of n
        k1 = k + 1
        k1_k = int(math.floor(math.log2(k1)))
        gamma = "0" * k1_k + bin(k1)[2:]
        remaining = bin(n)[3:] if k > 0 else ""
        result.append(gamma + remaining)
    return "".join(result)


def to_hex_sequence(data: List[int]) -> List[str]:
    """Convert integers to hex strings."""
    return [hex(n) for n in data]


def to_binary_sequence(data: List[int]) -> List[str]:
    """Convert integers to binary strings."""
    return [bin(n) for n in data]


def to_csv_string(data: List, delimiter: str = ",") -> str:
    """Convert list to CSV string."""
    return delimiter.join(str(x) for x in data)


def from_csv_string(s: str, delimiter: str = ",", dtype=float) -> List:
    """Parse CSV string to list."""
    return [dtype(x.strip()) for x in s.split(delimiter) if x.strip()]


def to_json_array(data: List) -> str:
    """Convert list to JSON array string."""
    import json
    return json.dumps(data)


def from_json_array(s: str) -> List:
    """Parse JSON array string."""
    import json
    return json.loads(s)


class Encoder:
    """Class-based interface for encoding operations."""

    @staticmethod
    def base64_encode(data: List[Union[int, float]]) -> str:
        return to_base64(data)

    @staticmethod
    def base64_decode(encoded: str) -> List[int]:
        return from_base64(encoded)

    @staticmethod
    def morse_encode(data: List[int]) -> str:
        return numbers_to_morse(data)

    @staticmethod
    def morse_decode(morse: str) -> List[int]:
        return morse_to_numbers(morse)

    @staticmethod
    def rle_encode(data: List) -> List[tuple]:
        return run_length_encode(data)

    @staticmethod
    def rle_decode(encoded: List[tuple]) -> List:
        return run_length_decode(encoded)

    @staticmethod
    def custom_base(data: List[int], base: int, alphabet: str) -> List[str]:
        return to_custom_base(data, base, alphabet)

    @staticmethod
    def hex_encode(data: List[int]) -> List[str]:
        return to_hex_sequence(data)

    @staticmethod
    def binary_encode(data: List[int]) -> List[str]:
        return to_binary_sequence(data)

    @staticmethod
    def to_csv(data: List, delimiter: str = ",") -> str:
        return to_csv_string(data, delimiter)

    @staticmethod
    def from_csv(s: str, delimiter: str = ",") -> List[float]:
        return from_csv_string(s, delimiter)
