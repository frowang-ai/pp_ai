"""Fast deterministic hash to shorten long strings.

Python port of `packages/ai/src/utils/hash.ts`. The TypeScript implementation
hashes UTF-16 code units (`String.prototype.charCodeAt`) using 32-bit signed
multiplication (`Math.imul`) and unsigned right shift (`>>>`). To produce byte-
identical output, this port iterates over the UTF-16LE code units of the input
(so non-BMP characters, encoded as UTF-16 surrogate pairs, hash identically to
the JS version) and masks every intermediate value to 32 bits.
"""

from __future__ import annotations

_MASK32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Emulate JavaScript's `Math.imul`: 32-bit signed integer multiplication."""
    result = (a * b) & _MASK32
    if result >= 0x80000000:
        result -= 0x100000000
    return result


def _to_uint32(value: int) -> int:
    return value & _MASK32


def _utf16_code_units(text: str) -> list[int]:
    """UTF-16 code units of ``text``, matching JavaScript `charCodeAt` per index."""
    raw = text.encode("utf-16-le", "surrogatepass")
    return [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2)]


def short_hash(text: str) -> str:
    """Hash ``text`` to a short base-36 string, matching the JS `shortHash`."""
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    for ch in _utf16_code_units(text):
        h1 = _to_uint32(_imul(h1 ^ ch, 2654435761))
        h2 = _to_uint32(_imul(h2 ^ ch, 1597334677))

    h1 = _to_uint32(_imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909))
    h2 = _to_uint32(_imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(h1 ^ (h1 >> 13), 3266489909))

    return _base36(_to_uint32(h2)) + _base36(_to_uint32(h1))


_BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value > 0:
        value, remainder = divmod(value, 36)
        digits.append(_BASE36_DIGITS[remainder])
    return "".join(reversed(digits))
