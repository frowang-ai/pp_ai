"""Time-ordered UUIDv7 generation.

Python port of `packages/ai/src/utils/uuid.ts`.
"""

from __future__ import annotations

import secrets
import threading

from ..types import now_ms

_lock = threading.Lock()
_last_timestamp = -1
_sequence = 0


def uuidv7() -> str:
    """Generate a time-ordered UUIDv7 string."""
    global _last_timestamp, _sequence

    random = secrets.token_bytes(16)
    timestamp = now_ms()

    with _lock:
        if timestamp > _last_timestamp:
            _sequence = random[6] * 0x1000000 + random[7] * 0x10000 + random[8] * 0x100 + random[9]
            _last_timestamp = timestamp
        else:
            _sequence = (_sequence + 1) & 0xFFFFFFFF
            if _sequence == 0:
                _last_timestamp += 1
        ts = _last_timestamp
        sequence = _sequence

    data = bytearray(16)
    data[0] = (ts // 0x10000000000) & 0xFF
    data[1] = (ts // 0x100000000) & 0xFF
    data[2] = (ts // 0x1000000) & 0xFF
    data[3] = (ts // 0x10000) & 0xFF
    data[4] = (ts // 0x100) & 0xFF
    data[5] = ts & 0xFF
    data[6] = 0x70 | ((sequence >> 28) & 0x0F)
    data[7] = (sequence >> 20) & 0xFF
    data[8] = 0x80 | ((sequence >> 14) & 0x3F)
    data[9] = (sequence >> 6) & 0xFF
    data[10] = ((sequence & 0x3F) << 2) | (random[10] & 0x03)
    data[11:16] = random[11:16]

    hex_digits = data.hex()
    return "-".join(
        [
            hex_digits[0:8],
            hex_digits[8:12],
            hex_digits[12:16],
            hex_digits[16:20],
            hex_digits[20:32],
        ]
    )
