"""UUIDv7 generation tests.

Includes the Python port of `packages/ai/test/uuid.test.ts`.
"""

import re
import time

from pi_ai.utils import uuid as uuid_module
from pi_ai.utils.uuid import uuidv7

_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def test_uuidv7_has_correct_8_4_4_4_12_hex_shape():
    value = uuidv7()
    assert _UUID_PATTERN.match(value), value


def test_uuidv7_version_nibble_is_7():
    value = uuidv7()
    # The version nibble is the first hex digit of the third group.
    third_group = value.split("-")[2]
    assert third_group[0] == "7"


def test_uuidv7_variant_bits_are_10():
    value = uuidv7()
    # The variant is encoded in the top two bits of the first byte of the
    # fourth group, which must read as binary 10xxxxxx.
    fourth_group = value.split("-")[3]
    first_byte = int(fourth_group[0:2], 16)
    assert (first_byte & 0b11000000) == 0b10000000


def test_uuidv7_values_are_monotonically_increasing_in_a_tight_loop():
    values = [uuidv7() for _ in range(500)]
    assert values == sorted(values)
    # And strictly increasing, not just non-decreasing.
    assert len(set(values)) == len(values)


def test_uuidv7_uniqueness_across_many_calls():
    values = [uuidv7() for _ in range(2000)]
    assert len(set(values)) == len(values)


def test_uuidv7_embedded_timestamp_matches_current_time_within_tolerance():
    before_ms = int(time.time() * 1000)
    value = uuidv7()
    after_ms = int(time.time() * 1000)

    hex_digits = value.replace("-", "")
    embedded_ms = int(hex_digits[0:12], 16)

    tolerance_ms = 2000
    assert before_ms - tolerance_ms <= embedded_ms <= after_ms + tolerance_ms


# --------------------------------------------------------------------------
# Ported from `packages/ai/test/uuid.test.ts`
# --------------------------------------------------------------------------

_UUID_V7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_TIMESTAMP = 0x0123456789AB


def _parse_timestamp(value: str) -> int:
    return int(value.replace("-", "")[0:12], 16)


def test_uses_the_rfc_9562_layout_and_preserves_monotonic_order(monkeypatch):
    random_values = [
        bytes([0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFE, 0x01, 0x11, 0x22, 0x33, 0x44, 0x55]),
        bytes(16),
        bytes(16),
    ]
    calls = []

    def fake_token_bytes(n: int) -> bytes:
        calls.append(n)
        return random_values.pop(0) if random_values else bytes(n)

    monkeypatch.setattr(uuid_module.secrets, "token_bytes", fake_token_bytes)
    monkeypatch.setattr(uuid_module, "now_ms", lambda: _TIMESTAMP)
    monkeypatch.setattr(uuid_module, "_last_timestamp", -1)
    monkeypatch.setattr(uuid_module, "_sequence", 0)

    first = uuidv7()
    second = uuidv7()
    third = uuidv7()

    assert first == "01234567-89ab-7fff-bfff-f91122334455"
    assert second == "01234567-89ab-7fff-bfff-fc0000000000"
    assert third == "01234567-89ac-7000-8000-000000000000"
    assert _UUID_V7_RE.match(first)
    assert _UUID_V7_RE.match(second)
    assert _UUID_V7_RE.match(third)
    assert _parse_timestamp(first) == _TIMESTAMP
    assert _parse_timestamp(second) == _TIMESTAMP
    assert _parse_timestamp(third) == _TIMESTAMP + 1
    assert first < second
    assert second < third
    assert len(calls) == 3
