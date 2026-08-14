"""Tests for `short_hash`.

Expected values were computed by running the original TypeScript implementation
(`packages/ai/src/utils/hash.ts`) directly with `node` for ASCII, accented,
CJK, emoji, and long-string inputs, to guarantee byte-identical output between
the two ports.
"""

import pytest
from pi_ai.utils.hash import short_hash


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "k4n83c7h0j2b"),
        ("a", "m8735310ae7sx"),
        ("hello", "1h6qa0qrowduu"),
        ("Hello, World!", "1r5jexi1bwk9ze"),
        ("café résumé", "1qa1zsmb50ba4"),
        ("こんにちは世界", "1dfe7mf18orugv"),
        ("🙈🙉🙊", "1pd5f9x1j6a281"),
        ("The quick brown fox jumps over the lazy dog", "eig47k1th3xf1"),
        ("a" * 1000, "kli8eammh8ym"),
    ],
)
def test_short_hash_matches_javascript_reference(text, expected):
    assert short_hash(text) == expected


def test_short_hash_is_deterministic():
    assert short_hash("some repeated text") == short_hash("some repeated text")


def test_short_hash_differs_for_different_inputs():
    assert short_hash("abc") != short_hash("abd")


def test_short_hash_is_sensitive_to_order():
    assert short_hash("ab") != short_hash("ba")


def test_short_hash_handles_non_bmp_characters_as_surrogate_pairs():
    # An emoji (non-BMP) must hash the same as its UTF-16 surrogate-pair code
    # units would in JavaScript's `charCodeAt`, not as a single Python code point.
    assert short_hash("🙈") == "kphsz0153ms3q"


def test_short_hash_output_is_lowercase_base36():
    result = short_hash("anything")
    assert all(c in "0123456789abcdefghijklmnopqrstuvwxyz" for c in result)
