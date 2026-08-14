"""Removes unpaired Unicode surrogate characters from a string.

Python port of `packages/ai/src/utils/sanitize-unicode.ts`.

Unpaired surrogates (high surrogates 0xD800-0xDBFF without a matching low
surrogate 0xDC00-0xDFFF, or vice versa) cause JSON serialization errors in many
API providers. Python strings can contain such lone surrogates when decoded
from data that originated as JavaScript UTF-16 (for example ``\\ud83d`` with no
paired low surrogate).

A real emoji or other non-BMP character is a single Python code point (Python
strings are sequences of code points, not UTF-16 code units), so it is
unaffected by this function. When a string does contain a *literal* adjacent
high+low surrogate pair (for example produced by decoding JS-originated data
one UTF-16 code unit at a time), this function preserves the pair as-is rather
than combining it into the single code point it would represent in UTF-16 -
matching the TypeScript behavior of only stripping *unpaired* surrogates and
leaving paired ones untouched.
"""

from __future__ import annotations

import re

_UNPAIRED_SURROGATE_PATTERN = re.compile("[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]")


def sanitize_surrogates(text: str) -> str:
    """Remove unpaired surrogate code points from ``text``, keeping paired ones."""
    return _UNPAIRED_SURROGATE_PATTERN.sub("", text)
