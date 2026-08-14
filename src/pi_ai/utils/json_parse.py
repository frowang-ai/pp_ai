"""Tolerant JSON parsing for streamed model output.

Python port of `packages/ai/src/utils/json-parse.ts`. The TypeScript version
delegates partial parsing to the `partial-json` npm package; this module
implements equivalent behaviour with a small recursive-descent parser that
returns whatever prefix of the document is well formed.
"""

from __future__ import annotations

import json
from typing import Any

VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')

_CONTROL_ESCAPES = {
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}

_WHITESPACE = " \t\n\r"
_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _is_control_character(char: str) -> bool:
    return 0x00 <= ord(char) <= 0x1F


def _escape_control_character(char: str) -> str:
    escaped = _CONTROL_ESCAPES.get(char)
    if escaped is not None:
        return escaped
    return f"\\u{ord(char):04x}"


def repair_json(text: str) -> str:
    """Escape raw control characters and invalid escapes inside string literals."""
    repaired: list[str] = []
    in_string = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            next_char = text[index + 1] if index + 1 < length else None
            if next_char is None:
                repaired.append("\\\\")
                index += 1
                continue

            if next_char == "u":
                unicode_digits = text[index + 2 : index + 6]
                if len(unicode_digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in unicode_digits):
                    repaired.append(f"\\u{unicode_digits}")
                    index += 6
                    continue

            if next_char in VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                index += 2
                continue

            repaired.append("\\\\")
            index += 1
            continue

        repaired.append(_escape_control_character(char) if _is_control_character(char) else char)
        index += 1

    return "".join(repaired)


def parse_json_with_repair(text: str) -> Any:
    """Parse ``text``, retrying once with :func:`repair_json` on failure."""
    try:
        return json.loads(text)
    except ValueError:
        repaired = repair_json(text)
        if repaired != text:
            return json.loads(repaired)
        raise


class PartialJsonError(ValueError):
    """Raised when a partial document has no parseable prefix at all."""


_MISSING = object()


class _PartialParser:
    """Recursive-descent parser that tolerates truncation at any point."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.terminated = False

    def parse(self) -> Any:
        self._skip_whitespace()
        value = self._parse_value()
        if value is _MISSING:
            raise PartialJsonError("no parseable JSON value")
        return value

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in _WHITESPACE:
            self.pos += 1

    def _peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _parse_value(self) -> Any:
        char = self._peek()
        if char is None:
            return _MISSING
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char == '"':
            return self._parse_string()
        return self._parse_literal()

    def _parse_object(self) -> dict[str, Any]:
        self.pos += 1  # consume "{"
        result: dict[str, Any] = {}
        while True:
            self._skip_whitespace()
            char = self._peek()
            if char is None:
                return result
            if char == "}":
                self.pos += 1
                return result
            if char == ",":
                self.pos += 1
                continue
            if char != '"':
                return result

            key = self._parse_string()
            if not self.terminated:
                return result

            self._skip_whitespace()
            if self._peek() != ":":
                return result
            self.pos += 1

            self._skip_whitespace()
            value = self._parse_value()
            if value is _MISSING:
                return result
            result[key] = value

            self._skip_whitespace()
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == "}":
                self.pos += 1
            return result

    def _parse_array(self) -> list[Any]:
        self.pos += 1  # consume "["
        result: list[Any] = []
        while True:
            self._skip_whitespace()
            char = self._peek()
            if char is None:
                return result
            if char == "]":
                self.pos += 1
                return result
            if char == ",":
                self.pos += 1
                continue

            value = self._parse_value()
            if value is _MISSING:
                return result
            result.append(value)

            self._skip_whitespace()
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == "]":
                self.pos += 1
            return result

    def _parse_string(self) -> str:
        self.pos += 1  # consume opening quote
        chars: list[str] = []
        text = self.text
        length = len(text)
        self.terminated = False
        while self.pos < length:
            char = text[self.pos]
            if char == "\\":
                if self.pos + 1 >= length:
                    self.pos += 1  # dangling escape: drop it
                    return "".join(chars)
                decoded = self._decode_escape(text[self.pos + 1])
                if decoded is None:
                    return "".join(chars)
                chars.append(decoded)
                continue
            if char == '"':
                self.pos += 1
                self.terminated = True
                return "".join(chars)
            chars.append(char)
            self.pos += 1
        return "".join(chars)

    def _decode_escape(self, escape: str) -> str | None:
        if escape in _SIMPLE_ESCAPES:
            self.pos += 2
            return _SIMPLE_ESCAPES[escape]
        if escape == "u":
            digits = self.text[self.pos + 2 : self.pos + 6]
            if len(digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in digits):
                self.pos += 6
                return chr(int(digits, 16))
            self.pos = len(self.text)
            return None
        # Invalid escape: keep the backslash literally, matching repair_json.
        self.pos += 2
        return "\\" + escape

    def _parse_literal(self) -> Any:
        start = self.pos
        text = self.text
        length = len(text)
        stop = ',:{}[]"' + _WHITESPACE
        while self.pos < length and text[self.pos] not in stop:
            self.pos += 1
        token = text[start : self.pos]
        if not token:
            self.pos += 1
            return _MISSING
        for end in range(len(token), 0, -1):
            try:
                value = json.loads(token[:end])
            except ValueError:
                continue
            self.pos = start + end
            return value
        return _MISSING


def parse_partial_json(text: str) -> Any:
    """Parse a possibly truncated JSON document.

    Incomplete trailing values are dropped and open containers/strings are kept
    with the text received so far, so ``'{"a": [1, 2'`` parses to
    ``{"a": [1, 2]}``.
    """
    stripped = text.strip()
    if not stripped:
        raise PartialJsonError("empty input")
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    return _PartialParser(stripped).parse()


def parse_streaming_json(partial_json: str | None) -> dict[str, Any]:
    """Best-effort parse of streamed JSON; always returns a dict."""
    if not partial_json or not partial_json.strip():
        return {}

    try:
        result = parse_json_with_repair(partial_json)
        return result if isinstance(result, dict) else {}
    except ValueError:
        pass

    for candidate in (partial_json, repair_json(partial_json)):
        try:
            result = parse_partial_json(candidate)
        except ValueError:
            continue
        return result if isinstance(result, dict) else {}
    return {}
