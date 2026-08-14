import json

import pytest
from pi_ai import parse_json_with_repair, parse_partial_json, parse_streaming_json, repair_json
from pi_ai.utils.json_parse import PartialJsonError


def test_repair_json_escapes_raw_control_characters():
    assert repair_json('{"a": "line\nbreak"}') == '{"a": "line\\nbreak"}'
    assert repair_json('{"a": "tab\there"}') == '{"a": "tab\\there"}'
    assert repair_json('{"a": "\x01"}') == '{"a": "\\u0001"}'


def test_repair_json_keeps_valid_escapes():
    source = '{"a": "quote\\"and\\\\slash\\u00e9"}'
    assert repair_json(source) == source
    assert json.loads(repair_json(source))["a"] == 'quote"and\\slashé'


def test_repair_json_doubles_invalid_escapes():
    assert repair_json('{"a": "c:\\path"}') == '{"a": "c:\\\\path"}'


def test_repair_json_keeps_short_unicode_escape_untouched():
    # "u" is in the valid-escape set, so a malformed \u sequence is preserved as-is.
    assert repair_json('{"a": "\\u12"}') == '{"a": "\\u12"}'


def test_repair_json_handles_trailing_backslash():
    assert repair_json('{"a": "x\\') == '{"a": "x\\\\'


def test_repair_json_leaves_structure_outside_strings():
    assert repair_json('{"a":\n1}') == '{"a":\n1}'


def test_parse_json_with_repair_falls_back_to_repair():
    assert parse_json_with_repair('{"a": "x\ny"}') == {"a": "x\ny"}


def test_parse_json_with_repair_reraises_when_repair_does_not_help():
    with pytest.raises(ValueError):
        parse_json_with_repair("{not json")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('{"a": 1', {"a": 1}),
        ('{"a": [1, 2', {"a": [1, 2]}),
        ('{"a": [1, 2],', {"a": [1, 2]}),
        ('{"a": "hel', {"a": "hel"}),
        ('{"a": tr', {}),
        ('{"a": true, "b"', {"a": True}),
        ('{"a": true, "b":', {"a": True}),
        ('{"a": {"b": {"c": ', {"a": {"b": {}}}),
        ("[1, 2, 3", [1, 2, 3]),
        ("[1, 2, 3.", [1, 2, 3]),
        ('["a", "b', ["a", "b"]),
        ("{", {}),
        ("[", []),
    ],
)
def test_parse_partial_json(text, expected):
    assert parse_partial_json(text) == expected


def test_parse_partial_json_rejects_empty_input():
    with pytest.raises(PartialJsonError):
        parse_partial_json("   ")


def test_parse_partial_json_decodes_escapes_in_truncated_strings():
    assert parse_partial_json('{"a": "x\\ny') == {"a": "x\ny"}
    assert parse_partial_json('{"a": "x\\') == {"a": "x"}
    assert parse_partial_json('{"a": "x\\u00e') == {"a": "x"}


def test_parse_streaming_json_returns_dict_for_empty_input():
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
    assert parse_streaming_json("   ") == {}


def test_parse_streaming_json_parses_complete_and_partial_payloads():
    assert parse_streaming_json('{"path": "a.txt", "content": "he') == {"path": "a.txt", "content": "he"}
    assert parse_streaming_json('{"path": "a.txt"}') == {"path": "a.txt"}


def test_parse_streaming_json_repairs_control_characters():
    assert parse_streaming_json('{"a": "x\ny"}') == {"a": "x\ny"}


def test_parse_streaming_json_returns_empty_dict_for_non_objects():
    assert parse_streaming_json("[1, 2]") == {}
    assert parse_streaming_json("hello") == {}


def test_parse_partial_json_handles_invalid_object_entries_after_valid_prefix():
    assert parse_partial_json('{"a": 1, invalid: 2}') == {"a": 1}
    assert parse_partial_json('{"a": 1,}') == {"a": 1}
    assert parse_partial_json('{"a": 1,, "b": 2}') == {"a": 1, "b": 2}
    assert parse_partial_json('{"a":, "b": 2}') == {}


def test_parse_partial_json_handles_array_edge_cases():
    assert parse_partial_json("[1] trailing") == [1]
    assert parse_partial_json("[1,]") == [1]
    assert parse_partial_json("[1,,2]") == [1, 2]
    assert parse_partial_json("[nonsense]") == []


def test_parse_partial_json_keeps_invalid_escapes_literal_inside_strings():
    assert parse_partial_json(r'{"a": "bad\xescape"}') == {"a": r"bad\xescape"}


def test_parse_partial_json_decodes_valid_unicode_escape_in_truncated_string():
    assert parse_partial_json('{"a": "\\u0041') == {"a": "A"}
