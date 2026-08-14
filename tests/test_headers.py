import httpx
from pi_ai.utils.headers import headers_to_record, provider_headers_to_record


def test_headers_to_record_converts_httpx_headers():
    headers = httpx.Headers({"Content-Type": "application/json", "X-Foo": "bar"})
    result = headers_to_record(headers)
    assert result == {"content-type": "application/json", "x-foo": "bar"}


def test_headers_to_record_handles_empty_headers():
    assert headers_to_record(httpx.Headers()) == {}


def test_headers_to_record_accepts_plain_dict():
    assert headers_to_record({"a": "1", "b": "2"}) == {"a": "1", "b": "2"}


def test_provider_headers_to_record_drops_none_values():
    result = provider_headers_to_record({"a": "1", "b": None, "c": "3"})
    assert result == {"a": "1", "c": "3"}


def test_provider_headers_to_record_returns_none_for_none_input():
    assert provider_headers_to_record(None) is None


def test_provider_headers_to_record_returns_none_for_empty_dict():
    assert provider_headers_to_record({}) is None


def test_provider_headers_to_record_returns_none_when_all_values_are_none():
    assert provider_headers_to_record({"a": None, "b": None}) is None


def test_provider_headers_to_record_keeps_empty_string_values():
    # Only None is dropped; empty string is a legitimate (if odd) header value.
    result = provider_headers_to_record({"a": ""})
    assert result == {"a": ""}
