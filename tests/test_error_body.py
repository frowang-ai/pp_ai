import httpx
from pi_ai.utils.error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    NormalizedProviderError,
    format_provider_error,
    normalize_provider_error,
    safe_json_stringify,
    truncate_error_text,
)


class _SdkErrorLike(Exception):
    def __init__(self, message, **attrs):
        super().__init__(message)
        for key, value in attrs.items():
            setattr(self, key, value)


def test_normalize_non_exception_uses_safe_json_stringify():
    norm = normalize_provider_error({"foo": "bar"})
    assert norm.message == '{"foo": "bar"}'
    assert norm.status is None
    assert norm.body is None
    assert norm.message_carries_body is False


def test_normalize_plain_error_with_no_extra_fields():
    norm = normalize_provider_error(ValueError("boom"))
    assert norm.message == "boom"
    assert norm.status is None
    assert norm.body is None
    # No body extracted => treated as message-carries-body (nothing extra to add).
    assert norm.message_carries_body is True


def test_normalize_extracts_status_code_attribute_mistral_style():
    error = _SdkErrorLike("oops", status_code=403)
    norm = normalize_provider_error(error)
    assert norm.status == 403


def test_normalize_extracts_status_attribute_openai_style():
    error = _SdkErrorLike("oops", status=429)
    norm = normalize_provider_error(error)
    assert norm.status == 429


def test_normalize_prefers_status_code_over_status():
    error = _SdkErrorLike("oops", status_code=403, status=429)
    norm = normalize_provider_error(error)
    assert norm.status == 403


def test_normalize_extracts_status_from_httpx_style_response():
    response = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))
    error = _SdkErrorLike("oops", response=response)
    norm = normalize_provider_error(error)
    assert norm.status == 500


def test_normalize_ignores_non_numeric_status_fields():
    error = _SdkErrorLike("oops", status_code="not-a-number")
    norm = normalize_provider_error(error)
    assert norm.status is None


def test_normalize_extracts_body_string_attribute():
    error = _SdkErrorLike("403 status code (no body)", status_code=403, body="Forbidden: bad token")
    norm = normalize_provider_error(error)
    assert norm.body == "Forbidden: bad token"
    assert norm.message_carries_body is False


def test_normalize_extracts_body_from_parsed_error_object():
    error = _SdkErrorLike("Unknown: UnknownError", error={"type": "invalid_request", "detail": "bad field"})
    norm = normalize_provider_error(error)
    assert norm.body == safe_json_stringify({"type": "invalid_request", "detail": "bad field"})


def test_normalize_ignores_empty_error_object():
    error = _SdkErrorLike("oops", error={})
    norm = normalize_provider_error(error)
    assert norm.body is None


def test_normalize_ignores_non_plain_error_object_instance():
    class CustomError:
        def __init__(self):
            self.detail = "x"

    error = _SdkErrorLike("oops", error=CustomError())
    norm = normalize_provider_error(error)
    assert norm.body is None


def test_normalize_extracts_body_from_response_body_string():
    class FakeResponse:
        body = "raw response body"

    error = _SdkErrorLike("oops", response=FakeResponse())
    norm = normalize_provider_error(error)
    assert norm.body == "raw response body"


def test_normalize_extracts_body_from_response_body_dict():
    class FakeResponse:
        def __init__(self):
            self.body = {"message": "bad request"}

    error = _SdkErrorLike("oops", response=FakeResponse())
    norm = normalize_provider_error(error)
    assert norm.body == safe_json_stringify({"message": "bad request"})


def test_normalize_ignores_stream_like_response_body():
    class StreamLike:
        def read(self):
            raise NotImplementedError

    class FakeResponse:
        body = StreamLike()

    error = _SdkErrorLike("oops", response=FakeResponse())
    norm = normalize_provider_error(error)
    assert norm.body is None


def test_normalize_extracts_httpx_response_text():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(403, request=request, text="access denied")
    error = _SdkErrorLike("403 status code (no body)", response=response)
    norm = normalize_provider_error(error)
    assert norm.body == "access denied"
    assert norm.status == 403


def test_normalize_body_truncated_to_cap():
    long_body = "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 500)
    error = _SdkErrorLike("oops", body=long_body)
    norm = normalize_provider_error(error)
    assert norm.body is not None
    assert norm.body.startswith("x" * 10)
    assert "truncated 500 chars" in norm.body


def test_normalize_body_whitespace_only_is_treated_as_no_body():
    error = _SdkErrorLike("oops", body="   ")
    norm = normalize_provider_error(error)
    assert norm.body is None


def test_message_carries_body_true_when_message_already_contains_body():
    error = _SdkErrorLike("request failed: bad token", body="bad token")
    norm = normalize_provider_error(error)
    assert norm.message_carries_body is True


def test_format_provider_error_returns_message_when_message_carries_body():
    norm = NormalizedProviderError(message="already has body", status=403, body="bad token", message_carries_body=True)
    assert format_provider_error(norm) == "already has body"


def test_format_provider_error_with_prefix_when_message_carries_body():
    norm = NormalizedProviderError(message="already has body", status=403, body="bad token", message_carries_body=True)
    assert format_provider_error(norm, "OpenAI") == "OpenAI (403): already has body"


def test_format_provider_error_no_prefix_status_and_body():
    norm = NormalizedProviderError(message="msg", status=403, body="Forbidden", message_carries_body=False)
    assert format_provider_error(norm) == "403: Forbidden"


def test_format_provider_error_with_prefix_status_and_body():
    norm = NormalizedProviderError(message="msg", status=403, body="Forbidden", message_carries_body=False)
    assert format_provider_error(norm, "Mistral") == "Mistral (403): Forbidden"


def test_format_provider_error_falls_back_to_message_when_no_status():
    norm = NormalizedProviderError(message="plain message", status=None, body=None, message_carries_body=False)
    assert format_provider_error(norm) == "plain message"
    assert format_provider_error(norm, "Prefix") == "plain message"


def test_format_provider_error_falls_back_to_message_when_no_body():
    norm = NormalizedProviderError(message="plain message", status=500, body=None, message_carries_body=False)
    assert format_provider_error(norm, "Prefix") == "Prefix (500): plain message"


def test_truncate_error_text_leaves_short_text_untouched():
    assert truncate_error_text("short", 100) == "short"


def test_truncate_error_text_truncates_and_reports_remainder():
    text = "a" * 20
    result = truncate_error_text(text, 10)
    assert result == "aaaaaaaaaa... [truncated 10 chars]"


def test_safe_json_stringify_serializes_plain_values():
    assert safe_json_stringify({"a": 1}) == '{"a": 1}'


def test_safe_json_stringify_falls_back_to_str_for_unserializable_values():
    class Unserializable:
        def __repr__(self):
            return "<Unserializable>"

    assert safe_json_stringify(Unserializable()) == "<Unserializable>"
