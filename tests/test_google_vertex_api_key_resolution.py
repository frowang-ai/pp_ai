"""Python port of `packages/ai/test/google-vertex-api-key-resolution.test.ts`.

TypeScript mocks `@google/genai` and inspects the `GoogleGenAI` constructor
config (`vertexai`, `project`, `location`, `apiVersion`, `httpOptions`). This
port speaks HTTP directly and has no SDK client, so each case asserts on the
request URL and headers that `build_url_and_headers` produces — the observable
result of that constructor config after the SDK builds a request URL:

- `apiKey` on the client -> `x-goog-api-key` header and no project/location
  path segment.
- ADC -> `authorization: Bearer ...` and a `projects/<p>/locations/<l>` segment.
- `httpOptions.baseUrl` + `baseUrlResourceScope: COLLECTION` -> the custom base
  URL replaces the generated host *and* suppresses the project/location
  segment (`shouldPrependVertexProjectPath` returns false for COLLECTION).
- `httpOptions.apiVersion: ""` -> no extra `/v1` segment.

ADC itself has no Python analogue in this port (see the `google_vertex` module
docstring: minting a token from Application Default Credentials is out of
scope), so the ADC cases pass an explicit `access_token`, which is the seam
this port uses in its place.
"""

from __future__ import annotations

import dataclasses

import pytest
from pi_ai.api.google_vertex import (
    GoogleVertexOptions,
    _resolve_custom_base_url,
    build_url_and_headers,
    resolve_api_key,
)
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Model

MODEL = get_builtin_model("google-vertex", "gemini-3-flash-preview")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_API_KEY", raising=False)


def adc_options(**overrides: object) -> GoogleVertexOptions:
    return GoogleVertexOptions(
        project="test-project",
        location="us-central1",
        access_token="test-access-token",
        **overrides,
    )


def with_base_url(base_url: str) -> Model:
    return dataclasses.replace(MODEL, base_url=base_url)


def test_falls_back_to_adc_when_api_key_is_a_placeholder_marker():
    options = adc_options(api_key="<authenticated>")
    assert resolve_api_key(options) is None

    url, headers = build_url_and_headers(MODEL, options)
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1"
        "/projects/test-project/locations/us-central1"
        "/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    )
    assert headers["authorization"] == "Bearer test-access-token"
    assert "x-goog-api-key" not in headers


def test_falls_back_to_adc_when_api_key_is_the_gcp_vertex_credentials_marker():
    options = adc_options(api_key="gcp-vertex-credentials")
    assert resolve_api_key(options) is None

    url, headers = build_url_and_headers(MODEL, options)
    assert "/projects/test-project/locations/us-central1/" in url
    assert headers["authorization"] == "Bearer test-access-token"
    assert "x-goog-api-key" not in headers


def test_falls_back_to_adc_when_google_cloud_api_key_is_a_placeholder_marker(
    monkeypatch: pytest.MonkeyPatch,
):
    # `GOOGLE_CLOUD_API_KEY` is never consulted by the adapter itself, so a
    # placeholder value in the environment must not turn on the API-key client.
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "<authenticated>")

    options = adc_options()
    assert resolve_api_key(options) is None

    url, headers = build_url_and_headers(MODEL, options)
    assert "/projects/test-project/locations/us-central1/" in url
    assert headers["authorization"] == "Bearer test-access-token"
    assert "x-goog-api-key" not in headers


def test_still_uses_the_api_key_client_for_real_api_keys():
    options = GoogleVertexOptions(api_key="AIzaSyExampleRealisticLookingApiKey123456")
    assert resolve_api_key(options) == "AIzaSyExampleRealisticLookingApiKey123456"

    url, headers = build_url_and_headers(MODEL, options)
    assert url == (
        "https://aiplatform.googleapis.com/v1"
        "/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    )
    assert headers["x-goog-api-key"] == "AIzaSyExampleRealisticLookingApiKey123456"
    assert "authorization" not in headers
    assert "/projects/" not in url
    assert "/locations/" not in url


def test_does_not_forward_generated_vertex_base_url_placeholders():
    # The generated catalog entry carries a `{location}` placeholder base URL,
    # which is the TypeScript "no httpOptions at all" case.
    assert "{location}" in MODEL.base_url
    assert _resolve_custom_base_url(MODEL.base_url) is None

    url, _headers = build_url_and_headers(MODEL, adc_options())
    assert "{location}" not in url
    assert url.startswith("https://us-central1-aiplatform.googleapis.com/v1/")


def test_forwards_custom_base_url_to_the_adc_client():
    url, headers = build_url_and_headers(with_base_url("https://proxy.example.com"), adc_options())
    assert url == (
        "https://proxy.example.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    )
    assert headers["authorization"] == "Bearer test-access-token"


def test_forwards_custom_base_url_to_the_api_key_client():
    url, headers = build_url_and_headers(
        with_base_url("https://proxy.example.com"),
        GoogleVertexOptions(api_key="AIzaSyExampleRealisticLookingApiKey123456"),
    )
    assert url == (
        "https://proxy.example.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    )
    assert headers["x-goog-api-key"] == "AIzaSyExampleRealisticLookingApiKey123456"


def test_does_not_append_api_version_when_custom_base_url_already_includes_one():
    base_url = "https://proxy.example.com/v1/projects/test-project/locations/global"
    url, _headers = build_url_and_headers(with_base_url(base_url), adc_options())
    assert url == (f"{base_url}/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse")
