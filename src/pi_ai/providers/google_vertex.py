"""Google Vertex AI provider factory.

Python port of `packages/ai/src/providers/google-vertex.ts`. The model list
comes from the generated catalog shard `pi_ai/providers/data/google-vertex.json`,
the Python equivalent of TypeScript's generated `providers/google-vertex.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Vertex accepts an explicit API key or Application Default Credentials
(`gcloud auth application-default login`). ADC additionally requires project
and location env vars, which the API implementation reads itself; `login` asks
for whichever of the two the user picks.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from ..api import google_vertex
from ..auth.types import (
    ApiKeyAuth,
    AuthEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    Credential,
    EnvLookup,
    ProviderAuth,
    ResolvedAuth,
)
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

GOOGLE_VERTEX_MODELS: list[Model] = load_models("google-vertex")

VERTEX_ADC_PATH = "~/.config/gcloud/application_default_credentials.json"


async def _read_env(env: EnvLookup | None, name: str) -> str | None:
    lookup = env if env is not None else os.environ.get
    value = lookup(name)
    if inspect.isawaitable(value):
        value = await value
    return value or None


async def _resolve_vertex_auth(
    credential: Credential | None = None,
    env: EnvLookup | None = None,
) -> AuthResult | None:
    stored_key = credential.key if credential is not None else None
    key = stored_key or await _read_env(env, "GOOGLE_CLOUD_API_KEY")
    if key:
        return AuthResult(
            auth=ResolvedAuth(api_key=key),
            source="stored credential" if stored_key else "GOOGLE_CLOUD_API_KEY",
        )

    stored_env = dict(credential.env) if credential is not None else {}
    adc_path = stored_env.get("GOOGLE_APPLICATION_CREDENTIALS") or await _read_env(
        env, "GOOGLE_APPLICATION_CREDENTIALS"
    )
    has_credentials = Path(adc_path or VERTEX_ADC_PATH).expanduser().exists()
    project = (
        stored_env.get("GOOGLE_CLOUD_PROJECT")
        or await _read_env(env, "GOOGLE_CLOUD_PROJECT")
        or await _read_env(env, "GCLOUD_PROJECT")
    )
    location = stored_env.get("GOOGLE_CLOUD_LOCATION") or await _read_env(env, "GOOGLE_CLOUD_LOCATION")
    if has_credentials and project and location:
        return AuthResult(
            auth=ResolvedAuth(),
            source="stored credential" if credential is not None else "gcloud application default credentials",
            env=stored_env,
        )
    return None


async def _vertex_login(interaction: AuthInteraction) -> Credential:
    interaction.signal.throw_if_aborted()
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Google Vertex AI authentication method:",
            options=(
                {"id": "api-key", "label": "Google Cloud API key"},
                {"id": "adc", "label": "Application Default Credentials"},
                {"id": "service-account", "label": "Service account credentials file"},
            ),
        )
    )
    interaction.signal.throw_if_aborted()
    if method == "api-key":
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Google Cloud API key"))
        return Credential(type="api_key", key=key)
    if method not in ("adc", "service-account"):
        raise ValueError(f"Unknown Google Vertex AI auth method: {method}")

    interaction.notify(
        AuthEvent(
            type="info",
            message=(
                "Run `gcloud auth application-default login`, then provide the project and location."
                if method == "adc"
                else "Provide a service account credentials file, project, and location."
            ),
            links=(
                AuthInfoLink(
                    label="Application Default Credentials",
                    url="https://cloud.google.com/docs/authentication/provide-credentials-adc",
                ),
            ),
        )
    )
    project = await interaction.prompt(AuthPrompt(type="text", message="Enter Google Cloud project ID"))
    location = await interaction.prompt(AuthPrompt(type="text", message="Enter Google Cloud location"))
    env = {"GOOGLE_CLOUD_PROJECT": project, "GOOGLE_CLOUD_LOCATION": location}
    if method == "service-account":
        env["GOOGLE_APPLICATION_CREDENTIALS"] = await interaction.prompt(
            AuthPrompt(type="text", message="Enter service account credentials file path")
        )
    return Credential(type="api_key", env=env)


def vertex_auth() -> ApiKeyAuth:
    """Google Cloud API key or Application Default Credentials."""
    return ApiKeyAuth(name="Google Cloud credentials", resolve=_resolve_vertex_auth, login=_vertex_login)


def google_vertex_provider() -> Provider:
    """Build the built-in Google Vertex AI provider."""
    return create_provider(
        id="google-vertex",
        name="Google Vertex AI",
        auth=ProviderAuth(api_key=vertex_auth()),
        api=google_vertex,
        models=GOOGLE_VERTEX_MODELS,
    )
