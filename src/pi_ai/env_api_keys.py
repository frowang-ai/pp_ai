"""Well-known provider environment variable lookup.

Python port of `packages/ai/src/env-api-keys.ts`.

The TypeScript version distinguishes `KnownProvider` (a generated literal
union of every catalog provider id) from a bare `string`; this port has no
generated provider-id union (see the pp README's "Generated model catalogs"
note), so every function here takes a plain ``str`` provider id.
"""

from __future__ import annotations

from pathlib import Path

from .utils.provider_env import get_provider_env_value

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

_cached_vertex_adc_credentials_exists: bool | None = None


def _has_vertex_adc_credentials(env: dict[str, str] | None = None) -> bool:
    global _cached_vertex_adc_credentials_exists

    explicit_credentials_path = (env or {}).get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_credentials_path:
        return Path(explicit_credentials_path).exists()

    if _cached_vertex_adc_credentials_exists is None:
        gac_path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
        if gac_path:
            _cached_vertex_adc_credentials_exists = Path(gac_path).exists()
        else:
            _cached_vertex_adc_credentials_exists = (
                Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            ).exists()
    return _cached_vertex_adc_credentials_exists


# Provider id -> its single api-key env var, for providers with standard
# single-env-var resolution. Providers needing multiple candidate env vars
# (github-copilot, anthropic) or ambient-credential checks (google-vertex,
# amazon-bedrock) are handled separately below.
_ENV_VAR_MAP: dict[str, str] = {
    "ant-ling": "ANT_LING_API_KEY",
    "qwen-token-plan": "QWEN_TOKEN_PLAN_API_KEY",
    "qwen-token-plan-cn": "QWEN_TOKEN_PLAN_CN_API_KEY",
    "qwen-token-plan-individual": "QWEN_TOKEN_PLAN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "radius": "RADIUS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "zai-coding-cn": "ZAI_CODING_CN_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshotai-cn": "MOONSHOT_API_KEY",
    "huggingface": "HF_TOKEN",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
    "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
    "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
    "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
}


def _get_api_key_env_vars(provider: str) -> tuple[str, ...] | None:
    if provider == "github-copilot":
        return ("COPILOT_GITHUB_TOKEN",)

    # ANTHROPIC_AUTH_TOKEN participates in env discovery/status, but
    # get_env_api_key() skips it because requests must pass it as
    # `Authorization: Bearer`, not as a bare api key.
    if provider == "anthropic":
        return (ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV)

    env_var = _ENV_VAR_MAP.get(provider)
    return (env_var,) if env_var else None


def find_env_keys(provider: str, env: dict[str, str] | None = None) -> list[str] | None:
    """Find configured environment variables that can provide an API key for a provider.

    This only reports actual API key variables. It intentionally excludes
    ambient credential sources such as AWS profiles, AWS IAM credentials, and
    Google Application Default Credentials.
    """
    env_vars = _get_api_key_env_vars(provider)
    if env_vars is None:
        return None

    found = [env_var for env_var in env_vars if get_provider_env_value(env_var, env)]
    return found or None


def get_env_api_key(provider: str, env: dict[str, str] | None = None) -> str | None:
    """Get the API key for a provider from known environment variables, e.g. ``OPENAI_API_KEY``.

    Will not return API keys for providers that require OAuth tokens.
    """
    env_keys = find_env_keys(provider, env)
    if env_keys:
        api_key_env = (
            next((key for key in env_keys if key != ANTHROPIC_AUTH_TOKEN_ENV), None)
            if provider == "anthropic"
            else env_keys[0]
        )
        if api_key_env:
            value = get_provider_env_value(api_key_env, env)
            if value:
                return value

    # Vertex AI supports either an explicit API key or Application Default
    # Credentials. Auth is configured via `gcloud auth application-default login`.
    if provider == "google-vertex":
        has_credentials = _has_vertex_adc_credentials(env)
        has_project = bool(
            get_provider_env_value("GOOGLE_CLOUD_PROJECT", env) or get_provider_env_value("GCLOUD_PROJECT", env)
        )
        has_location = bool(get_provider_env_value("GOOGLE_CLOUD_LOCATION", env))
        if has_credentials and has_project and has_location:
            return "<authenticated>"

    if provider == "amazon-bedrock" and (
        get_provider_env_value("AWS_PROFILE", env)
        or (get_provider_env_value("AWS_ACCESS_KEY_ID", env) and get_provider_env_value("AWS_SECRET_ACCESS_KEY", env))
        or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
        or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
        or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env)
        or get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env)
    ):
        # Amazon Bedrock supports multiple credential sources:
        # 1. AWS_PROFILE - named profile from ~/.aws/credentials
        # 2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY - standard IAM keys
        # 3. AWS_BEARER_TOKEN_BEDROCK - Bedrock bearer token
        # 4. AWS_CONTAINER_CREDENTIALS_RELATIVE_URI - ECS task roles
        # 5. AWS_CONTAINER_CREDENTIALS_FULL_URI - ECS task roles (full URI)
        # 6. AWS_WEB_IDENTITY_TOKEN_FILE - IRSA (IAM Roles for Service Accounts)
        return "<authenticated>"

    return None
