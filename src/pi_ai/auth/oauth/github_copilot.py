"""GitHub Copilot OAuth flow.

Python port of `packages/ai/src/auth/oauth/github-copilot.ts`. After login the
flow enables every catalog model for the user's account
(`enableAllGitHubCopilotModels`), reading the same generated
`github-copilot` catalog shard that the provider uses.
"""

from __future__ import annotations

import asyncio
import base64
import re
from urllib.parse import urlparse

import httpx

from ...model_catalog import load_models
from ...utils.abort import AbortSignal
from ...utils.url import normalize_http_url
from ..types import AuthEvent, AuthInteraction, AuthPrompt, Credential, OAuthAuth, ResolvedAuth
from .device_code import REAL_CLOCK, DeviceCodeClock, DeviceCodePollResult, poll_oauth_device_code_flow

CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode("ascii")

COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
COPILOT_API_VERSION = "2026-06-01"
REQUEST_TIMEOUT_S = 30.0

COPILOT_POLICY_CONCURRENCY = 4
DEFAULT_MODELS_TIMEOUT_S = 5.0


def normalize_domain(value: str) -> str | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    candidate = trimmed if "://" in trimmed else f"https://{trimmed}"
    parsed = urlparse(candidate)
    return parsed.hostname


def get_urls(domain: str) -> dict[str, str]:
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def get_base_url_from_token(token: str) -> str | None:
    """Parse the proxy-ep from a Copilot token and convert to an API base URL.

    Token format: ``tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...``
    """
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    api_host = re.sub(r"^proxy\.", "api.", proxy_host)
    return f"https://{api_host}"


def get_github_copilot_base_url(token: str | None = None, enterprise_domain: str | None = None) -> str:
    if token:
        url_from_token = get_base_url_from_token(token)
        if url_from_token:
            return url_from_token
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return "https://api.individual.githubcopilot.com"


async def _fetch_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: dict[str, str] | None = None,
    json_body: dict[str, object] | None = None,
    client: httpx.AsyncClient | None = None,
) -> object:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.request(method, url, headers=headers, data=data, json=json_body)
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} {response.reason_phrase}: {response.text}")
        return response.json()
    finally:
        if owns_client:
            await http_client.aclose()


def parse_available_copilot_model_ids(raw: object, allow_policy_fallback: bool) -> list[str]:
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("Invalid Copilot models response")

    picker_ids: list[str] = []
    policy_enabled_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        supports = capabilities.get("supports") if isinstance(capabilities.get("supports"), dict) else {}
        if supports.get("tool_calls") is False:
            continue
        policy = item.get("policy") if isinstance(item.get("policy"), dict) else {}
        if item.get("model_picker_enabled") is True and policy.get("state") != "disabled":
            picker_ids.append(model_id)
        if policy.get("state") == "enabled":
            policy_enabled_ids.append(model_id)
    return picker_ids if picker_ids or not allow_policy_fallback else policy_enabled_ids


async def fetch_available_github_copilot_model_ids(
    copilot_token: str, enterprise_domain: str | None = None, client: httpx.AsyncClient | None = None
) -> list[str]:
    base_url = get_github_copilot_base_url(copilot_token, enterprise_domain)
    # Some Individual accounts return false for every picker flag despite explicit enabled policies.
    # Limit the fallback to that endpoint so other account types keep strict picker semantics.
    allow_policy_fallback = base_url == "https://api.individual.githubcopilot.com"
    raw = await _fetch_json(
        "GET",
        f"{base_url}/models",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {copilot_token}",
            **COPILOT_HEADERS,
            "X-GitHub-Api-Version": COPILOT_API_VERSION,
        },
        client=client,
    )
    return parse_available_copilot_model_ids(raw, allow_policy_fallback)


async def start_device_flow(domain: str, client: httpx.AsyncClient | None = None) -> dict[str, object]:
    urls = get_urls(domain)
    data = await _fetch_json(
        "POST",
        urls["device_code_url"],
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        data={"client_id": CLIENT_ID, "scope": "read:user"},
        client=client,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Invalid device code response")

    verification_uri = data.get("verification_uri")
    if not isinstance(verification_uri, str):
        raise RuntimeError("Invalid device code response fields")
    # The verification URI is opened in the user's browser; force it to be a
    # trusted URL so a malicious response cannot make `open` launch something
    # else, and normalize it so control characters cannot reach the terminal.
    try:
        normalized_uri = normalize_http_url(verification_uri)
    except ValueError as error:
        raise RuntimeError("Untrusted verification_uri in device code response") from error
    if urlparse(normalized_uri).scheme not in ("http", "https"):
        raise RuntimeError("Untrusted verification_uri in device code response")

    if not isinstance(data.get("device_code"), str) or not isinstance(data.get("user_code"), str):
        raise RuntimeError("Invalid device code response fields")
    return {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": normalized_uri,
        "interval": data.get("interval"),
        "expires_in": data.get("expires_in"),
    }


async def poll_for_github_access_token(
    domain: str,
    device: dict[str, object],
    signal: AbortSignal,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> str:
    urls = get_urls(domain)

    async def poll() -> DeviceCodePollResult[str]:
        raw = await _fetch_json(
            "POST",
            urls["access_token_url"],
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": str(device["device_code"]),
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            client=client,
        )
        if isinstance(raw, dict) and isinstance(raw.get("access_token"), str):
            return DeviceCodePollResult(status="complete", value=raw["access_token"])
        if isinstance(raw, dict) and isinstance(raw.get("error"), str):
            error = raw["error"]
            description = raw.get("error_description")
            if error == "authorization_pending":
                return DeviceCodePollResult(status="pending")
            if error == "slow_down":
                interval = raw.get("interval")
                return DeviceCodePollResult(
                    status="slow_down", interval_seconds=interval if isinstance(interval, (int, float)) else None
                )
            suffix = f": {description}" if description else ""
            return DeviceCodePollResult(status="failed", message=f"Device flow failed: {error}{suffix}")
        return DeviceCodePollResult(status="failed", message="Invalid device token response")

    return await poll_oauth_device_code_flow(
        poll,
        signal,
        interval_seconds=device.get("interval"),
        expires_in_seconds=device.get("expires_in"),
        wait_before_first_poll=True,
        clock=clock,
    )


async def refresh_github_copilot_access_token(
    refresh_token: str, enterprise_domain: str | None = None, client: httpx.AsyncClient | None = None
) -> Credential:
    domain = enterprise_domain or "github.com"
    urls = get_urls(domain)
    raw = await _fetch_json(
        "GET",
        urls["copilot_token_url"],
        headers={"Accept": "application/json", "Authorization": f"Bearer {refresh_token}", **COPILOT_HEADERS},
        client=client,
    )
    if not isinstance(raw, dict):
        raise RuntimeError("Invalid Copilot token response")

    token = raw.get("token")
    expires_at = raw.get("expires_at")
    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        raise RuntimeError("Invalid Copilot token response fields")

    # `expires_at` is a Unix seconds timestamp; refresh five minutes early.
    return Credential(
        type="oauth",
        refresh=refresh_token,
        access=token,
        expires=expires_at * 1000 - 5 * 60 * 1000,
        data={"enterprise_url": enterprise_domain} if enterprise_domain else {},
    )


async def enable_github_copilot_model(
    token: str, model_id: str, enterprise_domain: str | None, client: httpx.AsyncClient | None = None
) -> bool:
    """Enable one model for the user's GitHub Copilot account (required for e.g. Claude/Grok)."""
    base_url = get_github_copilot_base_url(token, enterprise_domain)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.post(
            f"{base_url}/models/{model_id}/policy",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **COPILOT_HEADERS,
                "openai-intent": "chat-policy",
                "x-interaction-type": "chat-policy",
            },
            json={"state": "enabled"},
        )
        return response.status_code < 400
    except httpx.HTTPError:
        return False
    finally:
        if owns_client:
            await http_client.aclose()


async def enable_all_github_copilot_models(
    token: str, enterprise_domain: str | None, client: httpx.AsyncClient | None = None
) -> None:
    """Enable every catalog model that may require policy acceptance.

    Called after a successful login so all models are usable. Requests run in
    batches of ``COPILOT_POLICY_CONCURRENCY`` rather than all at once: the
    catalog holds dozens of models, and firing every policy request in parallel
    made GitHub reject the burst.
    """
    model_ids = [model.id for model in load_models("github-copilot")]
    for index in range(0, len(model_ids), COPILOT_POLICY_CONCURRENCY):
        batch = model_ids[index : index + COPILOT_POLICY_CONCURRENCY]
        await asyncio.gather(
            *(enable_github_copilot_model(token, model_id, enterprise_domain, client) for model_id in batch)
        )


def copilot_enterprise_domain(credential: Credential) -> str | None:
    enterprise_url = credential.data.get("enterprise_url")
    if not isinstance(enterprise_url, str) or not enterprise_url:
        return None
    return normalize_domain(enterprise_url)


async def login_github_copilot(
    interaction: AuthInteraction,
    *,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    domain_input = await interaction.prompt(
        AuthPrompt(
            type="text",
            message="GitHub Enterprise URL/domain (blank for github.com)",
            placeholder="company.ghe.com",
        )
    )
    if interaction.signal.aborted:
        raise RuntimeError("Login cancelled")

    trimmed = domain_input.strip()
    enterprise_domain = normalize_domain(domain_input)
    if trimmed and not enterprise_domain:
        raise RuntimeError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await start_device_flow(domain, client)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=str(device["user_code"]),
            verification_uri=str(device["verification_uri"]),
            interval_seconds=device.get("interval"),
            expires_in_seconds=device.get("expires_in"),
        )
    )

    github_access_token = await poll_for_github_access_token(domain, device, interaction.signal, client, clock)
    credential = await refresh_github_copilot_access_token(github_access_token, enterprise_domain, client)
    interaction.notify(AuthEvent(type="progress", message="Enabling models..."))
    await enable_all_github_copilot_models(credential.access, enterprise_domain, client)
    model_ids = await fetch_available_github_copilot_model_ids(credential.access, enterprise_domain, client)
    credential.data = {**credential.data, "availableModelIds": model_ids}
    return credential


async def refresh(
    credential: Credential, signal: AbortSignal, *, client: httpx.AsyncClient | None = None
) -> Credential:
    enterprise_domain = copilot_enterprise_domain(credential)
    refreshed = await refresh_github_copilot_access_token(credential.refresh or "", enterprise_domain, client)
    model_ids = await fetch_available_github_copilot_model_ids(refreshed.access, enterprise_domain, client)
    refreshed.data = {**refreshed.data, "availableModelIds": model_ids}
    return refreshed


async def to_auth(credential: Credential) -> ResolvedAuth:
    """Derive the credential-specific proxy endpoint for each request."""
    return ResolvedAuth(
        api_key=credential.access,
        base_url=get_github_copilot_base_url(credential.access, copilot_enterprise_domain(credential)),
    )


def build_github_copilot_oauth() -> OAuthAuth:
    return OAuthAuth(
        name="GitHub Copilot",
        is_subscription=True,
        login=login_github_copilot,
        refresh=refresh,
        to_auth=to_auth,
    )
