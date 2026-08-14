"""HTTP/HTTPS proxy resolution from environment variables.

Python port of `packages/ai/src/utils/node-http-proxy.ts`.

TypeScript installs a proxy-aware undici dispatcher; here the resolved proxy
URL is handed to `httpx` (see :mod:`pi_ai.utils.http`). The environment
handling is the same in both: `http_proxy` / `https_proxy` / `all_proxy` select
a proxy per target scheme, `no_proxy` exempts hosts, and both lower- and
upper-case spellings are accepted, with a provider-scoped `env` mapping taking
priority over the process environment.
"""

from __future__ import annotations

from urllib.parse import urlsplit

DEFAULT_PROXY_PORTS: dict[str, int] = {
    "ftp": 21,
    "gopher": 70,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}

UNSUPPORTED_PROXY_PROTOCOL_MESSAGE = (
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL."
)


def _get_proxy_env(key: str, env: dict[str, str] | None = None) -> str:
    from .provider_env import get_provider_env_value

    lowercase_key = key.lower()
    uppercase_key = key.upper()
    if env is not None:
        for name in (lowercase_key, uppercase_key):
            value = env.get(name)
            if value:
                return value
    for name in (lowercase_key, uppercase_key):
        value = get_provider_env_value(name)
        if value:
            return value
    return ""


def _should_proxy_hostname(hostname: str, port: int, env: dict[str, str] | None = None) -> bool:
    no_proxy = _get_proxy_env("no_proxy", env).lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False

    for entry in _split_no_proxy(no_proxy):
        if not entry:
            continue
        proxy_hostname = entry
        proxy_port = 0
        host, separator, port_text = entry.rpartition(":")
        if separator and port_text.isdigit():
            proxy_hostname = host
            proxy_port = int(port_text)
        if proxy_port and proxy_port != port:
            continue

        if not proxy_hostname.startswith((".", "*")):
            if hostname == proxy_hostname:
                return False
            continue

        if proxy_hostname.startswith("*"):
            proxy_hostname = proxy_hostname[1:]
        if hostname.endswith(proxy_hostname):
            return False
    return True


def _split_no_proxy(no_proxy: str) -> list[str]:
    entries = [no_proxy]
    for separator in (",", " ", "\t", "\n", "\r"):
        entries = [part for entry in entries for part in entry.split(separator)]
    return entries


def get_proxy_for_url(target_url: str, env: dict[str, str] | None = None) -> str:
    """The proxy URL configured for ``target_url``, or ``""`` when unproxied."""
    parsed = urlsplit(target_url)
    if not parsed.scheme or not parsed.netloc:
        return ""

    protocol = parsed.scheme
    hostname = parsed.hostname or ""
    try:
        port = parsed.port or 0
    except ValueError:
        port = 0
    port = port or DEFAULT_PROXY_PORTS.get(protocol, 0)
    if not _should_proxy_hostname(hostname, port, env):
        return ""

    proxy = _get_proxy_env(f"{protocol}_proxy", env) or _get_proxy_env("all_proxy", env)
    if proxy and "://" not in proxy:
        proxy = f"{protocol}://{proxy}"
    return proxy


def resolve_http_proxy_url_for_target(target_url: str, env: dict[str, str] | None = None) -> str | None:
    """The HTTP/HTTPS proxy to use for ``target_url``, or ``None``.

    Raises :class:`ValueError` for a malformed proxy URL or for a SOCKS/PAC
    proxy, which `httpx` cannot use without an extra transport.
    """
    proxy = get_proxy_for_url(target_url, env)
    if not proxy:
        return None

    parsed = urlsplit(proxy)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid proxy URL {proxy!r}: missing scheme or host")
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {parsed.scheme}:")
    return proxy
