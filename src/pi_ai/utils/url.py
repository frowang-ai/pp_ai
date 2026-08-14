"""WHATWG-compatible URL normalization.

Python port of the normalization JavaScript gets for free from
`new URL(raw).href`. `urllib.parse.urlparse(raw).geturl()` returns the input
verbatim, so a hostile OAuth `verification_uri` carrying terminal escape
sequences or spaces would reach the terminal (and the OS `open` launcher)
unescaped. `normalize_http_url` reproduces the WHATWG serialization the
TypeScript OAuth flows rely on: lower-cased scheme and host, default port
dropped, empty path defaulted to `/`, and C0 controls, spaces, non-ASCII, and
the URL-unsafe punctuation percent-encoded per component.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

DEFAULT_PORTS = {"http": "80", "https": "443", "ws": "80", "wss": "443", "ftp": "21"}

# WHATWG percent-encode sets, minus the C0/space/non-ASCII range handled below.
_PATH_UNSAFE = '"#<>?`{}'
_QUERY_UNSAFE = '"#<>'
_FRAGMENT_UNSAFE = '"<>`'


def _percent_encode(value: str, unsafe: str) -> str:
    encoded: list[str] = []
    for char in value:
        code = ord(char)
        if code <= 0x20 or code >= 0x7F or char in unsafe:
            encoded.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
        else:
            encoded.append(char)
    return "".join(encoded)


def normalize_http_url(raw: str) -> str:
    """Serialize ``raw`` the way `new URL(raw).href` does.

    Raises :class:`ValueError` when ``raw`` has no scheme or no host, matching
    the `new URL()` constructor throwing on a non-absolute URL.
    """
    parts = urlsplit(raw)
    if not parts.scheme or not parts.hostname:
        raise ValueError(f"Not an absolute URL: {raw!r}")

    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    netloc = f"[{host}]" if ":" in host else host
    port = parts.port
    if port is not None and str(port) != DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{port}"
    if parts.username is not None:
        userinfo = _percent_encode(parts.username, _PATH_UNSAFE)
        if parts.password is not None:
            userinfo = f"{userinfo}:{_percent_encode(parts.password, _PATH_UNSAFE)}"
        netloc = f"{userinfo}@{netloc}"

    path = _percent_encode(parts.path, _PATH_UNSAFE) or "/"
    query = _percent_encode(parts.query, _QUERY_UNSAFE)
    fragment = _percent_encode(parts.fragment, _FRAGMENT_UNSAFE)
    return urlunsplit((scheme, netloc, path, query, fragment))
