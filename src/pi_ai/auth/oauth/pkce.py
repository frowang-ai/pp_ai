"""PKCE (Proof Key for Code Exchange) utilities.

Python port of `packages/ai/src/auth/oauth/pkce.ts`.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass


@dataclass
class Pkce:
    verifier: str
    challenge: str


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> Pkce:
    """Generate a PKCE code verifier and its S256 challenge."""
    verifier = _base64url_encode(os.urandom(32))
    challenge = _base64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return Pkce(verifier=verifier, challenge=challenge)
