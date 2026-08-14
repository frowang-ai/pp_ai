"""OAuth login CLI for pi_ai.

Python port of `packages/ai/src/cli.ts`. Runs an OAuth provider's login flow on
the terminal and stores the resulting credential in `auth.json` in the current
directory.

The TypeScript reads stdin through Node's `readline`; this port uses `input()`
behind an injectable reader so the flow can be driven in tests without a TTY.

Like the TypeScript, the provider list is derived from `builtin_providers()`
filtered to those declaring an OAuth flow, rather than hardcoded. A hardcoded
list silently goes stale as providers are added, and it bypasses the
`lazy_oauth` wrapper that owns loading the flow -- which is how this CLI
previously ended up calling a raw `async` loader without awaiting it and
crashing with "\'coroutine\' object has no attribute \'login\'".
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auth.types import AuthEvent, AuthInteraction, AuthPrompt
from .providers.all import builtin_providers
from .registry import Provider
from .utils.abort import AbortSignal

AUTH_FILE = "auth.json"


def oauth_providers() -> list[Provider]:
    """Every built-in provider that declares an OAuth flow, in id order."""
    return sorted(
        (provider for provider in builtin_providers() if provider.auth.oauth is not None),
        key=lambda provider: provider.id,
    )


def provider_ids() -> list[str]:
    return [provider.id for provider in oauth_providers()]


def find_provider(provider_id: str) -> Provider | None:
    return next((provider for provider in oauth_providers() if provider.id == provider_id), None)


def load_auth(auth_file: str | Path = AUTH_FILE) -> dict[str, Any]:
    """Read stored credentials. A missing or corrupt file yields no credentials."""
    path = Path(auth_file)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_auth(auth: dict[str, Any], auth_file: str | Path = AUTH_FILE) -> None:
    Path(auth_file).write_text(json.dumps(auth, indent=2), encoding="utf-8")


@dataclass
class ConsoleInteraction(AuthInteraction):
    """Drives a login flow through the terminal."""

    signal: AbortSignal
    read_line: Callable[[str], str] = input
    write: Callable[[str], None] = print

    async def prompt(self, prompt: AuthPrompt) -> str:
        if getattr(prompt, "type", None) == "select":
            # Options are plain dicts: {"id": ..., "label": ..., "description": ...}
            options = list(prompt.options or ())
            self.write(f"\n{prompt.message}")
            for index, option in enumerate(options):
                self.write(f"  {index + 1}. {option.get('label', option.get('id', ''))}")
            answer = self.read_line(f"Enter number (1-{len(options)}): ")
            try:
                choice = int(answer) - 1
                if choice < 0:
                    raise IndexError(choice)
                selected = options[choice]
            except (ValueError, IndexError) as error:
                raise ValueError("Invalid selection") from error
            return selected["id"]

        suffix = f" ({prompt.placeholder})" if getattr(prompt, "placeholder", None) else ""
        return self.read_line(f"{prompt.message}{suffix}: ")

    def notify(self, event: AuthEvent) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "auth_url":
            self.write(f"\nOpen this URL in your browser:\n{event.url}")
            instructions = getattr(event, "instructions", None)
            if instructions:
                self.write(instructions)
        elif event_type == "device_code":
            self.write(f"\nOpen this URL in your browser:\n{event.verification_uri}")
            self.write(f"Enter code: {event.user_code}")
        elif event_type in ("info", "progress"):
            self.write(event.message)


async def login(
    provider_id: str,
    auth_file: str | Path = AUTH_FILE,
    interaction: AuthInteraction | None = None,
    write: Callable[[str], None] = print,
) -> None:
    """Run ``provider_id``'s OAuth login flow and persist the credential."""
    provider = find_provider(provider_id)
    if provider is None:
        raise ValueError(f"Unknown provider: {provider_id}")

    active = interaction or ConsoleInteraction(signal=AbortSignal(), write=write)
    # `provider.auth.oauth` is the `lazy_oauth` wrapper, which awaits the
    # underlying loader on first use.
    credential = await provider.auth.oauth.login(active)

    auth = load_auth(auth_file)
    auth[provider_id] = credential.data if hasattr(credential, "data") else credential
    save_auth(auth, auth_file)
    write(f"\nCredentials saved to {auth_file}")


def _usage() -> str:
    listing = "\n".join(f"  {provider.id:<20} {provider.name}" for provider in oauth_providers())
    return (
        "Usage: pp-ai <command> [provider]\n\n"
        "Commands:\n"
        "  login [provider]  Login to an OAuth provider\n"
        "  list              List available providers\n\n"
        f"Providers:\n{listing}"
    )


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else None

    if command is None or command in ("help", "--help", "-h"):
        write(_usage())
        return 0

    providers = oauth_providers()

    if command == "list":
        for provider in providers:
            write(f"{provider.id:<20} {provider.name}")
        return 0

    if command == "login":
        provider_id = args[1] if len(args) > 1 else None
        if not provider_id:
            for index, provider in enumerate(providers):
                write(f"  {index + 1}. {provider.name}")
            try:
                choice = int(input(f"Enter number (1-{len(providers)}): ")) - 1
                if choice < 0:
                    raise IndexError(choice)
                provider_id = providers[choice].id
            except (ValueError, IndexError):
                provider_id = None
        if not provider_id or find_provider(provider_id) is None:
            write(f"Error: Unknown provider: {provider_id or ''}")
            return 1
        try:
            asyncio.run(login(provider_id, write=write))
        except Exception as error:
            write(f"Error: {error}")
            return 1
        return 0

    write(f"Error: Unknown command: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
