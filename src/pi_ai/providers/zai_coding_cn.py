"""Z.AI Coding CN provider factory.

Python port of `packages/ai/src/providers/zai-coding-cn.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/zai-coding-cn.json`, which is the Python
equivalent of TypeScript's generated `providers/zai-coding-cn.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

ZAI_CODING_CN_MODELS: list[Model] = load_models("zai-coding-cn")


def zai_coding_cn_provider() -> Provider:
    """Build the built-in Z.AI Coding CN provider."""
    return create_provider(
        id="zai-coding-cn",
        name="Z.AI Coding CN",
        auth=ProviderAuth(api_key=env_api_key_auth("Z.AI Coding CN API key", ["ZAI_CODING_CN_API_KEY"])),
        api=openai_completions,
        models=ZAI_CODING_CN_MODELS,
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    )
