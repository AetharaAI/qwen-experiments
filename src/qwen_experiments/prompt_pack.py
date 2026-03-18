from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PromptCase:
    index: int
    category: str
    slug: str
    text: str


DEFAULT_PROMPT_PACK: tuple[PromptCase, ...] = (
    PromptCase(
        index=1,
        category="brand",
        slug="thanks_for_calling_aether",
        text="Thanks for calling Aether.",
    ),
    PromptCase(
        index=2,
        category="brand",
        slug="thanks_for_calling_aether_pro",
        text="Thanks for calling Aether Pro.",
    ),
    PromptCase(
        index=3,
        category="site",
        slug="welcome_to_aether_voice",
        text="Welcome to Aether Voice. How can I help you today?",
    ),
    PromptCase(
        index=4,
        category="site",
        slug="enterprise_voice_platform",
        text="Aether Voice is enterprise speech infrastructure built for speed, quality, and control.",
    ),
    PromptCase(
        index=5,
        category="telephony",
        slug="service_categories",
        text="I can help with scheduling, support, billing, and general questions.",
    ),
    PromptCase(
        index=6,
        category="telephony",
        slug="brief_hold_message",
        text="Please hold for just a moment while I pull up your account.",
    ),
    PromptCase(
        index=7,
        category="ads",
        slug="sovereign_voice_agents",
        text="Launch sovereign voice agents that sound premium without giving up operational control.",
    ),
    PromptCase(
        index=8,
        category="ads",
        slug="self_hosted_ai_voice",
        text="Aether Voice helps modern teams deploy self-hosted AI voice experiences with confidence.",
    ),
)


def prompt_pack_manifest() -> list[dict[str, str]]:
    return [asdict(case) for case in DEFAULT_PROMPT_PACK]
