from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PromptCase:
    slug: str
    text: str


DEFAULT_PROMPT_PACK: tuple[PromptCase, ...] = (
    PromptCase(
        slug="thanks_for_calling_aether",
        text="Thanks for calling Aether.",
    ),
    PromptCase(
        slug="thanks_for_calling_aether_pro",
        text="Thanks for calling Aether Pro.",
    ),
    PromptCase(
        slug="welcome_to_aether_voice",
        text="Welcome to Aether Voice. How can I help you today?",
    ),
    PromptCase(
        slug="service_categories",
        text="I can help with scheduling, support, billing, and general questions.",
    ),
    PromptCase(
        slug="brief_hold_message",
        text="Please hold for just a moment while I pull up your account.",
    ),
)


def prompt_pack_manifest() -> list[dict[str, str]]:
    return [asdict(case) for case in DEFAULT_PROMPT_PACK]
