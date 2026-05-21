"""Shared helper utilities for inference modules."""

from collections.abc import Sequence
from typing import Literal

InferenceMode = Literal["chat", "completion"]


def normalize_system_messages(
    system_messages: Sequence[str | None] | None,
    batch_size: int,
) -> list[str | None]:
    """Normalize optional system messages to one entry per prompt."""
    if system_messages is None:
        return [None] * batch_size
    return list(system_messages)


def validate_inference_mode(inference_mode: str) -> InferenceMode:
    """Validate the selected inference mode."""
    if inference_mode not in ("chat", "completion"):
        raise ValueError("`inference_mode` must be either 'chat' or 'completion'.")
    return inference_mode
