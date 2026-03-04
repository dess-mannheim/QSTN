"""Shared helper utilities for inference modules."""

from collections.abc import Sequence


def normalize_system_messages(
    system_messages: Sequence[str | None] | None,
    batch_size: int,
) -> list[str | None]:
    """Normalize optional system messages to one entry per prompt."""
    if system_messages is None:
        return [None] * batch_size
    return list(system_messages)
