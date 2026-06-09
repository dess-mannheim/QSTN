"""Shared image-input types and multimodal message construction."""

import base64
import binascii
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .utils import InferenceMode


@dataclass(frozen=True)
class ImageInput:
    """Image supplied to a multimodal model.

    Args:
        source: HTTP(S) URL, image data URL, or local image path.
        label: Optional text inserted immediately before the image.
    """

    source: str | Path
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, (str, Path)):
            raise TypeError("`source` must be a string or pathlib.Path.")
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("`source` must not be empty.")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("`label` must be a string or None.")

        source = str(self.source)
        if _is_http_url(source):
            return
        if source.startswith("data:"):
            _validate_image_data_url(source)
            return

        path = Path(self.source).expanduser()
        if not path.is_file():
            raise ValueError(f"Image path does not exist or is not a file: {path}")
        _get_image_mime_type(path)

    def to_url(self) -> str:
        """Return a backend-compatible URL or base64 data URL."""
        source = str(self.source)
        if _is_http_url(source) or source.startswith("data:"):
            return source

        path = Path(self.source).expanduser()
        mime_type = _get_image_mime_type(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


type ImageSource = ImageInput | str | Path
type ImageCollection = Sequence[ImageSource]
type PromptContentBlock = str | ImageInput
type PromptContent = str | Sequence[PromptContentBlock]
type BatchPromptContent = Sequence[PromptContent]
type ConversationPromptContent = Sequence[Sequence[PromptContent]]


def coerce_image_input(image: ImageSource) -> ImageInput:
    """Normalize an image value to ImageInput."""
    if isinstance(image, ImageInput):
        return image
    if isinstance(image, (str, Path)):
        return ImageInput(source=image)
    raise TypeError("Images must be ImageInput, string URLs, or pathlib.Path objects.")


def normalize_images(images: ImageCollection | None) -> tuple[ImageInput, ...]:
    """Normalize one request's image collection."""
    if images is None:
        return ()
    return tuple(coerce_image_input(image) for image in images)


def normalize_prompt_content(content: PromptContent) -> str | tuple[PromptContentBlock, ...]:
    """Validate and normalize one user prompt."""
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        raise TypeError("Prompt content must be a string or a sequence of content blocks.")

    blocks = tuple(content)
    if not blocks:
        raise ValueError("Structured prompt content must not be empty.")
    if not all(isinstance(block, (str, ImageInput)) for block in blocks):
        raise TypeError("Prompt content blocks must be strings or ImageInput objects.")
    return blocks


def combine_prompt_content(
    text: str,
    images: Sequence[ImageInput],
) -> PromptContent:
    """Return plain text unchanged or append ordered image blocks."""
    if not images:
        return text
    return (text, *images)


def prompt_content_text(content: PromptContent) -> str:
    """Return model-visible text while excluding image payloads."""
    normalized = normalize_prompt_content(content)
    if isinstance(normalized, str):
        return normalized
    return "".join(block if isinstance(block, str) else (block.label or "") for block in normalized)


def format_prompt_content(content: PromptContent) -> str:
    """Render prompt content as a readable text preview."""
    normalized = normalize_prompt_content(content)
    if isinstance(normalized, str):
        return normalized

    rendered: list[str] = []
    for block in normalized:
        if isinstance(block, str):
            rendered.append(block)
            continue

        source = str(block.source)
        if source.startswith("data:"):
            source = f"{source.partition(',')[0]},..."
        marker = f"[Image: {block.label or 'unlabelled'} | {source}]"
        if rendered and rendered[-1] and not rendered[-1].endswith("\n"):
            rendered.append("\n")
        rendered.append(marker)
        rendered.append("\n")

    if rendered and rendered[-1] == "\n":
        rendered.pop()
    return "".join(rendered)


def validate_text_only_completion_prompts(
    inference_mode: InferenceMode,
    *prompt_groups: Sequence[PromptContent],
) -> None:
    """Require every completion prompt to be plain text."""
    if inference_mode == "completion" and any(
        not isinstance(prompt, str) for prompts in prompt_groups for prompt in prompts
    ):
        raise ValueError("Structured prompt content is supported only when inference_mode='chat'.")


def build_user_content(content: PromptContent) -> str | list[dict[str, Any]]:
    """Build text-only or OpenAI-style multimodal user content."""
    normalized = normalize_prompt_content(content)
    if isinstance(normalized, str):
        return normalized

    parts: list[dict[str, Any]] = []
    for block in normalized:
        if isinstance(block, str):
            if block:
                parts.append({"type": "text", "text": block})
            continue
        if block.label is not None:
            parts.append({"type": "text", "text": block.label})
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": block.to_url()},
            }
        )
    if not parts:
        raise ValueError("Structured prompt content must produce at least one content part.")
    return parts


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_image_data_url(source: str) -> None:
    header, separator, payload = source.partition(",")
    if not separator or not payload:
        raise ValueError("Image data URLs must include a non-empty payload.")

    metadata = header.removeprefix("data:").split(";")
    media_type = metadata[0].lower()
    if not media_type.startswith("image/"):
        raise ValueError("Image data URLs must use an image MIME type (`image/*`).")
    if "base64" not in metadata[1:]:
        raise ValueError("Image data URLs must contain base64-encoded data.")
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data URLs must contain valid base64 data.") from exc


def _get_image_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Could not determine an image MIME type for '{path}'.")
    return mime_type
