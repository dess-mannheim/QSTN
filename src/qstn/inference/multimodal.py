"""Shared media-input types and multimodal message construction."""

import base64
import binascii
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from .utils import InferenceMode


@dataclass(frozen=True)
class _MediaInput:
    """Shared validation and URL encoding for native media inputs."""

    _kind: ClassVar[str]

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
            _validate_media_data_url(source, self._kind)
            return

        path = Path(self.source).expanduser()
        if not path.is_file():
            raise ValueError(
                f"{self._kind.capitalize()} path does not exist or is not a file: {path}"
            )
        _get_media_mime_type(path, self._kind)

    def to_url(self) -> str:
        """Return a backend-compatible URL or base64 data URL."""
        source = str(self.source)
        if _is_http_url(source) or source.startswith("data:"):
            return source

        path = Path(self.source).expanduser()
        mime_type = _get_media_mime_type(path, self._kind)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


@dataclass(frozen=True)
class ImageInput(_MediaInput):
    """Image supplied to a multimodal model.

    The backend/model determines supported image formats and processing.

    Args:
        source: HTTP(S) URL, base64 image data URL, or local image path.
        label: Optional text inserted immediately before the image.

    Raises:
        TypeError: If source or label has an invalid type.
        ValueError: If the source is empty, the file is missing, or its MIME type
            or data URL is invalid.
    """

    _kind: ClassVar[str] = "image"


@dataclass(frozen=True)
class AudioInput(_MediaInput):
    """Audio supplied to a multimodal model.

    Requires a model and endpoint accepting vLLM-style ``audio_url`` blocks.
    Supported formats and audio processing depend on the backend/model.

    Args:
        source: HTTP(S) URL, base64 audio data URL, or local audio path.
        label: Optional text inserted immediately before the audio.

    Raises:
        TypeError: If source or label has an invalid type.
        ValueError: If the source is empty, the file is missing, or its MIME type
            or data URL is invalid.
    """

    _kind: ClassVar[str] = "audio"


@dataclass(frozen=True)
class VideoInput(_MediaInput):
    """Video supplied to a multimodal model.

    Requires a model and endpoint accepting vLLM-style ``video_url`` blocks.
    Supported formats, sampling, and soundtrack consumption depend on the backend/model.

    Args:
        source: HTTP(S) URL, base64 video data URL, or local video path.
        label: Optional text inserted immediately before the video.

    Raises:
        TypeError: If source or label has an invalid type.
        ValueError: If the source is empty, the file is missing, or its MIME type
            or data URL is invalid.
    """

    _kind: ClassVar[str] = "video"


type ImageSource = ImageInput | str | Path
type ImageCollection = Sequence[ImageSource]
type AudioSource = AudioInput | str | Path
type VideoSource = VideoInput | str | Path
type MediaInput = ImageInput | AudioInput | VideoInput
type PromptContentBlock = str | MediaInput
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


def normalize_audios(audios: Sequence[AudioSource] | None) -> tuple[AudioInput, ...]:
    """Normalize one request's audio collection."""
    return _normalize_typed_media(audios, AudioInput)


def normalize_videos(videos: Sequence[VideoSource] | None) -> tuple[VideoInput, ...]:
    """Normalize one request's video collection."""
    return _normalize_typed_media(videos, VideoInput)


def _normalize_typed_media[T: _MediaInput](
    values: Sequence[T | str | Path] | None, media_type: type[T]
) -> tuple[T, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("Media collections must be sequences of media sources.")
    result: list[T] = []
    for value in values:
        if isinstance(value, media_type):
            result.append(value)
        elif isinstance(value, (str, Path)):
            result.append(media_type(value))
        else:
            raise TypeError(
                f"Media must be {media_type.__name__}, string URLs, or pathlib.Path objects."
            )
    return tuple(result)


def normalize_media(media: Sequence[MediaInput]) -> tuple[MediaInput, ...]:
    """Validate an ordered collection of explicitly typed media objects."""
    if not isinstance(media, Sequence) or isinstance(media, (str, bytes)):
        raise TypeError(
            "Media must be a sequence of ImageInput, AudioInput, or VideoInput objects."
        )
    blocks = tuple(media)
    if not all(isinstance(block, (ImageInput, AudioInput, VideoInput)) for block in blocks):
        raise TypeError("Media must contain only ImageInput, AudioInput, or VideoInput objects.")
    return blocks


def normalize_prompt_content(content: PromptContent) -> str | tuple[PromptContentBlock, ...]:
    """Validate and normalize one user prompt."""
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        raise TypeError("Prompt content must be a string or a sequence of content blocks.")

    blocks = tuple(content)
    if not blocks:
        raise ValueError("Structured prompt content must not be empty.")
    if not all(isinstance(block, (str, ImageInput, AudioInput, VideoInput)) for block in blocks):
        raise TypeError(
            "Prompt content blocks must be strings or ImageInput, "
            "AudioInput, or VideoInput objects."
        )
    return blocks


def combine_prompt_content(
    text: str,
    media: Sequence[MediaInput],
) -> PromptContent:
    """Return plain text unchanged or arrange media around the text.

    Gemma 4 expects video media before the text and audio media after it. Keep
    the relative order of videos, other media, and audios while moving only the
    video and audio groups around the prompt.
    """
    if not media:
        return text
    videos = tuple(block for block in media if isinstance(block, VideoInput))
    other_media = tuple(block for block in media if not isinstance(block, (VideoInput, AudioInput)))
    audios = tuple(block for block in media if isinstance(block, AudioInput))
    return (*videos, text, *other_media, *audios)


def prompt_content_text(content: PromptContent) -> str:
    """Return model-visible text while excluding media payloads."""
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
        marker = f"[{block._kind.capitalize()}: {block.label or 'unlabelled'} | {source}]"
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
    """Build text or URL media blocks for vLLM and compatible remote endpoints."""
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
        part_type = f"{block._kind}_url"
        parts.append({"type": part_type, part_type: {"url": block.to_url()}})
    if not parts:
        raise ValueError("Structured prompt content must produce at least one content part.")
    return parts


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_media_data_url(source: str, kind: str) -> None:
    header, separator, payload = source.partition(",")
    if not separator or not payload:
        raise ValueError(f"{kind.capitalize()} data URLs must include a non-empty payload.")

    metadata = header.removeprefix("data:").split(";")
    media_type = metadata[0].lower()
    if not media_type.startswith(f"{kind}/"):
        article = "an" if kind[0] in "aeiou" else "a"
        raise ValueError(
            f"{kind.capitalize()} data URLs must use {article} {kind} MIME type (`{kind}/*`)."
        )
    if "base64" not in metadata[1:]:
        raise ValueError(f"{kind.capitalize()} data URLs must contain base64-encoded data.")
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{kind.capitalize()} data URLs must contain valid base64 data.") from exc


def _get_media_mime_type(path: Path, kind: str) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith(f"{kind}/"):
        article = "an" if kind[0] in "aeiou" else "a"
        raise ValueError(f"Could not determine {article} {kind} MIME type for '{path}'.")
    return mime_type
