"""Tests for shared multimodal inputs and backend message construction."""

import base64

import pytest

from qstn.inference import local_inference, remote_inference
from qstn.inference.multimodal import (
    ImageInput,
    build_user_content,
    normalize_prompt_content,
    validate_text_only_completion_prompts,
)


class CaptureModel:
    def __init__(self):
        self.batch_messages = None

    def chat(self, batch_messages, **kwargs):
        del kwargs
        self.batch_messages = batch_messages
        output = type("Output", (), {})()
        output.outputs = [type("Candidate", (), {"text": "answer", "logprobs": []})()]
        return [output]


def test_image_input_supports_urls_data_urls_and_local_files(tmp_path):
    url = ImageInput("https://example.com/image.png")
    data_url = ImageInput("data:image/png;base64,aGVsbG8=")
    local_path = tmp_path / "image.png"
    local_path.write_bytes(b"image-bytes")
    local = ImageInput(local_path)

    assert url.to_url() == "https://example.com/image.png"
    assert data_url.to_url() == "data:image/png;base64,aGVsbG8="
    assert local.to_url() == (
        "data:image/png;base64," + base64.b64encode(b"image-bytes").decode("ascii")
    )


@pytest.mark.parametrize(
    "source, match",
    [
        ("", "must not be empty"),
        ("data:text/plain;base64,aGVsbG8=", "must use an image MIME type"),
        ("data:image/png;base64,", "non-empty payload"),
        ("data:image/png,aGVsbG8=", "base64-encoded"),
        ("data:image/png;base64,not-base64", "valid base64"),
        ("missing.png", "does not exist"),
    ],
)
def test_image_input_rejects_invalid_sources(source, match):
    with pytest.raises(ValueError, match=match):
        ImageInput(source)


def test_image_input_accepts_backend_specific_image_types(tmp_path):
    path = tmp_path / "image.svg"
    path.write_text("<svg />")

    image = ImageInput(path)

    assert image.to_url() == (
        "data:image/svg+xml;base64," + base64.b64encode(b"<svg />").decode("ascii")
    )


def test_image_input_rejects_non_image_local_type(tmp_path):
    path = tmp_path / "image.txt"
    path.write_text("not an image")

    with pytest.raises(ValueError, match="determine an image MIME type"):
        ImageInput(path)


def test_build_user_content_preserves_label_order():
    image = ImageInput("https://example.com/image.png", label="Reference image")

    content = build_user_content(["Question", image])

    assert content == [
        {"type": "text", "text": "Question"},
        {"type": "text", "text": "Reference image"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
    ]
    assert build_user_content("Question") == "Question"


def test_build_user_content_preserves_question_image_interleaving():
    first = ImageInput("https://example.com/one.png", label="First image")
    second = ImageInput("https://example.com/two.png")

    content = build_user_content(["Question 1", first, "Question 2", second])

    assert content == [
        {"type": "text", "text": "Question 1"},
        {"type": "text", "text": "First image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/one.png"}},
        {"type": "text", "text": "Question 2"},
        {"type": "image_url", "image_url": {"url": "https://example.com/two.png"}},
    ]


def test_structured_content_is_optional_per_batch_entry():
    image = ImageInput("https://example.com/image.png")
    model = CaptureModel()

    local_inference.run_vllm_batch(
        model,
        system_messages=[None, None],
        prompts=["plain prompt", ["Question", image]],
        print_progress=False,
    )

    assert model.batch_messages[0] == [{"role": "user", "content": "plain prompt"}]
    assert model.batch_messages[1][0]["content"] == [
        {"type": "text", "text": "Question"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


def test_prompt_content_validation_rejects_empty_and_invalid_blocks():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_prompt_content([])

    with pytest.raises(TypeError, match="strings or ImageInput"):
        normalize_prompt_content(["Question", object()])


def test_completion_mode_rejects_structured_content():
    with pytest.raises(ValueError, match="Structured prompt content.*supported only"):
        local_inference.run_vllm_batch(
            CaptureModel(),
            system_messages=[None],
            prompts=[["Question"]],
            inference_mode="completion",
        )


def test_completion_validation_accepts_flat_and_conversation_text_groups():
    validate_text_only_completion_prompts("completion", ["first", "second"])
    validate_text_only_completion_prompts(
        "completion",
        ["first turn", "second turn"],
        ["another conversation"],
    )


def test_completion_validation_rejects_structured_content_in_any_group():
    with pytest.raises(ValueError, match="Structured prompt content.*supported only"):
        validate_text_only_completion_prompts(
            "completion",
            ["plain conversation"],
            [["Question"]],
        )


def test_conversation_turns_accept_unified_prompt_content():
    image = ImageInput("https://example.com/image.png")
    model = CaptureModel()

    local_inference.run_vllm_batch_conversation(
        model,
        system_messages=[None],
        prompts=[[["First question", image], "Second question"]],
        assistant_messages=[["First answer"]],
        print_progress=False,
    )

    assert model.batch_messages == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First question"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            },
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
    ]


def test_local_and_remote_adapters_build_the_same_multimodal_content(monkeypatch):
    image = ImageInput("https://example.com/image.png", label="Stimulus")
    local_model = CaptureModel()
    captured = {}

    def fake_remote_helper(**kwargs):
        captured.update(kwargs)
        return (["answer"], [None], [None])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_remote_helper)

    local_inference.run_vllm_batch(
        local_model,
        system_messages=["system"],
        prompts=[["question", image]],
        print_progress=False,
    )
    remote_inference.run_openai_batch(
        object(),
        system_messages=["system"],
        prompts=[["question", image]],
        client_model_name="model",
        print_progress=False,
    )

    assert local_model.batch_messages == captured["batch_messages"]


@pytest.mark.parametrize(
    "runner, kwargs",
    [
        (
            local_inference.run_vllm_batch,
            {"model": CaptureModel(), "system_messages": [None]},
        ),
        (
            remote_inference.run_openai_batch,
            {"model": object(), "system_messages": [None]},
        ),
    ],
)
def test_completion_mode_rejects_images(runner, kwargs):
    with pytest.raises(ValueError, match="supported only"):
        runner(
            **kwargs,
            prompts=[["prompt", ImageInput("https://example.com/image.png")]],
            inference_mode="completion",
        )
