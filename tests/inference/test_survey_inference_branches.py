"""Branch-focused tests for survey inference model routing and error handling."""

import pytest

from qstn.inference import survey_inference


def test_batch_generation_raises_importerror_for_missing_vllm(monkeypatch):
    """Model type `LLM` should raise ImportError when vLLM support is unavailable."""
    FakeLLM = type("LLM", (), {})
    monkeypatch.setattr(survey_inference, "HAS_VLLM", False)

    with pytest.raises(ImportError, match="vLLM model"):
        survey_inference.batch_generation(model=FakeLLM())


def test_batch_generation_raises_importerror_for_missing_openai(monkeypatch):
    """Model type `AsyncOpenAI` should raise ImportError when OpenAI support is unavailable."""
    FakeAsyncOpenAI = type("AsyncOpenAI", (), {})
    monkeypatch.setattr(survey_inference, "HAS_OPENAI", False)

    with pytest.raises(ImportError, match="use OpenAI"):
        survey_inference.batch_generation(model=FakeAsyncOpenAI())


def test_batch_generation_vllm_branch_supports_print_with_none_methods(monkeypatch):
    """vLLM path should delegate and print conversation even with `response_generation_method=None`."""
    FakeLLM = type("LLM", (), {})
    monkeypatch.setattr(survey_inference, "HAS_VLLM", True)
    monkeypatch.setattr(survey_inference, "LLM", FakeLLM)

    captured = {}

    def fake_run_vllm_batch(model, **kwargs):
        captured["kwargs"] = kwargs
        return (["ok"], [None], [None])

    written = []
    monkeypatch.setattr(survey_inference, "run_vllm_batch", fake_run_vllm_batch)
    monkeypatch.setattr(survey_inference.tqdm, "write", lambda msg: written.append(msg))

    out = survey_inference.batch_generation(
        model=FakeLLM(),
        system_messages=["sys"],
        prompts=["user"],
        response_generation_method=None,
        print_conversation=True,
        print_progress=False,
    )

    assert out == (["ok"], [None], [None])
    assert captured["kwargs"]["print_progress"] is False
    assert "-- Generated Message --" in written[0]


def test_batch_turn_by_turn_openai_branch_delegates(monkeypatch):
    """OpenAI path should forward conversation args to async batch helper."""
    FakeAsyncOpenAI = type("AsyncOpenAI", (), {})
    monkeypatch.setattr(survey_inference, "HAS_OPENAI", True)
    monkeypatch.setattr(survey_inference, "AsyncOpenAI", FakeAsyncOpenAI)

    captured = {}

    def fake_run_openai_batch_conversation(model, **kwargs):
        captured["kwargs"] = kwargs
        return (["a"], [None], ["r"])

    monkeypatch.setattr(
        survey_inference,
        "run_openai_batch_conversation",
        fake_run_openai_batch_conversation,
    )

    out = survey_inference.batch_turn_by_turn_generation(
        model=FakeAsyncOpenAI(),
        system_messages=["s"],
        prompts=[["u1", "u2"]],
        assistant_messages=[["a1"]],
        client_model_name="gpt-x",
        api_concurrency=4,
        print_progress=False,
    )

    assert out == (["a"], [None], ["r"])
    assert captured["kwargs"]["client_model_name"] == "gpt-x"
    assert captured["kwargs"]["api_concurrency"] == 4
    assert captured["kwargs"]["assistant_messages"] == [["a1"]]
