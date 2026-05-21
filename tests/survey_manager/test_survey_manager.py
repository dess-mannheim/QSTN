"""Behavioral tests for survey manager orchestration helpers.

This module validates:
- call patterns for single/sequential/battery modes,
- handling of prefilled answers,
- intermediate-save path validation behavior.
"""

import pytest

from qstn import survey_manager
from qstn.parser.llm_answer_parser import raw_responses
from qstn.prompt_builder import LLMPrompt


def test_conduct_survey_single_item_runs_once_per_question(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    """Verify single-item survey triggers the API once per question and collects answers."""
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    call_counter = {"count": 0}

    def fake_batch_generation(**kwargs):
        call_counter["count"] += 1
        size = len(kwargs["prompts"])
        return (["mock answer"] * size, None, [None] * size)

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    results = survey_manager.conduct_survey_single_item(
        model=mock_openai_client,
        llm_prompts=prompt,
        client_model_name="mock-model",
        print_progress=False,
    )

    parsed = raw_responses(results)[prompt]
    assert call_counter["count"] == 2
    assert list(parsed["llm_response"]) == ["mock answer", "mock answer"]


def test_conduct_survey_sequential_uses_prefilled_response_without_api_call(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    """Ensure prefilled responses are used and no API call is made for them."""
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).prepare_prompt(
        prefilled_responses={1: "prefilled answer"}
    )
    call_counter = {"count": 0}

    def fake_batch_turn_by_turn_generation(**kwargs):
        call_counter["count"] += 1
        size = len(kwargs["system_messages"])
        return (["generated answer"] * size, None, [None] * size)

    monkeypatch.setattr(
        survey_manager,
        "batch_turn_by_turn_generation",
        fake_batch_turn_by_turn_generation,
    )

    results = survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=prompt,
        client_model_name="mock-model",
        print_progress=False,
    )

    parsed = raw_responses(results)[prompt]
    assert call_counter["count"] == 1
    assert (
        parsed.loc[parsed["questionnaire_item_id"] == 1, "llm_response"].item()
        == "prefilled answer"
    )
    assert (
        parsed.loc[parsed["questionnaire_item_id"] == 2, "llm_response"].item()
        == "generated answer"
    )


def test_conduct_survey_sequential_completion_renders_history_before_inference(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    captured_prompts: list[str] = []

    def fake_batch_generation(**kwargs):
        captured_prompts.extend(kwargs["prompts"])
        assert kwargs["inference_mode"] == "completion"
        assert kwargs["system_messages"] == [None]
        return ([f"answer {len(captured_prompts)}"], None, [None])

    def fail_batch_turn_by_turn_generation(**kwargs):
        raise AssertionError("completion mode should render before inference")

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)
    monkeypatch.setattr(
        survey_manager,
        "batch_turn_by_turn_generation",
        fail_batch_turn_by_turn_generation,
    )

    survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=prompt,
        client_model_name="mock-model",
        inference_mode="completion",
        print_progress=False,
    )

    assert len(captured_prompts) == 2
    assert "Assistant:\nanswer 1" in captured_prompts[1]
    assert captured_prompts[1].endswith("Assistant:")


def test_conduct_survey_battery_runs_once_and_keeps_separator(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    """Battery mode should call generation once and include the separator in the prompt."""
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    captured_prompts: list[str] = []

    def fake_batch_generation(**kwargs):
        captured_prompts[:] = kwargs["prompts"]
        size = len(kwargs["prompts"])
        return (["battery answer"] * size, None, [None] * size)

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    results = survey_manager.conduct_survey_battery(
        model=mock_openai_client,
        llm_prompts=[prompt],
        client_model_name="mock-model",
        print_progress=False,
        item_separator=" || ",
    )

    parsed = raw_responses(results)[prompt]
    assert len(results) == 1
    assert parsed["questionnaire_item_id"].tolist() == [-1]
    assert parsed["llm_response"].iloc[0] == "battery answer"
    assert " || " in captured_prompts[0]


@pytest.mark.parametrize(
    "n_save_step, path, match",
    [
        (-1, "intermediate.csv", "positive integer"),
        ("1", "intermediate.csv", "positive integer"),
        (1, None, "must be provided"),
        (1, "intermediate.json", "should be a .csv file"),
    ],
)
def test_intermediate_save_path_check_validates_inputs(n_save_step, path, match):
    """Invalid inputs to _intermediate_save_path_check raise helpful ValueErrors."""
    with pytest.raises(ValueError, match=match):
        survey_manager._intermediate_save_path_check(
            n_save_step=n_save_step, intermediate_save_path=path
        )


def test_intermediate_save_path_check_creates_directory(tmp_path):
    """When given a nested path, the helper ensures parent directories exist."""
    target = tmp_path / "new_folder" / "results.csv"

    survey_manager._intermediate_save_path_check(n_save_step=1, intermediate_save_path=str(target))

    assert target.parent.exists()


def test_survey_step_progress_label(monkeypatch):
    """Survey step progress should use the clearer tqdm description."""
    captured: dict[str, str] = {}

    def fake_tqdm(iterable, desc):
        captured["desc"] = desc
        return iterable

    monkeypatch.setattr(survey_manager, "tqdm", fake_tqdm)

    assert list(survey_manager._iter_survey_steps(2, print_progress=True)) == [0, 1]
    assert captured["desc"] == "Running survey steps"


def test_conduct_survey_single_item_renders_completion_prompt_before_inference(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).set_base_model_prompt_template(
        user_prefix="Instruction:",
        assistant_prefix="Response:",
    )
    captured = {}

    def fake_batch_generation(**kwargs):
        captured.update(kwargs)
        size = len(kwargs["prompts"])
        return (["ok"] * size, None, [None] * size)

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    survey_manager.conduct_survey_single_item(
        model=mock_openai_client,
        llm_prompts=prompt,
        client_model_name="mock-model",
        inference_mode="completion",
        print_progress=False,
    )

    assert captured["inference_mode"] == "completion"
    assert captured["system_messages"] == [None]
    assert captured["prompts"][0].startswith(
        "You will be given questions and possible answer options"
    )
    assert "Instruction:" in captured["prompts"][0]
    assert captured["prompts"][0].endswith("Response:")


def test_conduct_survey_single_item_chat_mode_does_not_forward_completion_template(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).set_base_model_prompt_template(
        user_prefix="Instruction:"
    )
    captured = {}

    def fake_batch_generation(**kwargs):
        captured.update(kwargs)
        size = len(kwargs["prompts"])
        return (["ok"] * size, None, [None] * size)

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    survey_manager.conduct_survey_single_item(
        model=mock_openai_client,
        llm_prompts=prompt,
        client_model_name="mock-model",
        print_progress=False,
    )

    assert captured["inference_mode"] == "chat"
