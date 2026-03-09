"""Additional survey manager tests for path validation, persistence, and creator helpers."""

import runpy

import pandas as pd
import pytest

from qstn import survey_manager
from qstn.inference.response_generation import (
    JSONResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
)
from qstn.parser.llm_answer_parser import raw_responses
from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.utilities.survey_objects import QuestionLLMResponseTuple


def test_intermediate_saves_writes_csv_and_skips_when_disabled(tmp_path):
    """`_intermediate_saves` should persist expected rows
    and skip writes when disabled."""
    questionnaire_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    prompt = LLMPrompt(questionnaire_source=questionnaire_df)
    pairs = [{1: QuestionLLMResponseTuple("Q1", "A1", None, None)}]

    save_path = tmp_path / "intermediate.csv"
    survey_manager._intermediate_saves(
        questionnaires=[prompt],
        n_save_step=1,
        intermediate_save_file=str(save_path),
        question_llm_response_pairs=pairs,
        i=0,
    )
    assert save_path.exists()
    saved = pd.read_csv(save_path)
    assert list(saved["questionnaire_name"]) == [prompt.questionnaire_name]
    assert list(saved["questionnaire_item_id"]) == [1]
    assert list(saved["question"]) == ["Q1"]
    assert list(saved["llm_response"]) == ["A1"]

    disabled_path = tmp_path / "disabled.csv"
    survey_manager._intermediate_saves(
        questionnaires=[prompt],
        n_save_step=None,
        intermediate_save_file=str(disabled_path),
        question_llm_response_pairs=pairs,
        i=0,
    )
    assert not disabled_path.exists()


def test_intermediate_save_path_check_handles_mkdir_failure(monkeypatch, tmp_path):
    """A failure creating directories should produce a ValueError."""
    target = tmp_path / "new_dir" / "results.csv"

    monkeypatch.setattr("qstn.survey_manager.Path.exists", lambda self: False)

    def fail_mkdir(self, parents=True, exist_ok=True):
        raise OSError("cannot create")

    monkeypatch.setattr("qstn.survey_manager.Path.mkdir", fail_mkdir)

    with pytest.raises(ValueError, match="Invalid intermediate save path"):
        survey_manager._intermediate_save_path_check(1, str(target))


def test_intermediate_save_path_check_handles_non_writable(monkeypatch, tmp_path):
    """Non-writable path leads to a ValueError in the path checker."""
    target = tmp_path / "results.csv"

    monkeypatch.setattr("qstn.survey_manager.os.access", lambda *_: False)

    with pytest.raises(ValueError, match="not writable"):
        survey_manager._intermediate_save_path_check(1, str(target))


def test_conduct_survey_battery_uses_json_rgm_and_keeps_logprobs(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    """Battery survey should respect JSON response method and expose logprobs."""
    rgm = JSONSingleResponseGenerationMethod()
    options = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        response_generation_method=rgm,
    )
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).prepare_prompt(
        answer_options=options
    )

    captured = {}

    def fake_batch_generation(**kwargs):
        captured["rgm"] = kwargs["response_generation_method"]
        return (["answer"], [{"token": -0.3}], ["reasoning"])

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    results = survey_manager.conduct_survey_battery(
        model=mock_openai_client,
        llm_prompts=prompt,
        client_model_name="mock-model",
        print_progress=False,
    )

    assert len(results) == 1
    assert captured["rgm"]
    assert list(results[0].results.values())[0].logprobs == {"token": -0.3}


def test_prepare_battery_batch_merges_mixed_json_methods_per_question():
    questionnaire_df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    option_5 = generate_likert_options(
        n=5,
        answer_texts=["SEHR GUT", "GUT", "TEILS/TEILS", "SCHLECHT", "SEHR SCHLECHT"],
        response_generation_method=JSONVerbalizedDistribution(),
    )
    option_2 = generate_likert_options(
        n=2,
        answer_texts=["BIN DERS.MEINUNG", "BIN ANDERER MEINUNG"],
        response_generation_method=JSONVerbalizedDistribution(),
    )
    prompt = LLMPrompt(questionnaire_source=questionnaire_df).prepare_prompt(
        answer_options={1: option_5, 2: option_2}
    )

    _, _, response_generation_methods = survey_manager._prepare_battery_batch(
        current_batch={0: prompt},
        i=0,
        item_separator="\n",
    )
    merged_method = response_generation_methods[0]
    assert isinstance(merged_method, JSONResponseGenerationMethod)

    merged_keys = list(merged_method.json_fields.keys())
    assert "Q1 | q1_o1" in merged_keys
    assert "Q1... | q1_o5" in merged_keys
    assert "Q2 | q2_o1" in merged_keys
    assert "Q2... | q2_o2" in merged_keys
    assert "Q2... | q2_o3" not in merged_keys


def test_conduct_survey_sequential_handles_partial_prefill_and_empty_logprobs(
    mock_openai_client, monkeypatch
):
    """Sequential survey should skip generation for prefilled prompts
    and handle missing logprobs."""
    one_item_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Only question?"}])
    prompt_prefilled = LLMPrompt(questionnaire_source=one_item_df).prepare_prompt(
        prefilled_responses={1: "prefilled"}
    )
    prompt_generated = LLMPrompt(questionnaire_source=one_item_df)

    calls = {"count": 0}

    def fake_batch_turn_by_turn_generation(**kwargs):
        calls["count"] += 1
        return (["generated"], [], [None])

    monkeypatch.setattr(
        survey_manager, "batch_turn_by_turn_generation", fake_batch_turn_by_turn_generation
    )

    results = survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=[prompt_prefilled, prompt_generated],
        client_model_name="mock-model",
        print_progress=False,
    )

    assert calls["count"] == 1
    assert len(results) == 2


def test_conduct_survey_sequential_mixed_lengths_keep_persona_history(
    mock_openai_client, monkeypatch
):
    """Sequential mode should keep prompt/assistant history aligned as batch size shrinks."""
    two_items_df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    one_item_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    prompt_a = LLMPrompt(questionnaire_source=two_items_df, questionnaire_name="A")
    prompt_b = LLMPrompt(questionnaire_source=one_item_df, questionnaire_name="B")
    prompt_c = LLMPrompt(questionnaire_source=two_items_df, questionnaire_name="C")

    recorded_calls = []

    def fake_batch_turn_by_turn_generation(**kwargs):
        recorded_calls.append(
            {
                "system_messages": list(kwargs["system_messages"]),
                "prompts": [list(conv) for conv in kwargs["prompts"]],
                "assistant_messages": [list(conv) for conv in kwargs["assistant_messages"]],
            }
        )
        call_index = len(recorded_calls) - 1
        batch_size = len(kwargs["system_messages"])
        return (
            [f"STEP{call_index}_IDX{i}" for i in range(batch_size)],
            [None] * batch_size,
            [None] * batch_size,
        )

    monkeypatch.setattr(
        survey_manager, "batch_turn_by_turn_generation", fake_batch_turn_by_turn_generation
    )

    results = survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=[prompt_a, prompt_b, prompt_c],
        client_model_name="mock-model",
        print_progress=False,
    )

    assert len(recorded_calls) == 2
    assert (
        len(recorded_calls[0]["system_messages"])
        == len(recorded_calls[0]["prompts"])
        == len(recorded_calls[0]["assistant_messages"])
        == 3
    )
    assert (
        len(recorded_calls[1]["system_messages"])
        == len(recorded_calls[1]["prompts"])
        == len(recorded_calls[1]["assistant_messages"])
        == 2
    )
    assert recorded_calls[1]["assistant_messages"] == [["STEP0_IDX0"], ["STEP0_IDX2"]]

    parsed = raw_responses(results)
    parsed_a = parsed[prompt_a]
    parsed_b = parsed[prompt_b]
    parsed_c = parsed[prompt_c]

    assert (
        parsed_a.loc[parsed_a["questionnaire_item_id"] == 1, "llm_response"].item() == "STEP0_IDX0"
    )
    assert (
        parsed_a.loc[parsed_a["questionnaire_item_id"] == 2, "llm_response"].item() == "STEP1_IDX0"
    )
    assert (
        parsed_b.loc[parsed_b["questionnaire_item_id"] == 1, "llm_response"].item() == "STEP0_IDX1"
    )
    assert (
        parsed_c.loc[parsed_c["questionnaire_item_id"] == 1, "llm_response"].item() == "STEP0_IDX2"
    )
    assert (
        parsed_c.loc[parsed_c["questionnaire_item_id"] == 2, "llm_response"].item() == "STEP1_IDX1"
    )


def test_conduct_survey_sequential_mixed_lengths_with_prefill_keeps_mapping(
    mock_openai_client, monkeypatch
):
    """Prefilled answers should bypass generation without breaking mixed-length alignment."""
    two_items_df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    one_item_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    prompt_a = LLMPrompt(questionnaire_source=two_items_df, questionnaire_name="A")
    prompt_b = LLMPrompt(questionnaire_source=one_item_df, questionnaire_name="B")
    prompt_c = LLMPrompt(
        questionnaire_source=two_items_df,
        questionnaire_name="C",
    ).prepare_prompt(prefilled_responses={2: "PREFILL_C2"})

    recorded_calls = []

    def fake_batch_turn_by_turn_generation(**kwargs):
        recorded_calls.append(
            {
                "system_messages": list(kwargs["system_messages"]),
                "prompts": [list(conv) for conv in kwargs["prompts"]],
                "assistant_messages": [list(conv) for conv in kwargs["assistant_messages"]],
            }
        )
        call_index = len(recorded_calls) - 1
        batch_size = len(kwargs["system_messages"])
        return (
            [f"STEP{call_index}_IDX{i}" for i in range(batch_size)],
            [None] * batch_size,
            [None] * batch_size,
        )

    monkeypatch.setattr(
        survey_manager, "batch_turn_by_turn_generation", fake_batch_turn_by_turn_generation
    )

    results = survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=[prompt_a, prompt_b, prompt_c],
        client_model_name="mock-model",
        print_progress=False,
    )

    assert len(recorded_calls) == 2
    assert (
        len(recorded_calls[1]["system_messages"])
        == len(recorded_calls[1]["prompts"])
        == len(recorded_calls[1]["assistant_messages"])
        == 1
    )
    assert recorded_calls[1]["assistant_messages"] == [["STEP0_IDX0"]]

    parsed = raw_responses(results)
    parsed_a = parsed[prompt_a]
    parsed_b = parsed[prompt_b]
    parsed_c = parsed[prompt_c]

    assert (
        parsed_a.loc[parsed_a["questionnaire_item_id"] == 1, "llm_response"].item() == "STEP0_IDX0"
    )
    assert (
        parsed_a.loc[parsed_a["questionnaire_item_id"] == 2, "llm_response"].item() == "STEP1_IDX0"
    )
    assert (
        parsed_b.loc[parsed_b["questionnaire_item_id"] == 1, "llm_response"].item() == "STEP0_IDX1"
    )
    assert (
        parsed_c.loc[parsed_c["questionnaire_item_id"] == 1, "llm_response"].item() == "STEP0_IDX2"
    )
    assert (
        parsed_c.loc[parsed_c["questionnaire_item_id"] == 2, "llm_response"].item() == "PREFILL_C2"
    )


def test_survey_creator_from_dataframe_and_from_path(tmp_path):
    """SurveyCreator should build from both DataFrame and file paths."""
    survey_df = pd.DataFrame(
        [
            {
                "questionnaire_name": "S1",
                "system_prompt": "SYS",
                "questionnaire_instruction": "ASK {questions}",
            }
        ]
    )
    questionnaire_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])

    created = survey_manager.SurveyCreator.from_dataframe(survey_df, questionnaire_df)
    assert len(created) == 1
    assert isinstance(created[0], LLMPrompt)
    assert created[0].questionnaire_name == "S1"

    survey_path = tmp_path / "survey.csv"
    questionnaire_path = tmp_path / "questionnaire.csv"
    survey_df.to_csv(survey_path, index=False)
    questionnaire_df.to_csv(questionnaire_path, index=False)

    loaded = survey_manager.SurveyCreator.from_path(str(survey_path), str(questionnaire_path))
    assert len(loaded) == 1
    assert loaded[0].questionnaire_name == "S1"


def test_survey_creator_allows_missing_system_prompt_column():
    survey_df = pd.DataFrame(
        [
            {
                "questionnaire_name": "S1",
                "questionnaire_instruction": "ASK {questions}",
            }
        ]
    )
    questionnaire_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])

    created = survey_manager.SurveyCreator.from_dataframe(survey_df, questionnaire_df)

    assert created[0].system_prompt is None


def test_survey_creator_converts_nan_system_prompt_to_none():
    survey_df = pd.DataFrame(
        [
            {
                "questionnaire_name": "S1",
                "system_prompt": float("nan"),
                "questionnaire_instruction": "ASK {questions}",
            }
        ]
    )
    questionnaire_df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])

    created = survey_manager.SurveyCreator.from_dataframe(survey_df, questionnaire_df)

    assert created[0].system_prompt is None


def test_survey_manager_main_guard_executes_without_side_effects():
    """Importing survey_manager as __main__ should not run unintended code."""
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        module_globals = runpy.run_module("qstn.survey_manager", run_name="__main__")
    assert "__name__" in module_globals
