"""Tests for JSON/logprob parsing helpers in `qstn.parser.llm_answer_parser`."""

import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from qstn.inference.response_generation import (
    JSONItem,
    JSONObject,
    JSONReasoningResponseGenerationMethod,
    JSONResponseGenerationMethod,
    JSONVerbalizedDistribution,
)
from qstn.parser.llm_answer_parser import (
    SOURCE_LLM_RESPONSE_COLUMN,
    _filter_logprobs_by_choices,
    _logprobs_filter,
    parse_json,
    parse_json_battery,
    parse_json_str,
    parse_logprobs,
    parse_with_llm,
    parse_with_llm_battery,
    raw_responses,
)
from qstn.prompt_builder import LLMPrompt
from qstn.utilities import constants
from qstn.utilities.survey_objects import (
    AnswerOptions,
    AnswerTexts,
    InferenceResult,
    QuestionLLMResponseTuple,
)


def _make_prompt():
    questionnaire = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Red?"},
            {"questionnaire_item_id": 2, "question_content": "Blue?"},
        ]
    )
    return LLMPrompt(questionnaire_source=questionnaire)


def _make_prompt_with_answer_options(
    answer_texts_q1: list[str],
    answer_texts_q2: list[str],
    indices_q1: list[str] | None = None,
    indices_q2: list[str] | None = None,
    response_generation_method=None,
):
    prompt = _make_prompt()
    answer_options = {
        1: AnswerOptions(
            answer_texts=AnswerTexts(answer_texts=answer_texts_q1, indices=indices_q1),
            response_generation_method=response_generation_method,
        ),
        2: AnswerOptions(
            answer_texts=AnswerTexts(answer_texts=answer_texts_q2, indices=indices_q2),
            response_generation_method=response_generation_method,
        ),
    }
    return prompt.prepare_prompt(answer_options=answer_options)


def test_parse_json_str_handles_valid_repair_and_invalid_json():
    assert parse_json_str('{"answer": "Yes"}') == {"answer": "Yes"}
    assert parse_json_str("{answer: 'Yes'}") == {"answer": "Yes"}


def test_parse_json_str_returns_none_when_both_parsers_fail(monkeypatch):
    monkeypatch.setattr(
        "qstn.parser.llm_answer_parser.json.loads",
        lambda *_: (_ for _ in ()).throw(ValueError("x")),
    )
    monkeypatch.setattr(
        "qstn.parser.llm_answer_parser.json_repair.loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("y")),
    )

    assert parse_json_str("anything") is None


def test_parse_json_removes_reserved_fields_and_keeps_reasoning_and_logprobs():
    prompt = _make_prompt()
    result = InferenceResult(
        questionnaire=prompt,
        results={
            1: QuestionLLMResponseTuple(
                question="Q1",
                llm_response='{"answer":"Yes","questionnaire_item_id":1,"question":"Q1"}',
                logprobs={"Yes": -0.1},
                reasoning="Because",
            )
        },
    )

    parsed = parse_json([result])[prompt]

    assert list(parsed.columns) == [
        constants.QUESTIONNAIRE_ITEM_ID,
        constants.QUESTION,
        "answer",
        "built_in_reasoning",
        "logprobs",
    ]
    assert parsed.loc[0, "answer"] == "Yes"
    assert parsed.loc[0, "built_in_reasoning"] == "Because"
    assert parsed.loc[0, "logprobs"] == {"Yes": -0.1}


def test_parse_json_falls_back_to_error_row_on_invalid_json():
    prompt = _make_prompt()
    result = InferenceResult(
        questionnaire=prompt,
        results={
            1: QuestionLLMResponseTuple(
                question="Q1",
                llm_response="not-json",
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json([result])[prompt]

    assert parsed.loc[0, constants.LLM_RESPONSE] == "not-json"
    assert parsed.loc[0, "error_col"] == "ERROR: Parsing"


def test_parse_json_battery_groups_columns_per_question():
    prompt = _make_prompt()
    battery_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response='{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([battery_result])[prompt]

    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]
    assert "answer" in parsed.columns
    assert set(parsed["answer"]) == {"1", "2"}


def test_parse_json_battery_unknown_question_keys_return_error():
    prompt = _make_prompt()
    battery_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response='{"Unknown":{"answer":"0.4"}}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([battery_result])[prompt]

    assert "error_col" in parsed.columns
    assert "Unknown battery JSON keys" in parsed.loc[0, "error_col"]
    assert "Unknown" in parsed.loc[0, "error_col"]


def test_parse_json_battery_requires_question_entries_to_be_objects():
    prompt = _make_prompt()
    battery_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response='{"Red?":"0.1","Blue?":{"answer":"0.9"}}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([battery_result])[prompt]

    assert "error_col" in parsed.columns
    assert "must be objects" in parsed.loc[0, "error_col"]
    assert "Red?" in parsed.loc[0, "error_col"]


def test_parse_json_battery_nested_distribution_maps_columns_per_question():
    prompt = _make_prompt_with_answer_options(
        answer_texts_q1=["GAR KEIN VERTRAUEN", "VIEL VERTRAUEN"],
        answer_texts_q2=["SCHLECHT", "GUT"],
        indices_q1=["1", "2"],
        indices_q2=["1", "2"],
        response_generation_method=JSONVerbalizedDistribution(),
    )
    battery_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response=(
                    '{"Red?":{"1: GAR KEIN VERTRAUEN":"0.1"},' '"Blue?":{"2: GUT":"0.9"}}'
                ),
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([battery_result])[prompt]

    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]
    assert "1: GAR KEIN VERTRAUEN" in parsed.columns
    assert "2: GUT" in parsed.columns
    assert set(parsed["1: GAR KEIN VERTRAUEN"].dropna()) == {"0.1"}
    assert set(parsed["2: GUT"].dropna()) == {"0.9"}


def test_parse_json_battery_nested_payload_flattens_inner_objects():
    prompt = _make_prompt()
    battery_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response='{"Red?":{"metrics":{"score":0.2}},"Blue?":{"metrics":{"score":0.8}}}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([battery_result])[prompt]

    assert "metrics.score" in parsed.columns
    assert set(parsed["metrics.score"]) == {0.2, 0.8}


def test_parse_json_battery_returns_error_frame_unchanged():
    prompt = _make_prompt()
    invalid_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response="broken",
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([invalid_result])[prompt]
    assert "error_col" in parsed.columns
    assert parsed.loc[0, "error_col"] == "ERROR: Parsing"


def test_parse_json_auto_routes_battery_to_battery_parser():
    prompt = _make_prompt()
    battery_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response='{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed_from_router = parse_json([battery_result])[prompt]
    parsed_explicit = parse_json_battery([battery_result])[prompt]
    assert_frame_equal(parsed_from_router, parsed_explicit)


def test_parse_json_mixed_input_handles_battery_and_non_battery():
    prompt_non_battery = _make_prompt()
    prompt_battery = _make_prompt()
    non_battery_result = InferenceResult(
        questionnaire=prompt_non_battery,
        results={
            1: QuestionLLMResponseTuple("Q1", '{"answer":"Yes"}', None, None),
        },
    )
    battery_result = InferenceResult(
        questionnaire=prompt_battery,
        results={
            -1: QuestionLLMResponseTuple(
                question="battery",
                llm_response='{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json([non_battery_result, battery_result])

    assert set(parsed.keys()) == {prompt_non_battery, prompt_battery}
    assert parsed[prompt_non_battery].shape[0] == 1
    assert sorted(parsed[prompt_battery][constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]


def test_parse_with_llm_returns_parse_json_shape_and_source_trace():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={
            1: QuestionLLMResponseTuple("Q1", "free-text-1", None, None),
            2: QuestionLLMResponseTuple("Q2", "free-text-2", None, None),
        },
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["response_generation_method"] = kwargs["response_generation_method"]
        return ['{"answer":"Yes"}', '{"answer":"No"}'], None, None

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist() == [1, 2]
    assert "answer" in parsed.columns
    assert SOURCE_LLM_RESPONSE_COLUMN in parsed.columns
    assert parsed[SOURCE_LLM_RESPONSE_COLUMN].tolist() == ["free-text-1", "free-text-2"]
    methods = captured["response_generation_method"]
    assert isinstance(methods, list)
    assert len(methods) == 2
    assert all(isinstance(method, JSONResponseGenerationMethod) for method in methods)


def test_parse_with_llm_invalid_json_keeps_error_row_and_source_trace():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        generation_fn=lambda **_kwargs: (["not-json"], None, None),
        print_progress=False,
    )[prompt]

    assert parsed.loc[0, constants.LLM_RESPONSE] == "not-json"
    assert parsed.loc[0, "error_col"] == "ERROR: Parsing"
    assert parsed.loc[0, SOURCE_LLM_RESPONSE_COLUMN] == "raw-answer"


def test_parse_with_llm_preserves_reasoning_and_logprobs():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        generation_fn=lambda **_kwargs: (
            ['{"answer":"Yes"}'],
            [{"Yes": -0.1}],
            ["Because"],
        ),
        print_progress=False,
    )[prompt]

    assert parsed.loc[0, "answer"] == "Yes"
    assert parsed.loc[0, "built_in_reasoning"] == "Because"
    assert parsed.loc[0, "logprobs"] == {"Yes": -0.1}
    assert parsed.loc[0, SOURCE_LLM_RESPONSE_COLUMN] == "raw-answer"


def test_parse_with_llm_forwards_custom_response_generation_method():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )
    custom_method = object()
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["response_generation_method"] = kwargs["response_generation_method"]
        return ['{"answer":"Yes"}'], None, None

    parse_with_llm(
        model=object(),
        survey_results=[source_result],
        response_generation_method=custom_method,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )

    assert captured["response_generation_method"] == [custom_method]


def test_parse_with_llm_uses_rgm_automatic_output_instructions_in_prompt():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )
    reasoning_rgm = JSONReasoningResponseGenerationMethod()
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["prompt"] = kwargs["prompts"][0]
        captured["method"] = kwargs["response_generation_method"][0]
        return ['{"reasoning":"Because","answer":"Yes"}'], None, None

    parse_with_llm(
        model=object(),
        survey_results=[source_result],
        response_generation_method=reasoning_rgm,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )

    automatic_template = captured["method"].get_automatic_prompt()
    assert automatic_template in captured["prompt"]


def test_parse_with_llm_reasoning_rgm_works_out_of_the_box():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        response_generation_method=JSONReasoningResponseGenerationMethod(),
        generation_fn=lambda **_kwargs: (
            ['{"reasoning":"Because","answer":"Yes"}'],
            None,
            None,
        ),
        print_progress=False,
    )[prompt]

    assert parsed.loc[0, "reasoning"] == "Because"
    assert parsed.loc[0, "answer"] == "Yes"


def test_parse_with_llm_no_answer_option_is_available():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["methods"] = kwargs["response_generation_method"]
        return ['{"answer":"LLM_DID_NOT_ANSWER"}'], None, None

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        no_answer_option="LLM_DID_NOT_ANSWER",
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert parsed.loc[0, "answer"] == "LLM_DID_NOT_ANSWER"
    answer_item = captured["methods"][0].json_object.children[0]
    assert "LLM_DID_NOT_ANSWER" in answer_item.constraints.enum


def test_parse_with_llm_accepts_answer_options_override_for_optionless_questions():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )
    captured: dict[str, object] = {}
    answer_options = {1: AnswerOptions(answer_texts=AnswerTexts(["Yes", "No"]))}

    def fake_generation_fn(**kwargs):
        captured["methods"] = kwargs["response_generation_method"]
        captured["prompt"] = kwargs["prompts"][0]
        return ['{"answer":"Yes"}'], None, None

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        answer_options=answer_options,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert parsed.loc[0, "answer"] == "Yes"
    answer_item = captured["methods"][0].json_object.children[0]
    assert answer_item.constraints.enum == ["Yes", "No"]
    assert "Yes" in captured["prompt"] and "No" in captured["prompt"]


def test_parse_with_llm_no_options_avoids_unsatisfiable_answer_constraints():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["method"] = kwargs["response_generation_method"][0]
        return ['{"answer":"free text"}'], None, None

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert parsed.loc[0, "answer"] == "free text"
    method = captured["method"]
    answer_item = method.json_object.children[0]
    assert answer_item.constraints.enum is None


def test_parse_with_llm_battery_parses_to_expanded_rows_with_source_trace():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        generation_fn=lambda **_kwargs: (
            ['{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}'],
            None,
            None,
        ),
        print_progress=False,
    )[prompt]

    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]
    assert set(parsed["answer"]) == {"1", "2"}
    assert set(parsed[SOURCE_LLM_RESPONSE_COLUMN]) == {"raw-battery"}


def test_parse_with_llm_battery_use_parser_false_returns_aggregated_row():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        use_parser=False,
        response_generation_method=None,
        generation_fn=lambda **_kwargs: (["open battery answer"], None, None),
        print_progress=False,
    )[prompt]

    assert parsed.shape[0] == 1
    assert parsed.loc[0, constants.QUESTIONNAIRE_ITEM_ID] == -1
    assert parsed.loc[0, constants.LLM_RESPONSE] == "open battery answer"
    assert parsed.loc[0, SOURCE_LLM_RESPONSE_COLUMN] == "raw-battery"


def test_parse_with_llm_auto_routes_battery_and_matches_explicit_battery_parser():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )

    def fake_generation_fn(**_kwargs):
        return ['{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}'], None, None

    parsed_from_router = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]
    parsed_explicit = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert_frame_equal(parsed_from_router, parsed_explicit)


def test_parse_with_llm_allows_minus_one_item_id_when_not_battery():
    questionnaire = pd.DataFrame(
        [
            {"questionnaire_item_id": -1, "question_content": "Q-neg"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    prompt = LLMPrompt(questionnaire_source=questionnaire)
    source_result = InferenceResult(
        questionnaire=prompt,
        results={
            -1: QuestionLLMResponseTuple("Q-neg", "raw-neg", None, None),
            2: QuestionLLMResponseTuple("Q2", "raw-two", None, None),
        },
    )

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        generation_fn=lambda **_kwargs: (
            ['{"answer":"NEG"}', '{"answer":"TWO"}'],
            None,
            None,
        ),
        print_progress=False,
    )[prompt]

    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [-1, 2]
    assert set(parsed["answer"]) == {"NEG", "TWO"}
    assert set(parsed[SOURCE_LLM_RESPONSE_COLUMN]) == {"raw-neg", "raw-two"}


def test_parse_with_llm_mixed_input_handles_battery_and_non_battery():
    prompt_non_battery = _make_prompt()
    prompt_battery = _make_prompt()
    non_battery_result = InferenceResult(
        questionnaire=prompt_non_battery,
        results={1: QuestionLLMResponseTuple("Q1", "raw-single", None, None)},
    )
    battery_result = InferenceResult(
        questionnaire=prompt_battery,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )

    def fake_generation_fn(**kwargs):
        outputs = []
        for prompt_text in kwargs["prompts"]:
            if "raw-battery" in prompt_text:
                outputs.append('{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}')
            else:
                outputs.append('{"answer":"Yes"}')
        return outputs, None, None

    parsed = parse_with_llm(
        model=object(),
        survey_results=[non_battery_result, battery_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )

    assert set(parsed.keys()) == {prompt_non_battery, prompt_battery}
    assert parsed[prompt_non_battery].shape[0] == 1
    assert sorted(parsed[prompt_battery][constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]


def test_parse_with_llm_allows_unconstrained_open_answers_without_parser():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={1: QuestionLLMResponseTuple("Q1", "raw-answer", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["response_generation_method"] = kwargs["response_generation_method"]
        return ["open and unconstrained answer"], None, None

    parsed = parse_with_llm(
        model=object(),
        survey_results=[source_result],
        use_parser=False,
        response_generation_method=None,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert captured["response_generation_method"] is None
    assert parsed.loc[0, constants.LLM_RESPONSE] == "open and unconstrained answer"
    assert parsed.loc[0, SOURCE_LLM_RESPONSE_COLUMN] == "raw-answer"
    assert constants.QUESTIONNAIRE_ITEM_ID in parsed.columns
    assert constants.QUESTION in parsed.columns


def test_parse_with_llm_battery_uses_default_battery_json_keys_when_parser_enabled():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["response_generation_method"] = kwargs["response_generation_method"]
        return ['{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}'], None, None

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    methods = captured["response_generation_method"]
    assert isinstance(methods, list)
    assert len(methods) == 1
    assert {child.json_field for child in methods[0].json_object.children} == {"Red?", "Blue?"}
    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]


def test_parse_with_llm_battery_accepts_custom_top_level_key_templates():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    custom_method = JSONResponseGenerationMethod(
        json_object=JSONObject(children=[JSONItem(json_field="answer")]),
        battery_question_key_template="item: {{QUESTION_CONTENT_PLACEHOLDER}}",
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["response_generation_method"] = kwargs["response_generation_method"]
        return ['{"item: Red?":{"answer":"1"},"item: Blue?":{"answer":"2"}}'], None, None

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        response_generation_method=custom_method,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    methods = captured["response_generation_method"]
    assert isinstance(methods, list)
    assert {child.json_field for child in methods[0].json_object.children} == {
        "item: Red?",
        "item: Blue?",
    }
    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]


def test_parse_with_llm_battery_forwards_custom_response_generation_method():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    custom_method = object()
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["response_generation_method"] = kwargs["response_generation_method"]
        return ['{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}'], None, None

    parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        response_generation_method=custom_method,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )

    assert captured["response_generation_method"] == [custom_method]


def test_parse_with_llm_battery_uses_rgm_automatic_output_instructions_in_prompt():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["prompt"] = kwargs["prompts"][0]
        captured["method"] = kwargs["response_generation_method"][0]
        return ['{"Red?":{"answer":"1"},"Blue?":{"answer":"2"}}'], None, None

    parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )

    automatic_template = captured["method"].get_automatic_prompt()
    assert automatic_template in captured["prompt"]


def test_parse_with_llm_battery_no_answer_option_is_available():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["methods"] = kwargs["response_generation_method"]
        return ['{"Red?":{"answer":"LLM_DID_NOT_ANSWER"},"Blue?":{"answer":"2"}}'], None, None

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        no_answer_option="LLM_DID_NOT_ANSWER",
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert "LLM_DID_NOT_ANSWER" in parsed["answer"].tolist()
    methods = captured["methods"]
    question_one = next(
        child for child in methods[0].json_object.children if child.json_field == "Red?"
    )
    answer_item = next(child for child in question_one.children if child.json_field == "answer")
    assert "LLM_DID_NOT_ANSWER" in answer_item.constraints.enum


def test_parse_with_llm_battery_accepts_answer_options_override_for_optionless_questions():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    captured: dict[str, object] = {}
    answer_options = {
        1: AnswerOptions(answer_texts=AnswerTexts(["Warm", "Cold"])),
        2: AnswerOptions(answer_texts=AnswerTexts(["Blue", "Green"])),
    }

    def fake_generation_fn(**kwargs):
        captured["methods"] = kwargs["response_generation_method"]
        captured["prompt"] = kwargs["prompts"][0]
        return ['{"Red?":{"answer":"Warm"},"Blue?":{"answer":"Blue"}}'], None, None

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        answer_options=answer_options,
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert set(parsed["answer"]) == {"Warm", "Blue"}
    question_one = next(
        child for child in captured["methods"][0].json_object.children if child.json_field == "Red?"
    )
    answer_one = next(child for child in question_one.children if child.json_field == "answer")
    question_two = next(
        child
        for child in captured["methods"][0].json_object.children
        if child.json_field == "Blue?"
    )
    answer_two = next(child for child in question_two.children if child.json_field == "answer")
    assert answer_one.constraints.enum == ["Warm", "Cold"]
    assert answer_two.constraints.enum == ["Blue", "Green"]
    assert "Warm" in captured["prompt"] and "Green" in captured["prompt"]


def test_parse_with_llm_battery_no_options_avoids_unsatisfiable_answer_constraints():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )
    captured: dict[str, object] = {}

    def fake_generation_fn(**kwargs):
        captured["method"] = kwargs["response_generation_method"][0]
        return ['{"Red?":{"answer":"free-a"},"Blue?":{"answer":"free-b"}}'], None, None

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        generation_fn=fake_generation_fn,
        print_progress=False,
    )[prompt]

    assert set(parsed["answer"]) == {"free-a", "free-b"}
    method = captured["method"]
    assert all(
        next(child for child in question.children if child.json_field == "answer").constraints.enum
        is None
        for question in method.json_object.children
    )


def test_parse_with_llm_battery_reasoning_json_parses_nested_question_objects():
    prompt = _make_prompt()
    source_result = InferenceResult(
        questionnaire=prompt,
        results={-1: QuestionLLMResponseTuple("battery", "raw-battery", None, None)},
    )

    parsed = parse_with_llm_battery(
        model=object(),
        survey_results=[source_result],
        response_generation_method=JSONReasoningResponseGenerationMethod(),
        generation_fn=lambda **_kwargs: (
            [
                '{"Red?":{"reasoning":"r1","answer":"a1"},'
                '"Blue?":{"reasoning":"r2","answer":"a2"}}'
            ],
            None,
            None,
        ),
        print_progress=False,
    )[prompt]

    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]
    assert set(parsed["answer"]) == {"a1", "a2"}
    assert set(parsed["reasoning"]) == {"r1", "r2"}


def test_raw_responses_returns_to_dataframe_mapping():
    prompt = _make_prompt()
    result = InferenceResult(
        questionnaire=prompt,
        results={
            1: QuestionLLMResponseTuple("Q1", "A1", None, None),
            2: QuestionLLMResponseTuple("Q2", "A2", None, None),
        },
    )

    parsed = raw_responses([result])[prompt]

    assert parsed.shape[0] == 2
    assert parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist() == [1, 2]


def test_filter_logprobs_by_choices_keeps_matching_prefix_tokens():
    logprob_df = pd.DataFrame({"token": ["Y", "N", "Z"], "prob": [0.4, 0.5, 0.1]})
    filtered = _filter_logprobs_by_choices(logprob_df, pd.Series(["Yes", "No"]))

    assert filtered["token"].tolist() == ["Y", "N"]


def test_logprobs_filter_normalizes_and_warns_for_missing_choice():
    with pytest.warns(UserWarning, match="Could not find logprobs"):
        filtered = _logprobs_filter(
            {"Y": -0.1, "N": -0.2},
            {"yes": ["Y"], "no": ["N"], "invalid": ["X"]},
        )

    assert pytest.approx(filtered["yes"] + filtered["no"], abs=1e-9) == 1.0
    assert math.isnan(filtered["invalid"])


def test_parse_logprobs_handles_list_choices():
    prompt = _make_prompt()
    result = InferenceResult(
        questionnaire=prompt,
        results={
            1: QuestionLLMResponseTuple("Q1", "A1", {"Y": -0.1, "N": -0.2}, None),
            2: QuestionLLMResponseTuple("Q2", "A2", {"Y": -0.3, "N": -0.4}, None),
        },
    )

    parsed = parse_logprobs([result], allowed_choices=["Y", "N"])[prompt]

    assert constants.QUESTIONNAIRE_ITEM_ID in parsed.columns
    assert parsed.shape[0] == 2
    assert "Y" in parsed.columns and "N" in parsed.columns
    assert "error_col" not in parsed.columns


def test_parse_logprobs_mixed_logprob_and_none_marks_missing_rows():
    prompt = _make_prompt()
    result = InferenceResult(
        questionnaire=prompt,
        results={
            1: QuestionLLMResponseTuple("Q1", "A1", {"Y": -0.1, "N": -0.2}, None),
            2: QuestionLLMResponseTuple("Q2", "A2", None, None),
        },
    )

    with pytest.warns(UserWarning, match="No logprobs found"):
        parsed = parse_logprobs([result], allowed_choices=["Y", "N"])[prompt]

    assert "error_col" in parsed.columns
    assert parsed.loc[0, "error_col"] != "MISSING_LOGPROBS"
    assert parsed.loc[1, "error_col"] == "MISSING_LOGPROBS"
    assert math.isnan(parsed.loc[1, "Y"])
    assert math.isnan(parsed.loc[1, "N"])
