"""Tests for JSON/logprob parsing helpers in `qstn.parser.llm_answer_parser`."""

import math

import pandas as pd
import pytest

from qstn.parser.llm_answer_parser import (
    _filter_logprobs_by_choices,
    _logprobs_filter,
    parse_json,
    parse_json_battery,
    parse_json_str,
    parse_logprobs,
    raw_responses,
)
from qstn.prompt_builder import LLMPrompt
from qstn.utilities import constants
from qstn.utilities.survey_objects import InferenceResult, QuestionLLMResponseTuple


def _make_prompt():
    questionnaire = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Red?"},
            {"questionnaire_item_id": 2, "question_content": "Blue?"},
        ]
    )
    return LLMPrompt(questionnaire_source=questionnaire)


def test_parse_json_str_handles_valid_repair_and_invalid_json():
    assert parse_json_str('{"answer": "Yes"}') == {"answer": "Yes"}
    assert parse_json_str("{answer: 'Yes'}") == {"answer": "Yes"}


def test_parse_json_str_returns_none_when_both_parsers_fail(monkeypatch):
    monkeypatch.setattr("qstn.parser.llm_answer_parser.json.loads", lambda *_: (_ for _ in ()).throw(ValueError("x")))
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
                llm_response='{"answer_Red?":"1","answer_Blue?":"2"}',
                logprobs=None,
                reasoning=None,
            )
        },
    )

    parsed = parse_json_battery([battery_result])[prompt]

    assert sorted(parsed[constants.QUESTIONNAIRE_ITEM_ID].tolist()) == [1, 2]
    assert "answer" in parsed.columns
    assert set(parsed["answer"]) == {"1", "2"}


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
