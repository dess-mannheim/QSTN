"""Tests for survey data objects and answer option transformations.

These tests cover:
- `AnswerTexts` formatting and validation behavior,
- `AnswerOptions` rendering and response-method adaptation,
- `InferenceResult` conversion helpers.
"""

import pytest

from qstn.utilities import survey_objects, constants
from qstn.inference.response_generation import (
    JSONVerbalizedDistribution,
    JSONResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)


def test_answertexts_various():
    """AnswerTexts should correctly combine texts and indices and enforce validation."""
    # both texts and indices
    at = survey_objects.AnswerTexts(answer_texts=["a","b"], indices=["1","2"])
    assert at.full_answers == ["1: a","2: b"]
    assert "1: a" in at.get_list_answer_texts()

    # only texts
    at2 = survey_objects.AnswerTexts(answer_texts=["x"], indices=None)
    assert at2.full_answers == ["x"]

    # only indices
    at3 = survey_objects.AnswerTexts(answer_texts=None, indices=["10","20"])
    assert at3.full_answers == ["10","20"]

    # scale mode
    at4 = survey_objects.AnswerTexts(answer_texts=["low","high"], indices=["1","5"], only_scale=True)
    assert at4.answer_texts[0] == "low"
    # get scale texts
    start, end = at4.get_scale_answer_texts()
    assert start == at4.full_answers[0]

    with pytest.raises(ValueError):
        survey_objects.AnswerTexts(answer_texts=None, indices=None)


def test_answeroptions_list_and_scale_and_response_methods():
    """AnswerOptions should render option strings and adapt to response methods."""
    at = survey_objects.AnswerTexts(answer_texts=["a","b"], indices=["1","2"])
    ao = survey_objects.AnswerOptions(answer_texts=at)
    assert "a" in ao.create_options_str()

    # scale prompts
    ao2 = survey_objects.AnswerOptions(answer_texts=at, from_to_scale=True)
    # since from_to_scale expects at least two
    assert "to" in ao2.create_options_str()

    # test response_generation_method adjustments
    rmd = JSONVerbalizedDistribution(output_index_only=True)
    ao3 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rmd)
    assert rmd.json_fields is not None

    # JSONResponseGenerationMethod with OPTIONS_ADJUST
    rj = JSONResponseGenerationMethod(json_fields={"answer": constants.OPTIONS_ADJUST}, constraints={"answer": constants.OPTIONS_ADJUST}, output_index_only=False)
    ao4 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rj)
    assert isinstance(rj.json_fields, dict)

    # choice method
    rc = ChoiceResponseGenerationMethod(allowed_choices=constants.OPTIONS_ADJUST)
    ao5 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rc)
    assert rc.allowed_choices == ao.answer_texts.full_answers or rc.allowed_choices == ao.answer_texts.indices

    # logprob method
    rl = LogprobResponseGenerationMethod(allowed_choices=constants.OPTIONS_ADJUST)
    ao6 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rl)
    assert rl.allowed_choices is not None


def test_inferenceresult_and_tuple():
    """InferenceResult conversion utilities produce DataFrame and transcript correctly."""
    class DummyQ:
        def __init__(self):
            self._questions = ["q1"]
        def generate_question_prompt(self, q):
            return f"Q:{q}"
    res = survey_objects.InferenceResult(
        questionnaire=DummyQ(),
        results={1: survey_objects.QuestionLLMResponseTuple(question="q1", llm_response="ans", logprobs=None, reasoning=None)}
    )
    df = res.to_dataframe()
    assert constants.QUESTIONNAIRE_ITEM_ID in df.columns
    transcript = res.get_questions_transcript()
    assert "Q:q1" in transcript
