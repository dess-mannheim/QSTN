"""Tests for survey data objects and answer option transformations.

These tests cover:
- `AnswerTexts` formatting and validation behavior,
- `AnswerOptions` rendering and response-method adaptation,
- `InferenceResult` conversion helpers.
"""

import pytest

from qstn.inference.response_generation import (
    ChoiceResponseGenerationMethod,
    JSONResponseGenerationMethod,
    JSONVerbalizedDistribution,
    LogprobResponseGenerationMethod,
)
from qstn.utilities import constants, survey_objects


def test_answertexts_various():
    """AnswerTexts should correctly combine texts and indices and enforce validation."""
    # both texts and indices
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    assert at.full_answers == ["1: a", "2: b"]
    assert "1: a" in at.get_list_answer_texts()

    # only texts
    at2 = survey_objects.AnswerTexts(answer_texts=["x"], indices=None)
    assert at2.full_answers == ["x"]

    # only indices
    at3 = survey_objects.AnswerTexts(answer_texts=None, indices=["10", "20"])
    assert at3.full_answers == ["10", "20"]

    # scale mode
    at4 = survey_objects.AnswerTexts(
        answer_texts=["low", "high"], indices=["1", "5"], only_scale=True
    )
    assert at4.answer_texts[0] == "low"
    # get scale texts
    start, end = at4.get_scale_answer_texts()
    assert start == at4.full_answers[0]

    with pytest.raises(ValueError):
        survey_objects.AnswerTexts(answer_texts=None, indices=None)


def test_answeroptions_list_and_scale_and_response_methods():
    """AnswerOptions should render option strings and adapt to response methods."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    ao = survey_objects.AnswerOptions(answer_texts=at)
    assert "a" in ao.create_options_str()

    # scale prompts
    ao2 = survey_objects.AnswerOptions(answer_texts=at, from_to_scale=True)
    # since from_to_scale expects at least two
    assert "to" in ao2.create_options_str()

    # test response_generation_method adjustments
    rmd = JSONVerbalizedDistribution(output_index_only=True)
    ao3 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rmd)
    assert ao3.response_generation_method.json_fields is not None
    assert rmd.json_fields is None

    # JSONResponseGenerationMethod with OPTIONS_ADJUST
    rj = JSONResponseGenerationMethod(
        json_fields={"answer": constants.OPTIONS_ADJUST},
        constraints={"answer": constants.OPTIONS_ADJUST},
        output_index_only=False,
    )
    ao4 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rj)
    assert isinstance(ao4.response_generation_method.json_fields, dict)
    assert rj.json_fields == {"answer": constants.OPTIONS_ADJUST}

    # choice method
    rc = ChoiceResponseGenerationMethod(allowed_choices=constants.OPTIONS_ADJUST)
    ao5 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rc)
    assert ao5.response_generation_method.allowed_choices in (
        ao.answer_texts.full_answers,
        ao.answer_texts.indices,
    )
    assert rc.allowed_choices == constants.OPTIONS_ADJUST

    # logprob method
    rl = LogprobResponseGenerationMethod(allowed_choices=constants.OPTIONS_ADJUST)
    ao6 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rl)
    assert ao6.response_generation_method.allowed_choices is not None
    assert rl.allowed_choices == constants.OPTIONS_ADJUST


def test_answeroptions_isolates_shared_response_generation_method_instances():
    """Reusing one response method instance should not leak state across questions."""
    shared_method = JSONVerbalizedDistribution()
    at_first = survey_objects.AnswerTexts(
        answer_texts=["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree"],
        indices=["1", "2", "3", "4", "5"],
    )
    at_second = survey_objects.AnswerTexts(
        answer_texts=["Agree", "Disagree"],
        indices=["1", "2"],
    )

    first_options = survey_objects.AnswerOptions(
        answer_texts=at_first,
        response_generation_method=shared_method,
    )
    second_options = survey_objects.AnswerOptions(
        answer_texts=at_second,
        response_generation_method=shared_method,
    )

    first_fields = list(first_options.response_generation_method.json_fields.keys())
    second_fields = list(second_options.response_generation_method.json_fields.keys())

    assert len(first_fields) == 5
    assert len(second_fields) == 2
    assert first_fields[0].startswith("1:")
    assert "1: Strongly agree" in first_fields[0]
    assert second_fields[0].startswith("1:")
    assert "1: Agree" in second_fields[0]
    assert shared_method.json_fields is None


def test_inferenceresult_and_tuple():
    """InferenceResult conversion utilities produce DataFrame and transcript correctly."""

    class DummyQ:
        def __init__(self):
            self._questions = ["q1"]

        def generate_question_prompt(self, q):
            return f"Q:{q}"

    res = survey_objects.InferenceResult(
        questionnaire=DummyQ(),
        results={
            1: survey_objects.QuestionLLMResponseTuple(
                question="q1", llm_response="ans", logprobs=None, reasoning=None
            )
        },
    )
    df = res.to_dataframe()
    assert constants.QUESTIONNAIRE_ITEM_ID in df.columns
    transcript = res.get_questions_transcript()
    assert "Q:q1" in transcript


def test_json_verbalized_distribution_templates_use_question_and_option_placeholders():
    method = JSONVerbalizedDistribution(
        json_field_template=(
            "Question={question}; q{question_order}_o{option_index}; Option={option}"
        ),
        json_explanation_template="probability for: {option}",
    )
    method.set_verbalized_options(options=["1: Ja", "2: Nein"], question="Q1", question_order=1)

    assert list(method.json_fields.keys()) == [
        "Question=Q1; q1_o1; Option=1: Ja",
        "Question=Q1...; q1_o2; Option=2: Nein",
    ]
    assert method.json_fields["Question=Q1; q1_o1; Option=1: Ja"] == "probability for: 1: Ja"
