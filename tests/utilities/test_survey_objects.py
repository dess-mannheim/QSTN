"""Tests for survey data objects and answer option transformations.

These tests cover:
- `AnswerTexts` formatting and validation behavior,
- `AnswerOptions` rendering and response-method adaptation,
- `InferenceResult` conversion helpers.
"""

import pytest

from qstn.inference.response_generation import (
    ChoiceResponseGenerationMethod,
    JSONItem,
    JSONObject,
    JSONReasoningResponseGenerationMethod,
    JSONResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
    LogprobResponseGenerationMethod,
)
from qstn.utilities import constants, survey_objects


def test_answertexts_various():
    """AnswerTexts should correctly combine texts and indices and enforce validation."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    assert at.full_answers == ["1: a", "2: b"]
    assert "1: a" in at.get_list_answer_texts()

    at2 = survey_objects.AnswerTexts(answer_texts=["x"], indices=None)
    assert at2.full_answers == ["x"]

    at3 = survey_objects.AnswerTexts(answer_texts=None, indices=["10", "20"])
    assert at3.full_answers == ["10", "20"]

    at4 = survey_objects.AnswerTexts(
        answer_texts=["low", "high"], indices=["1", "5"], only_scale=True
    )
    assert at4.full_answers == ["1: low", "2", "3", "4", "5: high"]
    assert at4.answer_texts[0] == "low"
    start, end = at4.get_scale_answer_texts()
    assert start == at4.full_answers[0]

    with pytest.raises(ValueError):
        survey_objects.AnswerTexts(answer_texts=None, indices=None)


def test_answeroptions_list_and_scale_and_response_methods():
    """AnswerOptions should render option strings and adapt to response methods."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    ao = survey_objects.AnswerOptions(answer_texts=at)
    assert "a" in ao.create_options_str()

    ao2 = survey_objects.AnswerOptions(answer_texts=at, from_to_scale=True)
    assert "to" in ao2.create_options_str()

    rmd = JSONVerbalizedDistribution(output_index_only=True)
    ao3 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rmd)
    assert ao3.response_generation_method.json_object.children
    assert not rmd.json_object.children

    rj = JSONResponseGenerationMethod(
        json_object=JSONObject(
            children=[JSONItem(json_field="answer", explanation="choose one of: {options}")]
        ),
        output_index_only=False,
    )
    ao4 = survey_objects.AnswerOptions(
        answer_texts=at,
        list_prompt_template="Options: {options}",
        response_generation_method=rj,
    )
    answer_item = ao4.response_generation_method.json_object.children[0]
    assert isinstance(answer_item, JSONItem)
    assert answer_item.explanation == "choose one of: Options: 1: a, 2: b"
    original_answer_item = rj.json_object.children[0]
    assert isinstance(original_answer_item, JSONItem)
    assert original_answer_item.explanation == "choose one of: {options}"

    scale_method = JSONResponseGenerationMethod(
        json_object=JSONObject(
            children=[JSONItem(json_field="answer", explanation="choose one of: {options}")]
        ),
        output_index_only=False,
    )
    ao4_scale = survey_objects.AnswerOptions(
        answer_texts=at,
        from_to_scale=True,
        scale_prompt_template="Scale: {start} to {end}",
        response_generation_method=scale_method,
    )
    scale_answer_item = ao4_scale.response_generation_method.json_object.children[0]
    assert isinstance(scale_answer_item, JSONItem)
    assert scale_answer_item.explanation == "choose one of: Scale: 1: a to 2: b"

    verbalized_scale = JSONVerbalizedDistribution(
        option_explanation_template=("probability for {option} {{SCALE_RANGE_PLACEHOLDER}}")
    )
    ao4_verbalized = survey_objects.AnswerOptions(
        answer_texts=at,
        from_to_scale=True,
        scale_prompt_template="Scale: {start} to {end}",
        response_generation_method=verbalized_scale,
    )
    verbalized_item = ao4_verbalized.response_generation_method.json_object.children[0]
    assert isinstance(verbalized_item, JSONItem)
    assert verbalized_item.explanation == "probability for 1: a Scale: 1: a to 2: b"
    verbalized_second_item = ao4_verbalized.response_generation_method.json_object.children[1]
    assert isinstance(verbalized_second_item, JSONItem)
    assert verbalized_second_item.explanation == "probability for 2: b"

    verbalized_list = JSONVerbalizedDistribution(
        option_explanation_template=("probability for {option} {{SCALE_RANGE_PLACEHOLDER}}")
    )
    ao4_list = survey_objects.AnswerOptions(
        answer_texts=at,
        response_generation_method=verbalized_list,
    )
    verbalized_list_item = ao4_list.response_generation_method.json_object.children[0]
    assert isinstance(verbalized_list_item, JSONItem)
    assert verbalized_list_item.explanation == "probability for 1: a"

    verbalized_scale_all = JSONVerbalizedDistribution(
        option_explanation_template=("probability for {option} {{SCALE_RANGE_PLACEHOLDER}}"),
        explanation_prompt_placeholders_first_option_only=False,
    )
    ao4_verbalized_all = survey_objects.AnswerOptions(
        answer_texts=at,
        from_to_scale=True,
        scale_prompt_template="Scale: {start} to {end}",
        response_generation_method=verbalized_scale_all,
    )
    all_second_item = ao4_verbalized_all.response_generation_method.json_object.children[1]
    assert isinstance(all_second_item, JSONItem)
    assert all_second_item.explanation == "probability for 2: b Scale: 1: a to 2: b"

    rc = ChoiceResponseGenerationMethod(allowed_choices_template="{options}")
    ao5 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rc)
    assert ao5.response_generation_method.allowed_choices in (
        ao.answer_texts.full_answers,
        ao.answer_texts.indices,
    )
    assert rc.allowed_choices is None

    rl = LogprobResponseGenerationMethod(allowed_choices_template="{options}")
    ao6 = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=rl)
    assert ao6.response_generation_method.allowed_choices is not None
    assert rl.allowed_choices is None


def test_answeroptions_constrains_json_single_answer_options():
    """JSON single-answer schemas should constrain answers to materialized options."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    ao = survey_objects.AnswerOptions(
        answer_texts=at,
        response_generation_method=JSONSingleResponseGenerationMethod(),
    )

    answer_item = ao.response_generation_method.json_object.children[0]

    assert isinstance(answer_item, JSONItem)
    assert answer_item.constraints.enum == ["1: a", "2: b"]


def test_answeroptions_constrains_json_single_answer_indices():
    """JSON answer constraints should respect `output_index_only`."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    ao = survey_objects.AnswerOptions(
        answer_texts=at,
        response_generation_method=JSONSingleResponseGenerationMethod(output_index_only=True),
    )

    answer_item = ao.response_generation_method.json_object.children[0]

    assert isinstance(answer_item, JSONItem)
    assert answer_item.constraints.enum == ["1", "2"]


def test_answeroptions_constrains_json_reasoning_answer_only():
    """JSON reasoning schemas should constrain only the answer field."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    ao = survey_objects.AnswerOptions(
        answer_texts=at,
        response_generation_method=JSONReasoningResponseGenerationMethod(),
    )

    reasoning_item = ao.response_generation_method.json_object.children[0]
    answer_item = ao.response_generation_method.json_object.children[1]

    assert isinstance(reasoning_item, JSONItem)
    assert isinstance(answer_item, JSONItem)
    assert reasoning_item.constraints.enum is None
    assert answer_item.constraints.enum == ["1: a", "2: b"]


def test_answeroptions_can_disable_json_answer_constraints():
    """Turning off JSON answer constraints should leave prompt formatting intact."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    ao = survey_objects.AnswerOptions(
        answer_texts=at,
        response_generation_method=JSONSingleResponseGenerationMethod(
            constrain_answer_options=False
        ),
    )

    answer_item = ao.response_generation_method.json_object.children[0]

    assert isinstance(answer_item, JSONItem)
    assert answer_item.explanation == "choose one of: Options are: 1: a, 2: b"
    assert answer_item.constraints.enum is None


def test_answeroptions_constrains_generic_json_response_field():
    """Generic JSON methods constrain the explicitly configured response field."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    method = JSONResponseGenerationMethod(
        json_object=JSONObject(
            children=[
                JSONItem(json_field="reasoning"),
                JSONItem(json_field="choice", explanation="choose one of: {options}"),
            ]
        ),
        response_field="choice",
    )
    ao = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=method)

    reasoning_item = ao.response_generation_method.json_object.children[0]
    choice_item = ao.response_generation_method.json_object.children[1]

    assert isinstance(reasoning_item, JSONItem)
    assert isinstance(choice_item, JSONItem)
    assert reasoning_item.constraints.enum is None
    assert choice_item.constraints.enum == ["1: a", "2: b"]


def test_answeroptions_generic_json_without_response_field_is_unconstrained():
    """Generic JSON methods need an explicit response field for automatic constraints."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    method = JSONResponseGenerationMethod(
        json_object=JSONObject(children=[JSONItem(json_field="answer")])
    )
    ao = survey_objects.AnswerOptions(answer_texts=at, response_generation_method=method)

    answer_item = ao.response_generation_method.json_object.children[0]

    assert isinstance(answer_item, JSONItem)
    assert answer_item.constraints.enum is None


def test_answeroptions_missing_json_response_field_raises():
    """Configured JSON response fields must exist when automatic constraints run."""
    at = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    method = JSONResponseGenerationMethod(
        json_object=JSONObject(children=[JSONItem(json_field="answer")]),
        response_field="choice",
    )

    with pytest.raises(ValueError, match="JSON response field 'choice' was not found"):
        survey_objects.AnswerOptions(answer_texts=at, response_generation_method=method)


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

    first_fields = [
        child.json_field for child in first_options.response_generation_method.json_object.children
    ]
    second_fields = [
        child.json_field for child in second_options.response_generation_method.json_object.children
    ]

    assert len(first_fields) == 5
    assert len(second_fields) == 2
    assert first_fields[0].startswith("1:")
    assert "1: Strongly agree" in first_fields[0]
    assert second_fields[0].startswith("1:")
    assert "1: Agree" in second_fields[0]
    assert not shared_method.json_object.children


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
