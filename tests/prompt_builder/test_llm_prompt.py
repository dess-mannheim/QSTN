"""Core behavior tests for `LLMPrompt` prompt building and questionnaire operations."""

import pandas as pd
import pytest

from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.utilities import placeholder
from qstn.utilities.constants import QuestionnairePresentation
from qstn.utilities.survey_objects import QuestionnaireItem


def test_load_questionnaire_and_len(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)

    assert len(prompt) == 2
    assert prompt.get_question(0).item_id == 1
    assert prompt.get_question(0).question_content == "How do you feel about Red?"


def test_verbose_parameter_is_deprecated():
    with pytest.warns(DeprecationWarning, match="verbose"):
        LLMPrompt(verbose=True)


def test_prepare_prompt_applies_per_item_configuration(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    options = generate_likert_options(n=2, answer_texts=["No", "Yes"])

    prompt.prepare_prompt(
        question_stem=[
            "Q1: " + placeholder.QUESTION_CONTENT,
            "Q2: " + placeholder.QUESTION_CONTENT,
        ],
        answer_options={1: options},
        prefilled_responses={2: "Already answered"},
    )

    assert prompt.get_question(0).answer_options is options
    assert prompt.get_question(0).prefilled_response is None
    assert prompt.get_question(1).answer_options is None
    assert prompt.get_question(1).prefilled_response == "Already answered"


def test_get_prompt_single_item_renders_question_and_options(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        system_prompt="System",
        prompt=f"Ask: {placeholder.PROMPT_QUESTIONS}",
    )
    options = generate_likert_options(n=2, answer_texts=["Bad", "Good"])
    prompt.prepare_prompt(
        question_stem=f"{placeholder.QUESTION_CONTENT}\n{placeholder.PROMPT_OPTIONS}",
        answer_options=options,
    )

    system_message, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM, item_position=0
    )

    assert system_message == "System"
    assert "How do you feel about Red?" in user_prompt
    assert "Bad" in user_prompt
    assert "Good" in user_prompt


def test_get_prompt_battery_uses_separator(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        prompt=f"Questions:\n{placeholder.PROMPT_QUESTIONS}",
    )

    _, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator=" || ",
    )

    assert "How do you feel about Red?" in user_prompt
    assert "How do you feel about Blue?" in user_prompt
    assert " || " in user_prompt


def test_insert_set_and_delete_questions(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)

    inserted = QuestionnaireItem(item_id=3, question_content="How do you feel about Green?")
    replacement = QuestionnaireItem(item_id=99, question_content="Replacement")
    prompt.insert_questions(inserted, 1)
    prompt.replace_question(0, replacement)
    prompt.remove_question(2)

    assert len(prompt) == 2
    assert prompt.get_question(0).item_id == 99
    assert prompt.get_question(1).item_id == 3


def test_load_questionnaire_format_rejects_empty_dataframe():
    prompt = LLMPrompt()

    with pytest.warns(UserWarning, match="provided Dataframe is empty"):
        with pytest.raises(ValueError, match="non empty DataFrame"):
            prompt.load_questionnaire_format(questionnaire_source=pd.DataFrame())
