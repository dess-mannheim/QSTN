"""Core behavior tests for `LLMPrompt` prompt building and questionnaire operations."""

import sys
import types

import pandas as pd
import pytest

from qstn.prompt_builder import BaseModelPromptTemplate, LLMPrompt, generate_likert_options
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


def test_base_model_prompt_template_setter_is_fluent(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)

    returned = prompt.set_base_model_prompt_template(
        user_prefix="### Instruction:",
        assistant_prefix="### Response:",
        separator="\n---\n",
    )

    assert returned is prompt
    assert prompt.base_model_prompt_template.user_prefix == "### Instruction:"
    assert prompt.base_model_prompt_template.assistant_prefix == "### Response:"
    assert prompt.base_model_prompt_template.separator == "\n---\n"


def test_generation_uses_plain_prompt_text_by_default(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    ).set_base_model_prompt_template()

    system_message, rendered_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM,
        item_position=0,
        inference_mode="completion",
    )

    assert system_message is None
    assert rendered_prompt.startswith("SYS\nASK")
    assert prompt.base_model_prompt_template.user_prefix is None
    assert prompt.base_model_prompt_template.assistant_prefix is None
    assert "User:" not in rendered_prompt
    assert "Assistant:" not in rendered_prompt


def test_get_prompt_for_questionnaire_type_generation_returns_exact_base_model_prompt(
    mock_questionnaires,
):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    ).set_base_model_prompt_template(
        user_prefix="Instruction:",
        assistant_prefix="Response:",
    )

    system_message, rendered_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM,
        item_position=0,
        inference_mode="completion",
    )

    assert system_message is None
    assert rendered_prompt.startswith("SYS\nInstruction:")
    assert "How do you feel about Red?" in rendered_prompt
    assert rendered_prompt.endswith("Response:")


def test_base_model_prompt_template_allows_none_prefixes(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    ).set_base_model_prompt_template(
        user_prefix=None,
        assistant_prefix=None,
        system_prefix=None,
    )

    _, rendered_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM,
        item_position=0,
        inference_mode="completion",
    )

    assert rendered_prompt.startswith("SYS\nASK")
    assert "User:" not in rendered_prompt
    assert "Assistant:" not in rendered_prompt
    assert not rendered_prompt.endswith("None")


def test_base_model_prompt_template_setter_accepts_template_object(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    template = BaseModelPromptTemplate(
        user_prefix="Instruction:",
        assistant_prefix="Response:",
    )

    returned = prompt.set_base_model_prompt_template(template)

    assert returned is prompt
    assert prompt.base_model_prompt_template is template


class _FakeTiktokenEncoding:
    def encode(self, text, disallowed_special=()):
        del disallowed_special
        return list(text)


def _install_fake_tiktoken(monkeypatch):
    module = types.ModuleType("tiktoken")
    calls = []

    def encoding_for_model(model_id):
        calls.append(model_id)
        return _FakeTiktokenEncoding()

    module.encoding_for_model = encoding_for_model
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    return calls


def _install_fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    calls = []

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(model_id)
            return FakeTokenizer()

    module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def test_calculate_input_token_estimate_uses_tiktoken_backend(monkeypatch):
    calls = _install_fake_tiktoken(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q"}]),
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )

    estimate = prompt.calculate_input_token_estimate(
        model_id="gpt-test",
        tokenizer_backend="tiktoken",
    )

    _, user_prompt = prompt.get_prompt_for_questionnaire_type()
    assert calls == ["gpt-test"]
    assert estimate == len("SYS") + len(user_prompt) + 2 * 3 + 3


def test_calculate_input_token_estimate_uses_lazy_transformers_backend(monkeypatch):
    calls = _install_fake_transformers(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q"}]),
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )

    estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
    )

    _, user_prompt = prompt.get_prompt_for_questionnaire_type()
    assert calls == ["hf-test"]
    assert estimate == len("SYS") + len(user_prompt)


def test_calculate_input_token_estimate_transformers_backend_requires_auto_tokenizer(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q"}])
    )

    with pytest.raises(ImportError, match="optional 'transformers' package"):
        prompt.calculate_input_token_estimate(
            model_id="hf-test",
            tokenizer_backend="transformers",
        )


def test_calculate_input_token_estimate_single_item_returns_largest_item(monkeypatch):
    _install_fake_transformers(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame(
            [
                {"questionnaire_item_id": 1, "question_content": "Q"},
                {"questionnaire_item_id": 2, "question_content": "A much longer question"},
            ]
        ),
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )

    estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
    )

    expected = max(
        len(system_prompt or "") + len(user_prompt)
        for system_prompt, user_prompt in (
            prompt.get_prompt_for_questionnaire_type(item_position=0),
            prompt.get_prompt_for_questionnaire_type(item_position=1),
        )
    )
    assert estimate == expected


def test_calculate_input_token_estimate_battery_renders_once(monkeypatch):
    _install_fake_transformers(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame(
            [
                {"questionnaire_item_id": 1, "question_content": "Q1"},
                {"questionnaire_item_id": 2, "question_content": "Q2"},
            ]
        ),
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )
    original_get_prompt = prompt.get_prompt_for_questionnaire_type
    calls = []

    def spy_get_prompt_for_questionnaire_type(*args, **kwargs):
        calls.append(kwargs.get("item_position"))
        return original_get_prompt(*args, **kwargs)

    monkeypatch.setattr(
        prompt,
        "get_prompt_for_questionnaire_type",
        spy_get_prompt_for_questionnaire_type,
    )

    estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator=" || ",
    )

    system_prompt, user_prompt = original_get_prompt(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_position=0,
        item_separator=" || ",
    )
    assert calls == [0]
    assert estimate == len(system_prompt or "") + len(user_prompt)


def test_calculate_input_token_estimate_sequential_uses_default_response_estimate(monkeypatch):
    _install_fake_transformers(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame(
            [
                {"questionnaire_item_id": 1, "question_content": "Q1"},
                {"questionnaire_item_id": 2, "question_content": "Q2"},
            ]
        ),
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )

    estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
        questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
    )

    first_system_prompt, first_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
        item_position=0,
    )
    _, second_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
        item_position=1,
    )
    assert estimate == len(first_system_prompt or "") + len(first_prompt) + len(second_prompt) + 100


def test_calculate_input_token_estimate_sequential_allows_custom_response_estimate(
    monkeypatch,
):
    _install_fake_transformers(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame(
            [
                {"questionnaire_item_id": 1, "question_content": "Q1"},
                {"questionnaire_item_id": 2, "question_content": "Q2"},
            ]
        ),
        system_prompt="SYS",
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )

    default_estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
        questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
    )
    custom_estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
        questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
        previous_response_token_estimate=50,
    )

    assert default_estimate - custom_estimate == 50


def test_calculate_input_token_estimate_counts_none_system_prompt_as_zero(monkeypatch):
    _install_fake_transformers(monkeypatch)
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q"}]),
        system_prompt=None,
        prompt="ASK {{QUESTION_PLACEHOLDER}}",
    )

    estimate = prompt.calculate_input_token_estimate(
        model_id="hf-test",
        tokenizer_backend="transformers",
        inference_mode="completion",
    )

    _, rendered_prompt = prompt.get_prompt_for_questionnaire_type(inference_mode="completion")
    assert estimate == len(rendered_prompt)
