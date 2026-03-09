"""Tests for the top-level persona creator module."""

import pandas as pd
import pytest

from qstn import survey_manager
from qstn.parser.llm_answer_parser import raw_responses
from qstn.persona_creator import LLMConfig, PersonaBuilder
from qstn.prompt_builder import LLMPrompt


def test_render_automatic_supports_order_templates_and_missing_policy():
    builder = PersonaBuilder()
    attributes = {"age": 30, "city": "Berlin", "income": None}

    text_skip = builder.render_automatic(
        attributes,
        attribute_order=["city", "income", "age"],
        attribute_templates={"city": "City={value}", "age": "Age={value}"},
        missing_value_policy="skip",
    )
    assert text_skip == "City=Berlin\nAge=30"

    text_placeholder = builder.render_automatic(
        attributes,
        attribute_order=["city", "income", "age"],
        missing_value_policy="placeholder",
        missing_value_placeholder="n/a",
    )
    assert text_placeholder == "city: Berlin\nincome: n/a\nage: 30"


def test_render_automatic_normalizes_integer_like_floats_and_booleans():
    builder = PersonaBuilder()
    text = builder.render_automatic({"age": 19.0, "is_student": True})
    assert "age: 19" in text
    assert "is_student: Yes" in text


def test_render_automatic_supports_custom_bool_text_and_float_cast_override():
    builder = PersonaBuilder()
    text = builder.render_automatic(
        {"age": 19.0, "is_student": False},
        true_text="JA",
        false_text="NEIN",
        cast_integer_like_floats=False,
    )
    assert "age: 19.0" in text
    assert "is_student: NEIN" in text


def test_attribute_aliases_support_renaming_without_dataframe_preprocessing():
    builder = PersonaBuilder()
    text = builder.render_automatic(
        {"years_experience": 19.0},
        attribute_aliases={"years_experience": "years_of_experience"},
    )
    assert text == "years_of_experience: 19"


def test_render_interview_text_supports_default_and_custom_questions():
    builder = PersonaBuilder(default_interview_question_map={"age": "How old are you?"})
    text = builder.render_interview_text(
        {"age": 30, "city": "Berlin"},
        attribute_order=["age", "city"],
        question_template="What about your {attribute}?",
    )

    assert "Q: How old are you?" in text
    assert "A: 30" in text
    assert "Q: What about your city?" in text
    assert "A: Berlin" in text


def test_build_persona_dataframe_supports_method_specific_kwargs_bundles():
    builder = PersonaBuilder()
    personas = pd.DataFrame([{"years_experience": 19.0, "is_manager": True, "city": "Berlin"}])
    result = builder.build_persona_dataframe(
        personas,
        attribute_columns=["years_experience", "is_manager", "city"],
        automatic_kwargs={
            "attribute_aliases": {"years_experience": "years_of_experience"},
            "attribute_templates": {
                "years_of_experience": "Experience: {value}",
                "is_manager": "Manager: {value}",
            },
        },
        interview_kwargs={
            "attribute_aliases": {"years_experience": "years_of_experience"},
            "question_map": {"years_of_experience": "How many years of experience do you have?"},
        },
        human_prompt_kwargs={
            "attribute_aliases": {"years_experience": "years_of_experience"},
        },
    )
    assert "Experience: 19" in result["persona_automatic"].iloc[0]
    assert "Manager: Yes" in result["persona_automatic"].iloc[0]
    assert "How many years of experience do you have?" in result["persona_interview"].iloc[0]
    assert "- years_of_experience: 19" in result["persona_human_prompt"].iloc[0]


def test_build_human_prompt_and_generate_human_description(monkeypatch):
    builder = PersonaBuilder()
    attrs = {"age": 30, "city": "Berlin"}

    prompt = builder.build_human_prompt(attrs, attribute_order=["city", "age"])
    assert "- city: Berlin" in prompt
    assert "- age: 30" in prompt

    captured = {}

    def fake_batch_generation(**kwargs):
        captured["prompts"] = kwargs["prompts"]
        captured["system_messages"] = kwargs["system_messages"]
        return (["Generated persona"], [None], [None])

    monkeypatch.setattr(
        "qstn.persona_creator.persona_creator.batch_generation", fake_batch_generation
    )
    output = builder.generate_human_description(
        model=object(),
        attributes=attrs,
        attribute_order=["city", "age"],
        llm_config=LLMConfig(print_progress=False),
    )

    assert output == "Generated persona"
    assert len(captured["prompts"]) == 1
    assert "- city: Berlin" in captured["prompts"][0]


def test_build_and_generate_interview_from_automatic(monkeypatch):
    builder = PersonaBuilder()
    automatic_text = "age: 30\ncity: Berlin"
    prompt = builder.build_interview_from_automatic_prompt(automatic_text)
    assert automatic_text in prompt
    assert "Keep each answer value exactly as provided" in prompt

    captured = {}

    def fake_batch_generation(**kwargs):
        captured["prompts"] = kwargs["prompts"]
        return (["Q: How old are you?\nA: 30"], [None], [None])

    monkeypatch.setattr(
        "qstn.persona_creator.persona_creator.batch_generation", fake_batch_generation
    )
    output = builder.generate_interview_from_automatic(
        model=object(),
        automatic_persona=automatic_text,
        llm_config=LLMConfig(print_progress=False),
    )
    assert output.startswith("Q:")
    assert automatic_text in captured["prompts"][0]


def test_build_persona_dataframe_creates_mode_columns_and_default_system_prompt():
    builder = PersonaBuilder()
    personas = pd.DataFrame(
        [
            {"id": 1, "age": 30, "city": "Berlin"},
            {"id": 2, "age": 40, "city": "Hamburg"},
        ]
    )

    result = builder.build_persona_dataframe(
        personas,
        attribute_columns=["age", "city"],
        attribute_order=["city", "age"],
    )

    assert "persona_automatic" in result.columns
    assert "persona_interview" in result.columns
    assert "persona_human_prompt" in result.columns
    assert "persona_human_llm" not in result.columns
    assert "system_prompt" in result.columns
    assert result["system_prompt"].tolist() == result["persona_automatic"].tolist()


def test_build_persona_dataframe_optional_human_llm_mode(monkeypatch):
    builder = PersonaBuilder()
    personas = pd.DataFrame([{"age": 30, "city": "Berlin"}])

    def fake_batch_generation(**kwargs):
        return (["Human profile"], [None], [None])

    monkeypatch.setattr(
        "qstn.persona_creator.persona_creator.batch_generation", fake_batch_generation
    )
    result = builder.build_persona_dataframe(
        personas,
        attribute_columns=["age", "city"],
        include_human_llm=True,
        model=object(),
        default_system_prompt_mode="human_llm",
        llm_config=LLMConfig(print_progress=False),
    )
    assert result["persona_human_llm"].tolist() == ["Human profile"]
    assert result["system_prompt"].tolist() == ["Human profile"]

    with pytest.raises(ValueError, match="`model` must be provided"):
        builder.build_persona_dataframe(personas, include_human_llm=True, model=None)


def test_build_persona_dataframe_optional_interview_llm_mode(monkeypatch):
    builder = PersonaBuilder()
    personas = pd.DataFrame([{"age": 30, "city": "Berlin"}])

    captured = {}

    def fake_batch_generation(**kwargs):
        captured["prompts"] = kwargs["prompts"]
        return (["Q: Age?\nA: 30"], [None], [None])

    monkeypatch.setattr(
        "qstn.persona_creator.persona_creator.batch_generation", fake_batch_generation
    )
    result = builder.build_persona_dataframe(
        personas,
        attribute_columns=["age", "city"],
        include_interview_llm=True,
        model=object(),
        default_system_prompt_mode="interview_llm",
        llm_config=LLMConfig(print_progress=False),
    )
    assert result["persona_interview_llm"].tolist() == ["Q: Age?\nA: 30"]
    assert result["system_prompt"].tolist() == ["Q: Age?\nA: 30"]
    assert "age: 30" in captured["prompts"][0]

    with pytest.raises(ValueError, match="`model` must be provided"):
        builder.build_persona_dataframe(personas, include_interview_llm=True, model=None)


def test_build_persona_dataframe_reuses_interview_questions_once(monkeypatch):
    builder = PersonaBuilder()
    personas = pd.DataFrame(
        [
            {"age": 30, "city": "Berlin"},
            {"age": 40, "city": "Hamburg"},
        ]
    )
    counter = {"calls": 0}
    original = builder._resolve_interview_question_templates

    def wrapped(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(builder, "_resolve_interview_question_templates", wrapped)
    result = builder.build_persona_dataframe(
        personas,
        attribute_columns=["age", "city"],
        reuse_interview_questions=True,
    )

    assert counter["calls"] == 1
    assert "Q: What is your age?" in result["persona_interview"].iloc[0]
    assert "A: 30" in result["persona_interview"].iloc[0]
    assert "A: 40" in result["persona_interview"].iloc[1]


def test_reused_interview_questions_still_allow_value_in_question_template():
    builder = PersonaBuilder()
    personas = pd.DataFrame(
        [
            {"age": 30, "city": "Berlin"},
            {"age": 40, "city": "Hamburg"},
        ]
    )
    result = builder.build_persona_dataframe(
        personas,
        attribute_columns=["age", "city"],
        interview_kwargs={"question_template": "Is your {attribute} {value}?"},
        reuse_interview_questions=True,
    )
    assert "Q: Is your age 30?" in result["persona_interview"].iloc[0]
    assert "Q: Is your age 40?" in result["persona_interview"].iloc[1]


def test_apply_interview_as_prefilled_turns_prepends_and_keeps_order():
    questionnaire = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    prompt = LLMPrompt(questionnaire_source=questionnaire)
    builder = PersonaBuilder()

    transformed = builder.apply_interview_as_prefilled_turns(
        prompt,
        attributes={"age": 30, "city": "Berlin"},
        attribute_order=["age", "city"],
    )

    assert len(prompt) == 2
    assert len(transformed) == 4

    transformed_questions = transformed.get_questions()
    assert str(transformed_questions[0].item_id).startswith("__persona_interview__")
    assert transformed_questions[0].prefilled_response == "30"
    assert transformed_questions[1].prefilled_response == "Berlin"
    assert transformed_questions[2].item_id == 1
    assert transformed_questions[3].item_id == 2


def test_prefilled_interview_turns_seed_sequential_flow(monkeypatch):
    questionnaire = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Final Q?"}])
    prompt = LLMPrompt(questionnaire_source=questionnaire, questionnaire_name="seeded")
    builder = PersonaBuilder()
    transformed = builder.apply_interview_as_prefilled_turns(
        prompt,
        attributes={"age": 30, "city": "Berlin"},
        attribute_order=["age", "city"],
    )

    recorded_calls = []

    def fake_batch_turn_by_turn_generation(**kwargs):
        recorded_calls.append(
            {
                "prompts": [list(conv) for conv in kwargs["prompts"]],
                "assistant_messages": [list(conv) for conv in kwargs["assistant_messages"]],
            }
        )
        return (["GENERATED_FINAL"], [None], [None])

    monkeypatch.setattr(
        survey_manager, "batch_turn_by_turn_generation", fake_batch_turn_by_turn_generation
    )

    results = survey_manager.conduct_survey_sequential(
        model=object(),
        llm_prompts=[transformed],
        print_progress=False,
    )
    parsed = raw_responses(results)[transformed]

    assert len(recorded_calls) == 1
    assert len(recorded_calls[0]["prompts"][0]) == 3
    assert recorded_calls[0]["assistant_messages"][0] == ["30", "Berlin"]
    assert (
        parsed.loc[parsed["questionnaire_item_id"] == 1, "llm_response"].item() == "GENERATED_FINAL"
    )
