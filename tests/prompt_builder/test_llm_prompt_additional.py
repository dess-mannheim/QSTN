"""Additional branch and validation tests for `LLMPrompt` and likert option generation."""

import pandas as pd
import pytest

from qstn.inference.response_generation import (
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
)
from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.utilities import placeholder
from qstn.utilities.constants import QuestionnairePresentation
from qstn.utilities.survey_objects import QuestionnaireItem


def test_check_valid_questionnaire_and_duplicate_and_get_questions():
    prompt = LLMPrompt()

    assert prompt._check_valid_questionnaire(None) is False
    assert prompt._check_valid_questionnaire("") is False
    with pytest.warns(UserWarning, match="provided Dataframe is empty"):
        assert prompt._check_valid_questionnaire(pd.DataFrame()) is False

    loaded = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    )
    clone = loaded.duplicate()

    assert clone is not loaded
    assert clone.get_questions() is not loaded.get_questions()
    assert clone.get_questions()[0].item_id == loaded.get_questions()[0].item_id


def test_load_questionnaire_format_from_csv_without_optional_columns(tmp_path):
    csv_path = tmp_path / "questionnaire.csv"
    pd.DataFrame([{"questionnaire_item_id": 9}]).to_csv(csv_path, index=False)

    prompt = LLMPrompt()
    prompt.load_questionnaire_format(str(csv_path))

    assert len(prompt) == 1
    assert prompt.get_question(0).item_id == 9
    assert prompt.get_question(0).question_content is None
    assert prompt.get_question(0).question_stem is None


def test_prepare_prompt_other_combinations_and_randomized_order(monkeypatch):
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    shared_options = generate_likert_options(n=2, answer_texts=["No", "Yes"])
    opt_1 = generate_likert_options(n=2, answer_texts=["A", "B"])
    opt_2 = generate_likert_options(n=2, answer_texts=["C", "D"])

    p1 = LLMPrompt(questionnaire_source=df).prepare_prompt(
        question_stem="Stem",
        answer_options={1: opt_1, 2: opt_2},
    )
    assert p1.get_question(0).answer_options is opt_1
    assert p1.get_question(1).answer_options is opt_2

    p2 = LLMPrompt(questionnaire_source=df).prepare_prompt(
        question_stem=["S1", "S2"],
        answer_options=shared_options,
    )
    assert p2.get_question(0).question_stem == "S1"
    assert p2.get_question(1).question_stem == "S2"
    assert p2.get_question(0).answer_options is shared_options

    shuffled = {"called": False}

    def fake_shuffle(seq):
        shuffled["called"] = True
        seq.reverse()

    monkeypatch.setattr("qstn.prompt_builder.random.shuffle", fake_shuffle)
    p3 = LLMPrompt(questionnaire_source=df).prepare_prompt(
        question_stem=["S1", "S2"],
        answer_options={1: opt_1, 2: opt_2},
        randomized_item_order=True,
    )
    assert shuffled["called"] is True
    assert p3.get_question(0).item_id == 2


def test_get_prompt_for_questionnaire_type_error_and_battery_auto_instructions():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Red?"},
            {"questionnaire_item_id": 2, "question_content": "Blue?"},
        ]
    )
    prompt = LLMPrompt(questionnaire_source=df)

    with pytest.raises(ValueError, match="item_order_id"):
        prompt.get_prompt_for_questionnaire_type(item_position=99)

    options = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        response_generation_method=JSONSingleResponseGenerationMethod(),
    )
    prompt.system_prompt = f"SYS {placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"
    prompt.prompt = (
        f"ASK {placeholder.PROMPT_QUESTIONS}\n"
        f"{placeholder.PROMPT_OPTIONS}\n"
        f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"
    )
    prompt.prepare_prompt(
        question_stem=f"{placeholder.QUESTION_CONTENT}\n{placeholder.PROMPT_OPTIONS}",
        answer_options={1: options},
    )

    system_prompt, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_id=1,
        item_separator=" || ",
    )

    assert " || " in user_prompt
    assert "JSON format" in system_prompt
    assert "Red?" in user_prompt
    assert "Blue?" in user_prompt


def test_battery_mixed_scales_use_per_question_distribution_schema():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options_5 = generate_likert_options(
        n=5,
        answer_texts=["SEHR GUT", "GUT", "TEILS/TEILS", "SCHLECHT", "SEHR SCHLECHT"],
        response_generation_method=JSONVerbalizedDistribution(),
    )
    options_2 = generate_likert_options(
        n=2,
        answer_texts=["BIN DERS.MEINUNG", "BIN ANDERER MEINUNG"],
        response_generation_method=JSONVerbalizedDistribution(),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options_5, 2: options_2})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert '"Q1 | q1_o1": <probability for: 1: SEHR GUT>' in system_prompt
    assert "<probability for: 5: SEHR SCHLECHT>" in system_prompt
    assert '"Q1... | q1_o5": <probability for: 5: SEHR SCHLECHT>' in system_prompt
    assert '"Q2 | q2_o1": <probability for: 1: BIN DERS.MEINUNG>' in system_prompt
    assert '"Q2... | q2_o2": <probability for: 2: BIN ANDERER MEINUNG>' in system_prompt


def test_battery_verbalized_distribution_supports_custom_field_and_explanation_templates():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )

    options_q1 = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        response_generation_method=JSONVerbalizedDistribution(
            json_field_template="Question: {question} - Option: {option}",
            json_explanation_template="probability for option {option}",
        ),
    )
    options_q2 = generate_likert_options(
        n=2,
        answer_texts=["A", "B"],
        response_generation_method=JSONVerbalizedDistribution(
            json_field_template="Question: {question} - Option: {option}",
            json_explanation_template="probability for option {option}",
        ),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options_q1, 2: options_q2})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert '"Question: Q1 - Option: 1: No": <probability for option 1: No>' in system_prompt
    assert '"Question: Q2... - Option: 2: B": <probability for option 2: B>' in system_prompt


def test_custom_template_with_question_placeholder_does_not_duplicate_question_suffix():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        response_generation_method=JSONVerbalizedDistribution(
            json_field_template="Question: {question} | Option: {option}_{question}",
            json_explanation_template="probability for: {option}",
        ),
    )
    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options, 2: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert "_Q1_Q1" not in system_prompt
    assert "_Q2_Q2" not in system_prompt


def test_default_verbalized_distribution_shortens_question_in_battery_fields():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Sind Sie bei der folgenden Aussage derselben Meinung?",
            },
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        response_generation_method=JSONVerbalizedDistribution(),
    )
    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options, 2: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert '"Sind Sie bei der folgenden Aussage derselben Meinung? | q1_o1"' in system_prompt
    assert '"Sind Sie... | q1_o2"' in system_prompt


def test_custom_question_only_template_requires_unique_placeholders_in_battery():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Wie groß ist Ihr Vertrauen in die politischen Parteien?",
            },
            {
                "questionnaire_item_id": 2,
                "question_content": "Wie groß ist Ihr Vertrauen in die Bundesregierung?",
            },
        ]
    )
    options = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        response_generation_method=JSONVerbalizedDistribution(json_field_template="{question}"),
    )
    prompt = LLMPrompt(questionnaire_source=df).prepare_prompt(
        answer_options={1: options, 2: options}
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        prompt.get_prompt_for_questionnaire_type(
            questionnaire_type=QuestionnairePresentation.BATTERY
        )


def test_get_prompt_for_questionnaire_type_allows_none_system_prompt():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Red?"}])
    prompt = LLMPrompt(
        questionnaire_source=df, system_prompt=None, prompt="ASK {{QUESTION_PLACEHOLDER}}"
    )

    system_prompt, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM
    )

    assert system_prompt is None
    assert "Red?" in user_prompt


def test_get_prompt_for_questionnaire_type_keeps_empty_system_prompt():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Red?"}])
    prompt = LLMPrompt(
        questionnaire_source=df, system_prompt="", prompt="ASK {{QUESTION_PLACEHOLDER}}"
    )

    system_prompt, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM
    )

    assert system_prompt == ""
    assert "Red?" in user_prompt


def test_generate_question_prompt_without_placeholder_and_with_none_options_template():
    options = generate_likert_options(n=2, answer_texts=["Low", "High"])
    options.list_prompt_template = None

    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame(
            [{"questionnaire_item_id": 1, "question_content": "Color?"}]
        )
    )
    item = QuestionnaireItem(
        item_id=1,
        question_content="Color?",
        question_stem="State preference for",
        answer_options=options,
    )

    generated = prompt.generate_question_prompt(item)
    assert generated == "State preference for Color?"


def test_str_and_insert_questions_default_position():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    prompt = LLMPrompt(questionnaire_source=df, questionnaire_name="Demo")

    prompt.insert_questions(
        [
            QuestionnaireItem(item_id=3, question_content="Q3"),
            QuestionnaireItem(item_id=4, question_content="Q4"),
        ]
    )
    rendered = str(prompt)

    assert prompt.get_question(-1).item_id == 4
    assert "=== Demo ===" in rendered
    assert "=== SYSTEM_PROMPT ===" in rendered
    assert "=== USER_PROMPT_WITH_ALL_QUESTIONS ===" in rendered


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "n": 2,
                "answer_texts": ["A", "B"],
                "only_from_to_scale": True,
                "idx_type": "char_upper",
            },
            "integer scale index",
        ),
        ({"n": 2, "answer_texts": ["A"]}, "same length"),
        ({"n": 4, "answer_texts": ["A", "B", "C", "D"], "even_order": True}, "should be odd"),
        ({"n": 3, "answer_texts": ["A", "B", "C"], "add_middle_category": True}, "should be even"),
        ({"n": 1, "answer_texts": ["A"], "random_order": True}, "at least two"),
        ({"n": 1, "answer_texts": ["A"], "reversed_order": True}, "at least two"),
    ],
)
def test_generate_likert_options_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_likert_options(**kwargs)


def test_generate_likert_options_covers_index_and_order_variants():
    evened = generate_likert_options(
        n=5,
        answer_texts=["1", "2", "3", "4", "5"],
        even_order=True,
    )
    assert "3" not in evened.answer_texts.answer_texts

    with_middle = generate_likert_options(
        n=4,
        answer_texts=["1", "2", "4", "5"],
        add_middle_category=True,
        str_middle_cat="MID",
    )
    assert "MID" in with_middle.answer_texts.answer_texts

    no_index = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        idx_type="no_index",
    )
    assert no_index.answer_texts.indices is None

    with_refusal = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        add_refusal=True,
        refusal_code="-77",
        start_idx=10,
    )
    assert with_refusal.answer_texts.indices[-1] == "-77"
    assert with_refusal.answer_texts.indices[0] == "10"

    char_lower = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        idx_type="char_lower",
        start_idx=1,
    )
    char_upper = generate_likert_options(
        n=2,
        answer_texts=["No", "Yes"],
        idx_type="char_upper",
        start_idx=1,
    )

    assert char_lower.answer_texts.indices == ["b", "c"]
    assert char_upper.answer_texts.indices == ["B", "C"]
