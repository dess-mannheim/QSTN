"""Additional branch and validation tests for `LLMPrompt` and likert option generation."""

import pandas as pd
import pytest

from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.inference.response_generation import JSONSingleResponseGenerationMethod
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
        questionnaire_source=pd.DataFrame(
            [{"questionnaire_item_id": 1, "question_content": "Q1"}]
        )
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
    assert prompt[0].item_id == 9
    assert prompt[0].question_content is None
    assert prompt[0].question_stem is None


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
    assert p1[0].answer_options is opt_1
    assert p1[1].answer_options is opt_2

    p2 = LLMPrompt(questionnaire_source=df).prepare_prompt(
        question_stem=["S1", "S2"],
        answer_options=shared_options,
    )
    assert p2[0].question_stem == "S1"
    assert p2[1].question_stem == "S2"
    assert p2[0].answer_options is shared_options

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
    assert p3[0].item_id == 2


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

    assert prompt[-1].item_id == 4
    assert "=== Demo ===" in rendered
    assert "=== SYSTEM_PROMPT ===" in rendered
    assert "=== USER_PROMPT_WITH_ALL_QUESTIONS ===" in rendered


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"n": 2, "answer_texts": ["A", "B"], "only_from_to_scale": True, "idx_type": "char_upper"},
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
