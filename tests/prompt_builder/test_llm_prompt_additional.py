"""Additional branch and validation tests for `LLMPrompt` and likert option generation."""

import pandas as pd
import pytest

from qstn.inference.response_generation import (
    ChoiceResponseGenerationMethod,
    JSONItem,
    JSONObject,
    JSONResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
    LogprobResponseGenerationMethod,
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


def test_load_questionnaire_format_accepts_dataframe_python_lists():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": "risk",
                "question_content": "Is this a risk or chance?",
                "question_stem": "Please answer:",
                "prefilled_response": "prefilled",
                "answer_texts": [
                    "RISIKO UEBERWIEGT",
                    "EHER RISIKO",
                    "WEDER NOCH",
                    "EHER CHANCE",
                    "CHANCE UEBERWIEGT",
                ],
                "answer_codes": ["1", "2", "3", "4", "5"],
            }
        ]
    )

    prompt = LLMPrompt(questionnaire_source=df)
    question = prompt.get_question(0)

    assert question.item_id == "risk"
    assert question.question_stem == "Please answer:"
    assert question.prefilled_response == "prefilled"
    assert question.answer_options.answer_texts.full_answers == [
        "1: RISIKO UEBERWIEGT",
        "2: EHER RISIKO",
        "3: WEDER NOCH",
        "4: EHER CHANCE",
        "5: CHANCE UEBERWIEGT",
    ]


def test_load_questionnaire_format_accepts_csv_python_list_strings(tmp_path):
    csv_path = tmp_path / "questionnaire.csv"
    pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": '["No", "Yes"]',
                "answer_codes": '["0", "1"]',
                "response_generation_method": "json_single",
                "output_index_only": "True",
            }
        ]
    ).to_csv(csv_path, index=False)

    prompt = LLMPrompt(questionnaire_source=str(csv_path))
    options = prompt.get_question(0).answer_options

    assert options.answer_texts.full_answers == ["0: No", "1: Yes"]
    assert isinstance(options.response_generation_method, JSONResponseGenerationMethod)
    answer_item = options.response_generation_method.json_object.children[0]
    assert isinstance(answer_item, JSONItem)
    assert answer_item.constraints.enum == ["0", "1"]


def test_load_questionnaire_format_generates_likert_with_inferred_n():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["No", "Maybe", "Yes"],
                "likert_start_idx": 0,
            }
        ]
    )

    prompt = LLMPrompt(questionnaire_source=df)
    options = prompt.get_question(0).answer_options

    assert options.answer_texts.full_answers == ["0: No", "1: Maybe", "2: Yes"]


def test_load_questionnaire_format_generates_likert_with_explicit_n_and_start_idx():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["Low", "High"],
                "likert_n": 4,
                "likert_only_from_to_scale": True,
                "likert_start_idx": 2,
                "scale_prompt_template": "Scale: {start} through {end}",
            }
        ]
    )

    prompt = LLMPrompt(questionnaire_source=df)
    options = prompt.get_question(0).answer_options

    assert options.create_options_str() == "Scale: 2: Low through 5: High"
    assert options.answer_texts.full_answers == ["2: Low", "3", "4", "5: High"]


def test_load_questionnaire_format_from_to_scale_requires_explicit_n():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["Low", "High"],
                "likert_only_from_to_scale": True,
            }
        ]
    )

    with pytest.raises(ValueError, match="likert_n"):
        LLMPrompt(questionnaire_source=df)


@pytest.mark.parametrize(
    ("preset", "expected_type"),
    [
        ("choice", ChoiceResponseGenerationMethod),
        ("logprob", LogprobResponseGenerationMethod),
        ("json_single", JSONResponseGenerationMethod),
        ("json_reasoning", JSONResponseGenerationMethod),
        ("json_distribution", JSONVerbalizedDistribution),
    ],
)
def test_load_questionnaire_format_response_generation_presets(preset, expected_type):
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["No", "Yes"],
                "response_generation_method": preset,
                "output_index_only": True,
            }
        ]
    )

    prompt = LLMPrompt(questionnaire_source=df)
    method = prompt.get_question(0).answer_options.response_generation_method

    assert isinstance(method, expected_type)
    assert method.output_index_only is True


@pytest.mark.parametrize("preset", ["choice", "logprob", "json_single"])
def test_load_questionnaire_format_can_disable_answer_option_constraints(preset):
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame(
            [
                {
                    "questionnaire_item_id": 1,
                    "question_content": "Q1",
                    "answer_texts": ["No", "Yes"],
                    "answer_codes": ["0", "1"],
                    "response_generation_method": preset,
                    "constrain_answer_options": False,
                }
            ]
        )
    )

    method = prompt.get_question(0).answer_options.response_generation_method

    assert method.constrain_answer_options is False


def test_load_questionnaire_format_rejects_invalid_python_list_string():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": "No|Yes",
            }
        ]
    )

    with pytest.raises(ValueError, match="Python list literal"):
        LLMPrompt(questionnaire_source=df)


def test_load_questionnaire_format_rejects_non_python_bool_literal():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["No", "Yes"],
                "likert_random_order": "true",
            }
        ]
    )

    with pytest.raises(ValueError, match="True or False"):
        LLMPrompt(questionnaire_source=df)


def test_load_questionnaire_format_rejects_invalid_likert_idx_type():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["No", "Yes"],
                "likert_idx_type": "roman",
            }
        ]
    )

    with pytest.raises(ValueError, match="likert_idx_type"):
        LLMPrompt(questionnaire_source=df)


def test_load_questionnaire_format_rejects_mismatched_answer_texts_and_codes():
    df = pd.DataFrame(
        [
            {
                "questionnaire_item_id": 1,
                "question_content": "Q1",
                "answer_texts": ["No", "Yes"],
                "answer_codes": ["1"],
            }
        ]
    )

    with pytest.raises(ValueError, match="same length"):
        LLMPrompt(questionnaire_source=df)


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


@pytest.mark.parametrize(
    "prompt_template, expected",
    [
        (
            (
                f"QUESTIONS\n{placeholder.PROMPT_QUESTIONS}\n"
                f"OPTIONS\n{placeholder.PROMPT_OPTIONS}"
            ),
            (
                "QUESTIONS\nQ1 || Q2\nOPTIONS\n"
                "Options are: 1: A1, 2: A2 || Options are: 1: B1, 2: B2"
            ),
        ),
        (
            (
                f"OPTIONS\n{placeholder.PROMPT_OPTIONS}\n"
                f"QUESTIONS\n{placeholder.PROMPT_QUESTIONS}"
            ),
            (
                "OPTIONS\nOptions are: 1: A1, 2: A2 || "
                "Options are: 1: B1, 2: B2\nQUESTIONS\nQ1 || Q2"
            ),
        ),
    ],
)
def test_battery_main_prompt_aggregates_all_options_in_placeholder_order(prompt_template, expected):
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options_1 = generate_likert_options(n=2, answer_texts=["A1", "A2"])
    options_2 = generate_likert_options(n=2, answer_texts=["B1", "B2"])
    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=None,
        prompt=prompt_template,
    ).prepare_prompt(answer_options={1: options_1, 2: options_2})

    system_prompt, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator=" || ",
    )
    _, completion_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator=" || ",
        inference_mode="completion",
    )

    assert system_prompt is None
    assert user_prompt == expected
    assert completion_prompt == expected
    rendered = str(prompt)
    assert rendered.count("Options are:") == 2
    assert rendered.index("1: A1") < rendered.index("1: B1")


def test_battery_question_stem_renders_each_questions_own_options():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options_1 = generate_likert_options(n=2, answer_texts=["A1", "A2"])
    options_2 = generate_likert_options(n=2, answer_texts=["B1", "B2"])
    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=None,
        prompt=placeholder.PROMPT_QUESTIONS,
    ).prepare_prompt(
        question_stem=f"{placeholder.QUESTION_CONTENT}\n{placeholder.PROMPT_OPTIONS}",
        answer_options={1: options_1, 2: options_2},
    )

    _, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator="\n--\n",
    )

    assert user_prompt == ("Q1\nOptions are: 1: A1, 2: A2\n--\n" "Q2\nOptions are: 1: B1, 2: B2")


def test_battery_populates_main_and_question_stem_option_placeholders():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options = generate_likert_options(n=2, answer_texts=["A1", "A2"])
    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=None,
        prompt=(f"{placeholder.PROMPT_QUESTIONS}\nALL OPTIONS\n" f"{placeholder.PROMPT_OPTIONS}"),
    ).prepare_prompt(
        question_stem=f"{placeholder.QUESTION_CONTENT}: {placeholder.PROMPT_OPTIONS}",
        answer_options={1: options},
    )

    _, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator=" || ",
    )

    assert user_prompt == (
        "Q1: Options are: 1: A1, 2: A2 || Q2: \n" "ALL OPTIONS\nOptions are: 1: A1, 2: A2"
    )


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

    assert '"Q1": {' in system_prompt
    assert '"1: SEHR GUT": "probability for: 1: SEHR GUT"' in system_prompt
    assert '"5: SEHR SCHLECHT": "probability for: 5: SEHR SCHLECHT"' in system_prompt
    assert '"Q2": {' in system_prompt
    assert '"1: BIN DERS.MEINUNG": "probability for: 1: BIN DERS.MEINUNG"' in system_prompt
    assert '"2: BIN ANDERER MEINUNG": "probability for: 2: BIN ANDERER MEINUNG"' in system_prompt


def test_json_single_answer_reuses_scale_prompt_format_in_explanations():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    options = generate_likert_options(
        n=7,
        answer_texts=["UNWICHTIG", "SEHR WICHTIG"],
        only_from_to_scale=True,
        scale_prompt_template="Antwortmöglichkeiten: {start} bis {end}",
        response_generation_method=JSONSingleResponseGenerationMethod(),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert '"Q1": {' in system_prompt
    assert '"answer": "Antwortmöglichkeiten: 1: UNWICHTIG bis 7: SEHR WICHTIG"' in system_prompt


def test_verbalized_distribution_uses_plain_middle_scale_indices():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    options = generate_likert_options(
        n=7,
        answer_texts=["SEHR FALSCH", "SEHR RICHTIG"],
        only_from_to_scale=True,
        response_generation_method=JSONVerbalizedDistribution(),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert '"1: SEHR FALSCH": "probability for: 1: SEHR FALSCH"' in system_prompt
    assert '"2": "probability for: 2"' in system_prompt
    assert '"4": "probability for: 4"' in system_prompt
    assert '"7: SEHR RICHTIG": "probability for: 7: SEHR RICHTIG"' in system_prompt


def test_battery_json_can_use_custom_top_level_key_template():
    df = pd.DataFrame(
        [
            {"questionnaire_item_id": 1, "question_content": "Q1"},
            {"questionnaire_item_id": 2, "question_content": "Q2"},
        ]
    )
    options = generate_likert_options(
        n=2,
        answer_texts=["Yes", "No"],
        response_generation_method=JSONResponseGenerationMethod(
            json_object=JSONObject(children=[JSONItem(json_field="answer")]),
            battery_question_key_template="item: {{QUESTION_CONTENT_PLACEHOLDER}}",
        ),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options, 2: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert '"item: Q1": {' in system_prompt
    assert '"item: Q2": {' in system_prompt


def test_verbalized_distribution_explanation_supports_options_placeholder():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    options = generate_likert_options(
        n=7,
        answer_texts=["UNWICHTIG", "SEHR WICHTIG"],
        only_from_to_scale=True,
        scale_prompt_template="Antwortmöglichkeiten: {start} bis {end}",
        response_generation_method=JSONVerbalizedDistribution(
            option_explanation_template=(
                "Wahrscheinlichkeit für {option} auf {{OPTIONS_PLACEHOLDER}}"
            )
        ),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert (
        '"1: UNWICHTIG": "Wahrscheinlichkeit für 1: UNWICHTIG auf Antwortmöglichkeiten: 1: '
        'UNWICHTIG bis 7: SEHR WICHTIG"' in system_prompt
    )
    assert '"2": "Wahrscheinlichkeit für 2 auf"' in system_prompt


def test_verbalized_distribution_options_placeholder_can_be_repeated_for_all_options():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    options = generate_likert_options(
        n=7,
        answer_texts=["UNWICHTIG", "SEHR WICHTIG"],
        only_from_to_scale=True,
        scale_prompt_template="Antwortmöglichkeiten: {start} bis {end}",
        response_generation_method=JSONVerbalizedDistribution(
            option_explanation_template=(
                "Wahrscheinlichkeit für {option} auf {{OPTIONS_PLACEHOLDER}}"
            ),
            explanation_prompt_placeholders_first_option_only=False,
        ),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert (
        '"2": "Wahrscheinlichkeit für 2 auf Antwortmöglichkeiten: 1: UNWICHTIG bis 7: SEHR '
        'WICHTIG"' in system_prompt
    )


def test_verbalized_distribution_explanation_supports_scale_range_placeholder():
    df = pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    options = generate_likert_options(
        n=7,
        answer_texts=["UNWICHTIG", "SEHR WICHTIG"],
        only_from_to_scale=True,
        scale_prompt_template="Antwortmöglichkeiten: {start} bis {end}",
        response_generation_method=JSONVerbalizedDistribution(
            option_explanation_template=(
                "Wahrscheinlichkeit für {option} {{SCALE_RANGE_PLACEHOLDER}}"
            )
        ),
    )

    prompt = LLMPrompt(
        questionnaire_source=df,
        system_prompt=f"SYS\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
    ).prepare_prompt(answer_options={1: options})

    system_prompt, _ = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert (
        '"1: UNWICHTIG": "Wahrscheinlichkeit für 1: UNWICHTIG Antwortmöglichkeiten: 1: '
        'UNWICHTIG bis 7: SEHR WICHTIG"' in system_prompt
    )
    assert '"2": "Wahrscheinlichkeit für 2"' in system_prompt


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
