"""Notebook-style integration tests for the German-persona workflow.

These tests mirror the data flow in `docs/resources/german_general_personas.ipynb`
while mocking heavy inference calls.
"""

import json
import pandas as pd

from qstn import survey_manager
from qstn.inference import response_generation
from qstn.parser import parse_json, raw_responses
from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.utilities import create_one_dataframe, placeholder


def _build_notebook_like_inputs():
    personas = pd.DataFrame(
        [
            {"persona": "Person A: lebt in Berlin."},
            {"persona": "Person B: lebt in Hamburg."},
        ]
    )

    json_questionnaire = {
        "mp18": {
            "statement": "Wie zufrieden sind Sie mit der Demokratie?",
            "answers": ["1: Sehr unzufrieden", "2: Sehr zufrieden"],
        },
        "pizza": {
            "statement": "Wie stehen Sie zu Ananas auf Pizza?",
            "answers": ["1: Schrecklich", "2: Lecker"],
        },
    }

    questionnaire = pd.DataFrame(
        [
            {"questionnaire_item_id": qid, "question_content": payload["statement"]}
            for qid, payload in json_questionnaire.items()
        ]
    )

    all_cleaned_answers = []
    for qid, payload in json_questionnaire.items():
        cleaned = [answer.split(": ", 1)[1] for answer in payload["answers"]]
        from_to_scale = any("-" in text for text in cleaned)
        all_cleaned_answers.append(
            {"question": qid, "answer": cleaned, "from_to_scale": from_to_scale}
        )

    return personas, questionnaire, all_cleaned_answers


def _create_llm_prompt_for_persona(persona_row, questionnaire, all_cleaned_answers):
    system_prompt = "Nehme die Perspektive der folgenden Person ein: {persona}"
    prompt = (
        f"Welche der Antwortmöglichkeiten ist die Reaktion der Person auf folgende Frage: "
        f"{placeholder.PROMPT_QUESTIONS}\n"
        f"{placeholder.PROMPT_OPTIONS}\n"
        f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"
    )

    llm_prompt = LLMPrompt(
        questionnaire_source=questionnaire,
        questionnaire_name=str(persona_row.name),
        system_prompt=system_prompt.format(persona=persona_row["persona"]),
        prompt=prompt,
    )

    answer_options = {}
    for answer_bundle in all_cleaned_answers:
        rgm = response_generation.ChoiceResponseGenerationMethod(
            answer_bundle["answer"],
            output_template="Antworte nur mit der exakten Antwort.",
        )

        answer_options[answer_bundle["question"]] = generate_likert_options(
            n=len(answer_bundle["answer"]),
            answer_texts=answer_bundle["answer"],
            only_from_to_scale=answer_bundle["from_to_scale"],
            list_prompt_template="Antwortmöglichkeiten: {options}",
            scale_prompt_template="Antwortmöglichkeiten: {start} bis {end}",
            response_generation_method=rgm,
        )

    llm_prompt.prepare_prompt(answer_options=answer_options)
    return llm_prompt


def _create_json_llm_prompt_for_persona(persona_row, questionnaire, all_cleaned_answers):
    """Create notebook-like prompts but with JSON reasoning output constraints."""
    system_prompt = "Nehme die Perspektive der folgenden Person ein: {persona}"
    prompt = (
        f"Welche der Antwortmöglichkeiten ist die Reaktion der Person auf folgende Frage: "
        f"{placeholder.PROMPT_QUESTIONS}\n"
        f"{placeholder.PROMPT_OPTIONS}\n"
        f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"
    )

    llm_prompt = LLMPrompt(
        questionnaire_source=questionnaire,
        questionnaire_name=str(persona_row.name),
        system_prompt=system_prompt.format(persona=persona_row["persona"]),
        prompt=prompt,
    )

    answer_options = {}
    for answer_bundle in all_cleaned_answers:
        rgm = response_generation.JSONReasoningResponseGenerationMethod(
            output_template=f"Antworte nur im folgenden JSON format:\n{placeholder.JSON_TEMPLATE}"
        )
        answer_options[answer_bundle["question"]] = generate_likert_options(
            n=len(answer_bundle["answer"]),
            answer_texts=answer_bundle["answer"],
            only_from_to_scale=answer_bundle["from_to_scale"],
            list_prompt_template="Antwortmöglichkeiten: {options}",
            scale_prompt_template="Antwortmöglichkeiten: {start} bis {end}",
            response_generation_method=rgm,
        )

    llm_prompt.prepare_prompt(answer_options=answer_options)
    return llm_prompt


def test_notebook_like_prompt_construction():
    """Persona workflow should build prompts with question options and output instructions."""
    personas, questionnaire, all_cleaned_answers = _build_notebook_like_inputs()
    llm_prompts = personas.apply(
        _create_llm_prompt_for_persona,
        axis=1,
        args=(questionnaire, all_cleaned_answers),
    ).to_list()

    assert len(llm_prompts) == 2

    system_msg, user_prompt = llm_prompts[0].get_prompt_for_questionnaire_type(
        item_id="mp18"
    )
    assert "Person A" in system_msg
    assert "Wie zufrieden sind Sie mit der Demokratie?" in user_prompt
    assert "Antwortmöglichkeiten:" in user_prompt
    assert "Antworte nur mit der exakten Antwort." in user_prompt


def test_notebook_like_single_item_end_to_end_with_mocked_generation(monkeypatch):
    """Single-item survey should produce merged results for all personas and items."""
    personas, questionnaire, all_cleaned_answers = _build_notebook_like_inputs()
    llm_prompts = personas.apply(
        _create_llm_prompt_for_persona,
        axis=1,
        args=(questionnaire, all_cleaned_answers),
    ).to_list()

    def fake_batch_generation(**kwargs):
        size = len(kwargs["prompts"])
        answers = [f"MOCK_ANSWER_{i}" for i in range(size)]
        return answers, [None] * size, [None] * size

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    results = survey_manager.conduct_survey_single_item(
        model=object(),
        llm_prompts=llm_prompts,
        print_progress=False,
    )

    parsed = raw_responses(results)
    merged = create_one_dataframe(parsed)

    assert len(results) == 2
    assert merged.shape[0] == 4  # 2 personas * 2 questions
    assert set(merged["questionnaire_name"]) == {"0", "1"}
    assert set(merged["questionnaire_item_id"]) == {"mp18", "pizza"}
    assert all(value.startswith("MOCK_ANSWER_") for value in merged["llm_response"])


def test_notebook_like_sequential_keeps_history_and_respects_prefill(monkeypatch):
    """Sequential flow should preserve prompt history and skip generation for prefilled items."""
    personas, questionnaire, all_cleaned_answers = _build_notebook_like_inputs()
    prompt = personas.apply(
        _create_llm_prompt_for_persona,
        axis=1,
        args=(questionnaire, all_cleaned_answers),
    ).to_list()[0]

    recorded_calls = []

    def fake_batch_turn_by_turn_generation(**kwargs):
        snapshot = {
            "prompts": [list(conv) for conv in kwargs["prompts"]],
            "assistant_messages": [list(conv) for conv in kwargs["assistant_messages"]],
        }
        recorded_calls.append(snapshot)
        batch_size = len(kwargs["system_messages"])
        return (["GEN_0"] * batch_size, [None] * batch_size, [None] * batch_size)

    monkeypatch.setattr(
        survey_manager,
        "batch_turn_by_turn_generation",
        fake_batch_turn_by_turn_generation,
    )

    results = survey_manager.conduct_survey_sequential(
        model=object(),
        llm_prompts=[prompt],
        print_progress=False,
    )
    parsed = raw_responses(results)[prompt]

    assert len(recorded_calls) == 2
    assert len(recorded_calls[0]["prompts"][0]) == 1
    assert recorded_calls[0]["assistant_messages"][0] == []
    assert len(recorded_calls[1]["prompts"][0]) == 2
    assert recorded_calls[1]["assistant_messages"][0] == ["GEN_0"]
    assert parsed["llm_response"].tolist() == ["GEN_0", "GEN_0"]

    prompt_prefilled = prompt.duplicate()
    prompt_prefilled._questions[1].prefilled_response = "PREFILLED_PIZZA"
    recorded_calls.clear()

    prefilled_results = survey_manager.conduct_survey_sequential(
        model=object(),
        llm_prompts=[prompt_prefilled],
        print_progress=False,
    )
    parsed_prefilled = raw_responses(prefilled_results)[prompt_prefilled]

    assert len(recorded_calls) == 1
    assert parsed_prefilled["llm_response"].tolist() == ["GEN_0", "PREFILLED_PIZZA"]


def test_notebook_like_json_output_parsing_and_merge(monkeypatch):
    """JSON workflow should parse to structured columns and merge across personas."""
    personas, questionnaire, all_cleaned_answers = _build_notebook_like_inputs()
    llm_prompts = personas.apply(
        _create_json_llm_prompt_for_persona,
        axis=1,
        args=(questionnaire, all_cleaned_answers),
    ).to_list()

    def fake_batch_generation(**kwargs):
        size = len(kwargs["prompts"])
        payload = json.dumps({"reasoning": "weil", "answer": "Sehr zufrieden"})
        return [payload] * size, [None] * size, [None] * size

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    results = survey_manager.conduct_survey_single_item(
        model=object(),
        llm_prompts=llm_prompts,
        print_progress=False,
    )

    parsed = parse_json(results)
    merged = create_one_dataframe(parsed)

    assert merged.shape[0] == 4
    assert set(merged["questionnaire_name"]) == {"0", "1"}
    assert "reasoning" in merged.columns
    assert "answer" in merged.columns
    assert set(merged["reasoning"]) == {"weil"}
    assert set(merged["answer"]) == {"Sehr zufrieden"}


def test_notebook_like_battery_end_to_end_with_mocked_generation(monkeypatch):
    """Battery flow should aggregate all questions into one response per persona."""
    personas, questionnaire, all_cleaned_answers = _build_notebook_like_inputs()
    llm_prompts = personas.apply(
        _create_llm_prompt_for_persona,
        axis=1,
        args=(questionnaire, all_cleaned_answers),
    ).to_list()

    captured_prompts = []

    def fake_batch_generation(**kwargs):
        captured_prompts[:] = kwargs["prompts"]
        size = len(kwargs["prompts"])
        answers = [f"BATTERY_ANSWER_{i}" for i in range(size)]
        return answers, [None] * size, [None] * size

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    results = survey_manager.conduct_survey_battery(
        model=object(),
        llm_prompts=llm_prompts,
        print_progress=False,
        item_separator=" || ",
    )

    parsed = raw_responses(results)
    merged = create_one_dataframe(parsed)

    assert len(results) == 2
    assert merged.shape[0] == 2  # one battery response per persona
    assert set(merged["questionnaire_name"]) == {"0", "1"}
    assert set(merged["questionnaire_item_id"]) == {-1}
    assert all(value.startswith("BATTERY_ANSWER_") for value in merged["llm_response"])
    assert len(captured_prompts) == 2
    assert "Wie zufrieden sind Sie mit der Demokratie?" in captured_prompts[0]
    assert "Wie stehen Sie zu Ananas auf Pizza?" in captured_prompts[0]
    assert " || " in captured_prompts[0]
