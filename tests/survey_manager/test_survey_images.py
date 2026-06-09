"""Tests for image routing through survey presentation modes."""

import pytest

from qstn import survey_manager
from qstn.prompt_builder import ImageInput, LLMPrompt


def _sources(images):
    return [str(image.source) for image in images]


def test_single_item_routes_global_and_current_item_images(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    prompt.add_image("https://example.com/global.png")
    prompt.add_image("https://example.com/one.png", item_id=1)
    prompt.add_image("https://example.com/two.png", item_id=2)
    captured = []

    def fake_batch_generation(**kwargs):
        captured.append(kwargs["prompts"])
        return (["answer"], [None], [None])

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    survey_manager.conduct_survey_single_item(
        model=mock_openai_client,
        llm_prompts=prompt,
        print_progress=False,
    )

    assert _sources(captured[0][0][1:]) == [
        "https://example.com/global.png",
        "https://example.com/one.png",
    ]
    assert _sources(captured[1][0][1:]) == [
        "https://example.com/global.png",
        "https://example.com/two.png",
    ]


def test_battery_interleaves_each_question_with_its_images(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        prompt="PREFIX\n{{QUESTION_PLACEHOLDER}}\nSUFFIX",
    )
    global_image = ImageInput("https://example.com/global.png", label="Global")
    first_image = ImageInput("https://example.com/one.png", label="First detail")
    second_image = ImageInput("https://example.com/two.png", label="Second detail")
    prompt.add_image(global_image)
    prompt.add_image(first_image, item_id=1)
    prompt.add_image(second_image, item_id=2)
    captured = {}

    def fake_batch_generation(**kwargs):
        captured.update(kwargs)
        return (["answer"], [None], [None])

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    survey_manager.conduct_survey_battery(
        model=mock_openai_client,
        llm_prompts=prompt,
        item_separator="\n--\n",
        print_progress=False,
    )

    assert captured["prompts"] == [
        (
            "PREFIX\n",
            global_image,
            "Questionnaire item 1\nHow do you feel about Red?",
            first_image,
            "\n--\nQuestionnaire item 2\nHow do you feel about Blue?",
            second_image,
            "\nSUFFIX",
        )
    ]


def test_battery_without_images_keeps_original_text_prompt(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        prompt="PREFIX\n{{QUESTION_PLACEHOLDER}}\nSUFFIX",
    )
    expected_system, expected_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=survey_manager.QuestionnairePresentation.BATTERY,
        item_separator="\n--\n",
    )
    captured = {}

    def fake_batch_generation(**kwargs):
        captured.update(kwargs)
        return (["answer"], [None], [None])

    monkeypatch.setattr(survey_manager, "batch_generation", fake_batch_generation)

    survey_manager.conduct_survey_battery(
        model=mock_openai_client,
        llm_prompts=prompt,
        item_separator="\n--\n",
        print_progress=False,
    )

    assert captured["system_messages"] == [expected_system]
    assert captured["prompts"] == [expected_prompt]
    assert "image_inputs" not in captured
    assert "ordered_content" not in captured


def test_sequential_preserves_image_history_across_prefilled_turns(
    mock_questionnaires, mock_openai_client, monkeypatch
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).prepare_prompt(
        prefilled_responses={1: "prefilled"}
    )
    prompt.add_image("https://example.com/global.png")
    prompt.add_image("https://example.com/one.png", item_id=1)
    prompt.add_image("https://example.com/two.png", item_id=2)
    captured = {}

    def fake_conversation_generation(**kwargs):
        captured.update(kwargs)
        captured["assistant_messages"] = [
            list(messages) for messages in kwargs["assistant_messages"]
        ]
        return (["generated"], [None], [None])

    monkeypatch.setattr(
        survey_manager,
        "batch_turn_by_turn_generation",
        fake_conversation_generation,
    )

    survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=prompt,
        print_progress=False,
    )

    prompt_history = captured["prompts"][0]
    assert [_sources(turn[1:]) for turn in prompt_history] == [
        ["https://example.com/global.png", "https://example.com/one.png"],
        ["https://example.com/two.png"],
    ]
    assert captured["assistant_messages"] == [["prefilled"]]


def test_survey_completion_mode_rejects_attached_images(mock_questionnaires, mock_openai_client):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).add_image(
        "https://example.com/image.png"
    )

    with pytest.raises(ValueError, match="supported only"):
        survey_manager.conduct_survey_single_item(
            model=mock_openai_client,
            llm_prompts=prompt,
            inference_mode="completion",
            print_progress=False,
        )
