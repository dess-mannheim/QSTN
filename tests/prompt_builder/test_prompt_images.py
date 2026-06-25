"""Tests for image ownership on LLMPrompt."""

import pandas as pd
import pytest

from qstn.prompt_builder import (
    ImageInput,
    LLMPrompt,
    QuestionnairePresentation,
    generate_likert_options,
)
from qstn.utilities.survey_objects import QuestionnaireItem


def test_llm_prompt_manages_global_and_per_item_images(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    global_image = ImageInput("https://example.com/global.png")
    item_image = ImageInput("https://example.com/item.png")

    returned = prompt.set_images([global_image]).add_image(item_image, item_id=1)

    assert returned is prompt
    assert prompt.get_images() == (global_image,)
    assert prompt.get_images(item_id=1) == (global_image, item_image)
    assert prompt.get_images(item_id=1, include_global=False) == (item_image,)
    assert prompt.get_images(item_id=2) == (global_image,)

    prompt.set_images([], item_id=1)
    assert prompt.get_images(item_id=1) == (global_image,)


def test_llm_prompt_coerces_sources_and_duplicates_image_state(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).add_image(
        "https://example.com/global.png"
    )

    duplicate = prompt.duplicate().set_images([])

    assert isinstance(prompt.get_images()[0], ImageInput)
    assert duplicate.get_images() == ()
    assert len(prompt.get_images()) == 1


def test_llm_prompt_rejects_unknown_item_ids(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)

    with pytest.raises(ValueError, match="does not exist"):
        prompt.add_image("https://example.com/image.png", item_id=99)


def test_llm_prompt_drops_stale_item_images_on_mutation_and_reload(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    prompt.add_image("https://example.com/one.png", item_id=1)
    prompt.add_image("https://example.com/two.png", item_id=2)

    prompt.replace_question(0, QuestionnaireItem(item_id=3, question_content="Q3"))
    assert set(prompt._item_images) == {2}

    prompt.remove_question(1)
    assert prompt._item_images == {}

    prompt.add_image("https://example.com/three.png", item_id=3)
    prompt.load_questionnaire_format(
        pd.DataFrame([{"questionnaire_item_id": 4, "question_content": "Q4"}])
    )
    assert prompt._item_images == {}


def test_renderer_returns_structured_content_for_single_and_sequential(
    mock_questionnaires,
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    global_image = ImageInput("https://example.com/global.png", label="Global")
    item_image = ImageInput("https://example.com/item.png", label="Item")
    prompt.add_image(global_image).add_image(item_image, item_id=1)

    _, single_content = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM,
        item_id=1,
    )
    _, sequential_content = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
        item_id=1,
    )

    assert single_content[1:] == (global_image, item_image)
    assert sequential_content == single_content


def test_renderer_interleaves_battery_images(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        prompt="""PREFIX
{{QUESTION_PLACEHOLDER}}
SUFFIX""",
    )
    global_image = ImageInput("https://example.com/global.png")
    first_image = ImageInput("https://example.com/one.png")
    second_image = ImageInput("https://example.com/two.png")
    prompt.add_image(global_image)
    prompt.add_image(first_image, item_id=1)
    prompt.add_image(second_image, item_id=2)

    _, content = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator="""
--
""",
    )

    assert content == (
        """PREFIX
""",
        global_image,
        "How do you feel about Red?",
        first_image,
        """
--
How do you feel about Blue?""",
        second_image,
        """
SUFFIX""",
    )
    assert "Questionnaire item" not in "".join(block for block in content if isinstance(block, str))


def test_renderer_keeps_aggregated_battery_options_with_item_images(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_source=mock_questionnaires,
        prompt=("PREFIX\n{{QUESTION_PLACEHOLDER}}\n" "OPTIONS\n{{OPTIONS_PLACEHOLDER}}\nSUFFIX"),
    )
    first_options = generate_likert_options(n=2, answer_texts=["A1", "A2"])
    second_options = generate_likert_options(n=2, answer_texts=["B1", "B2"])
    prompt.prepare_prompt(answer_options={1: first_options, 2: second_options})
    first_image = ImageInput("https://example.com/one.png")
    second_image = ImageInput("https://example.com/two.png")
    prompt.add_image(first_image, item_id=1)
    prompt.add_image(second_image, item_id=2)

    _, content = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY,
        item_separator="\n--\n",
    )

    assert content == (
        "PREFIX\n",
        "How do you feel about Red?",
        first_image,
        "\n--\nHow do you feel about Blue?",
        second_image,
        ("\nOPTIONS\nOptions are: 1: A1, 2: A2\n--\n" "Options are: 1: B1, 2: B2\nSUFFIX"),
    )


def test_image_free_renderer_and_str_remain_text_only(mock_questionnaires):
    prompt = LLMPrompt(
        questionnaire_name="Demo",
        questionnaire_source=mock_questionnaires,
    )
    system_prompt, user_prompt = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.BATTERY
    )

    assert isinstance(user_prompt, str)
    assert str(prompt) == (
        f"""=== Demo ===
=== SYSTEM_PROMPT ===
{system_prompt}
"""
        f"""=== USER_PROMPT_WITH_ALL_QUESTIONS ===
{user_prompt}"""
    )


def test_str_formats_image_labels_and_sources_without_data_payload(tmp_path):
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    )
    local_path = tmp_path / "local.png"
    local_path.write_bytes(b"png")
    data_url = "data:image/png;base64,aGVsbG8="
    prompt.set_images(
        [
            ImageInput("https://example.com/global.png", label="Remote"),
            ImageInput(local_path, label="Local"),
            ImageInput(data_url),
        ]
    )

    rendered = str(prompt)

    assert "[Image: Remote | https://example.com/global.png]" in rendered
    assert f"[Image: Local | {local_path}]" in rendered
    assert "[Image: unlabelled | data:image/png;base64,...]" in rendered
    assert "aGVsbG8=" not in rendered


def test_image_bearing_completion_rendering_is_rejected(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).add_image(
        "https://example.com/image.png"
    )

    with pytest.raises(ValueError, match="Image-bearing prompts.*chat"):
        prompt.get_prompt_for_questionnaire_type(inference_mode="completion")


def test_token_estimate_counts_labels_but_not_image_sources(monkeypatch):
    prompt = LLMPrompt(
        questionnaire_source=pd.DataFrame([{"questionnaire_item_id": 1, "question_content": "Q1"}])
    )
    text_only = prompt.duplicate()
    prompt.add_image(
        ImageInput(
            "https://example.com/a-very-long-image-source-that-is-not-counted.png",
            label="Visible label",
        )
    )

    def counter(text):
        return len(text or "")

    monkeypatch.setattr(prompt, "_get_token_counter", lambda *args: counter)
    monkeypatch.setattr(text_only, "_get_token_counter", lambda *args: counter)

    image_estimate = prompt.calculate_input_token_estimate(
        model_id="test",
        tokenizer_backend="transformers",
    )
    text_estimate = text_only.calculate_input_token_estimate(
        model_id="test",
        tokenizer_backend="transformers",
    )

    assert image_estimate - text_estimate == len("Visible label")
