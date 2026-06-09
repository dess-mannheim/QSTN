"""Tests for image ownership on LLMPrompt."""

import pandas as pd
import pytest

from qstn.prompt_builder import ImageInput, LLMPrompt
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
