"""Tests for ordered mixed-media ownership on LLMPrompt."""

import pandas as pd
import pytest

from qstn.inference import AudioInput, ImageInput, VideoInput
from qstn.prompt_builder import LLMPrompt, QuestionnairePresentation
from qstn.utilities.survey_objects import QuestionnaireItem


@pytest.mark.parametrize("item_id", [None, 1])
@pytest.mark.parametrize(
    "kind, cls", [("image", ImageInput), ("audio", AudioInput), ("video", VideoInput)]
)
def test_typed_replacement_appends_and_preserves_other_media(
    mock_questionnaires, item_id, kind, cls
):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    original = (
        AudioInput("https://example.com/a"),
        ImageInput("https://example.com/i"),
        VideoInput("https://example.com/v"),
        cls("https://example.com/second"),
    )
    prompt.set_media(original, item_id=item_id)
    replacement = cls("https://example.com/new")
    setter = getattr(prompt, f"set_{kind}s")
    getter = getattr(prompt, f"get_{kind}s")
    assert setter([replacement], item_id=item_id) is prompt
    retained = tuple(block for block in original if not isinstance(block, cls))
    assert prompt.get_media(item_id=item_id) == (*retained, replacement)
    assert getter(item_id=item_id) == (replacement,)
    setter([], item_id=item_id)
    assert prompt.get_media(item_id=item_id) == retained
    # With no previous attachment of this modality, replacements still append.
    setter([replacement], item_id=item_id)
    assert prompt.get_media(item_id=item_id) == (*retained, replacement)


def test_mixed_additions_scopes_and_duplication(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    assert prompt.add_audio("https://example.com/a") is prompt
    prompt.add_image("https://example.com/i").add_video("https://example.com/v")
    global_media = prompt.get_media()
    assert tuple(type(block) for block in global_media) == (AudioInput, ImageInput, VideoInput)
    item_audio = AudioInput("https://example.com/item")
    assert prompt.add_media(item_audio, item_id=1) is prompt
    assert prompt.get_media(item_id=1) == (*global_media, item_audio)
    assert prompt.get_media(item_id=1, include_global=False) == (item_audio,)
    assert prompt.get_media(include_global=False) == ()
    assert prompt.get_audios(item_id=1) == (global_media[0], item_audio)
    assert prompt.get_media(item_id=2) == global_media
    duplicate = prompt.duplicate().set_media([]).set_media([], item_id=1)
    assert duplicate.get_media(item_id=1) == ()
    assert prompt.get_media(item_id=1) == (*global_media, item_audio)
    prompt.set_videos(["https://example.com/new"], item_id=1)
    assert prompt.get_media() == global_media
    assert len(prompt.get_media(item_id=1, include_global=False)) == 2
    prompt.set_media([], item_id=1)
    assert prompt.get_media(item_id=1) == global_media


@pytest.mark.parametrize(
    "kind, cls, suffix", [("audio", AudioInput, "wav"), ("video", VideoInput, "mp4")]
)
def test_typed_methods_coerce_paths_and_reject_wrong_modality(
    mock_questionnaires, tmp_path, kind, cls, suffix
):
    path = tmp_path / f"media.{suffix}"
    path.write_bytes(b"media")
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    setter = getattr(prompt, f"set_{kind}s")
    adder = getattr(prompt, f"add_{kind}")
    getter = getattr(prompt, f"get_{kind}s")
    setter([path, str(path)])
    assert getter() == (cls(path), cls(str(path)))
    before = prompt.get_media()
    with pytest.raises(TypeError):
        setter([path, ImageInput("https://example.com/image")])
    with pytest.raises(TypeError):
        adder(ImageInput("https://example.com/image"))
    assert prompt.get_media() == before


@pytest.mark.parametrize("invalid", ["https://example.com/a.wav", b"raw", object()])
def test_mixed_methods_require_typed_objects(mock_questionnaires, invalid):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    with pytest.raises(TypeError, match="ImageInput"):
        prompt.add_media(invalid)
    with pytest.raises(TypeError, match="ImageInput"):
        prompt.set_media([invalid])
    assert prompt.get_media() == ()


def test_mixed_mutations_are_atomic_and_validate_item_ids(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    audio = AudioInput("https://example.com/a")
    prompt.add_media(audio)
    with pytest.raises(TypeError):
        prompt.set_media([audio, "bare-url"])
    for operation in (
        lambda: prompt.add_media(audio, item_id=99),
        lambda: prompt.set_media([], item_id=99),
        lambda: prompt.get_media(item_id=99),
    ):
        with pytest.raises(ValueError, match="does not exist"):
            operation()
    assert prompt.get_media() == (audio,)


def test_mixed_media_cleanup_on_question_changes(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    global_audio = AudioInput("https://example.com/global")
    prompt.add_audio(global_audio).add_audio(global_audio, item_id=1)
    prompt.add_video("https://example.com/video", item_id=2)
    prompt.replace_question(0, QuestionnaireItem(item_id=3, question_content="Q3"))
    assert set(prompt._item_media) == {2}
    prompt.remove_question(1)
    assert prompt._item_media == {}
    prompt.add_video("https://example.com/video", item_id=3)
    prompt.load_questionnaire_format(
        pd.DataFrame([{"questionnaire_item_id": 4, "question_content": "Q4"}])
    )
    assert prompt._item_media == {}
    assert prompt.get_media() == (global_audio,)


@pytest.mark.parametrize("cls", [AudioInput, VideoInput])
@pytest.mark.parametrize("presentation", list(QuestionnairePresentation))
def test_media_completion_rendering_rejected(mock_questionnaires, cls, presentation):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires).add_media(
        cls("https://example.com/media")
    )
    with pytest.raises(ValueError, match="chat"):
        prompt.get_prompt_for_questionnaire_type(
            questionnaire_type=presentation, item_id=1, inference_mode="completion"
        )


def test_renderer_orders_video_before_text_and_audio_after_text(mock_questionnaires):
    prompt = LLMPrompt(questionnaire_source=mock_questionnaires)
    audio = AudioInput("https://example.com/audio.wav", label="Audio")
    video = VideoInput("https://example.com/video.mp4", label="Video")
    prompt.add_audio(audio).add_video(video)

    _, content = prompt.get_prompt_for_questionnaire_type(
        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM,
        item_id=1,
    )

    assert content[0] is video
    assert isinstance(content[1], str)
    assert content[-1] is audio
