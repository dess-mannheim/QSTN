# """Lossless persistence tests for ``LLMPrompt``."""

# from io import StringIO
# from pathlib import Path

# import pandas as pd
# import pytest

# from qstn.inference.response_generation import (
#     ChoiceResponseGenerationMethod,
#     Constraints,
#     JSONItem,
#     JSONObject,
#     JSONReasoningResponseGenerationMethod,
#     JSONResponseGenerationMethod,
#     JSONSingleResponseGenerationMethod,
#     JSONVerbalizedDistribution,
#     LogprobResponseGenerationMethod,
#     ResponseGenerationMethod,
# )
# from qstn.prompt_builder import BaseModelPromptTemplate, ImageInput, LLMPrompt
# from qstn.utilities import placeholder
# from qstn.utilities.constants import QuestionnairePresentation
# from qstn.utilities.survey_objects import AnswerOptions, AnswerTexts, QuestionnaireItem


# def _assert_prompt_behavior_equal(left: LLMPrompt, right: LLMPrompt) -> None:
#     assert right.to_dict() == left.to_dict()
#     assert len(right) == len(left)
#     for presentation in QuestionnairePresentation:
#         if not left.questions:
#             continue
#         for position in range(len(left)):
#             assert right.get_prompt_for_questionnaire_type(
#                 questionnaire_type=presentation,
#                 item_position=position,
#             ) == left.get_prompt_for_questionnaire_type(
#                 questionnaire_type=presentation,
#                 item_position=position,
#             )
#             assert right.get_prompt_for_questionnaire_type(
#                 questionnaire_type=presentation,
#                 item_position=position,
#                 inference_mode="completion",
#             ) == left.get_prompt_for_questionnaire_type(
#                 questionnaire_type=presentation,
#                 item_position=position,
#                 inference_mode="completion",
#             )


# def _round_trip_all_formats(prompt: LLMPrompt, tmp_path: Path) -> list[LLMPrompt]:
#     json_path = tmp_path / "prompt.json"
#     csv_path = tmp_path / "prompt.csv"
#     prompt.to_json(json_path)
#     prompt.to_csv(csv_path)
#     json_text = prompt.to_json()

#     return [
#         LLMPrompt.from_dict(prompt.to_dict()),
#         LLMPrompt.from_json(json_text),
#         LLMPrompt.from_json(json_path),
#         LLMPrompt.from_dataframe(prompt.to_dataframe()),
#         LLMPrompt.from_csv(csv_path),
#     ]


# def test_empty_prompt_round_trips_all_formats(tmp_path):
#     prompt = LLMPrompt(
#         questionnaire_name="empty",
#         system_prompt=None,
#         prompt="No questions",
#     ).set_base_model_prompt_template(
#         user_prefix=None,
#         assistant_prefix="Answer:",
#         separator="\n---\n",
#         system_prefix="System:",
#     )

#     for restored in _round_trip_all_formats(prompt, tmp_path):
#         _assert_prompt_behavior_equal(prompt, restored)


# def test_full_prompt_round_trip_preserves_materialized_state(tmp_path):
#     image_path = tmp_path / "pixel.png"
#     image_path.write_bytes(b"not-decoded-by-image-input")
#     generic_json = JSONResponseGenerationMethod(
#         json_object=JSONObject(
#             children=[
#                 JSONObject(
#                     json_field="result",
#                     explanation="nested",
#                     children=[
#                         JSONItem(
#                             json_field="score",
#                             value_type="float",
#                             constraints=Constraints(ge=0.0, le=1.0, nullable=True),
#                         )
#                     ],
#                 )
#             ]
#         ),
#         output_template="JSON: {JSON_TEMPLATE_PLACEHOLDER}",
#         output_index_only=True,
#         battery_question_key_template="item {{QUESTION_CONTENT_PLACEHOLDER}}",
#         constrain_answer_options=False,
#         response_field=None,
#     )
#     options = AnswerOptions(
#         answer_texts=AnswerTexts(
#             answer_texts=["Low", "", "High"],
#             indices=["1", "2", "3"],
#             index_answer_seperator=" => ",
#             option_seperators=" | ",
#         ),
#         from_to_scale=True,
#         list_prompt_template="Choices: {options}",
#         scale_prompt_template="Range {start} through {end}",
#         response_generation_method=generic_json,
#     )
#     options.answer_texts.full_answers[1] = "2 (middle)"
#     prompt = LLMPrompt(
#         questionnaire_name="full",
#         system_prompt=f"System\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}",
#         prompt=f"Ask\n{placeholder.PROMPT_QUESTIONS}",
#     )
#     prompt._questions = [
#         QuestionnaireItem(
#             item_id=1.5,
#             question_content=7,
#             question_stem="Stem:",
#             answer_options=options,
#             prefilled_response="saved",
#         ),
#         QuestionnaireItem(item_id=True, question_content=None),
#     ]
#     prompt.base_model_prompt_template = BaseModelPromptTemplate(
#         user_prefix="User:",
#         assistant_prefix="Assistant:",
#         separator="\n\n",
#         system_prefix=None,
#     )
#     prompt.add_image(ImageInput(image_path, label="local"))
#     prompt.add_image(ImageInput(str(image_path), label="string path"), item_id=True)
#     prompt.add_image("https://example.com/image.png", item_id=1.5)

#     for restored in _round_trip_all_formats(prompt, tmp_path):
#         _assert_prompt_behavior_equal(prompt, restored)
#         assert isinstance(restored.get_images()[0].source, Path)
#         assert type(restored.get_images(item_id=True, include_global=False)[0].source) is str


# @pytest.mark.parametrize(
#     "method",
#     [
#         ChoiceResponseGenerationMethod(
#             allowed_choices=["x", "y"],
#             allowed_choices_template=None,
#             output_template="Choose",
#             output_index_only=True,
#         ),
#         LogprobResponseGenerationMethod(
#             token_position=2,
#             token_limit=3,
#             top_logprobs=7,
#             allowed_choices=["a"],
#             allowed_choices_template=None,
#             ignore_reasoning=False,
#             output_template="Logprob",
#             output_index_only=True,
#         ),
#         JSONResponseGenerationMethod(
#             json_object=JSONObject(children=[JSONItem("value")]),
#             output_template="Generic {JSON_TEMPLATE_PLACEHOLDER}",
#             constrain_answer_options=False,
#         ),
#         JSONSingleResponseGenerationMethod(
#             output_template="Single {JSON_TEMPLATE_PLACEHOLDER}",
#             output_index_only=True,
#             answer_field="selection",
#             answer_explanation="select",
#             constrain_answer_options=False,
#         ),
#         JSONReasoningResponseGenerationMethod(
#             output_template="Reason {JSON_TEMPLATE_PLACEHOLDER}",
#             reasoning_field="why",
#             answer_field="selection",
#             constrain_answer_options=False,
#         ),
#         JSONVerbalizedDistribution(
#             output_template="Distribution {JSON_TEMPLATE_PLACEHOLDER}",
#             output_index_only=True,
#             option_field_template="p_{option}",
#             option_explanation_template="P({option})",
#             explanation_prompt_placeholders_first_option_only=False,
#         ),
#     ],
# )
# def test_supported_response_methods_round_trip_without_repreparation(method):
#     options = AnswerOptions(
#         answer_texts=AnswerTexts(["No", "Yes"], indices=["0", "1"]),
#         response_generation_method=method,
#     )
#     prompt = LLMPrompt()
#     prompt._questions = [
#         QuestionnaireItem(item_id="q", question_content="Question?", answer_options=options)
#     ]

#     restored = LLMPrompt.from_dict(prompt.to_dict())

#     _assert_prompt_behavior_equal(prompt, restored)
#     assert type(restored.get_question(0).answer_options.response_generation_method) is type(
#         prompt.get_question(0).answer_options.response_generation_method
#     )


# def test_dataframe_and_csv_use_versioned_records():
#     prompt = LLMPrompt()
#     prompt._questions = [
#         QuestionnaireItem(item_id="a", question_content="A"),
#         QuestionnaireItem(item_id="b", question_content="B"),
#     ]

#     dataframe = prompt.to_dataframe()

#     assert list(dataframe.columns) == [
#         "schema_identifier",
#         "schema_version",
#         "record_type",
#         "position",
#         "payload",
#     ]
#     assert dataframe["record_type"].tolist() == ["prompt", "question", "question"]
#     assert isinstance(dataframe.loc[0, "payload"], dict)

#     buffer = StringIO()
#     prompt.to_csv(buffer)
#     buffer.seek(0)
#     csv_dataframe = pd.read_csv(buffer)
#     assert isinstance(csv_dataframe.loc[0, "payload"], str)


# def test_invalid_dataframe_records_are_rejected():
#     prompt = LLMPrompt()
#     prompt._questions = [QuestionnaireItem(item_id=1, question_content="Q")]
#     dataframe = prompt.to_dataframe()

#     with pytest.raises(ValueError, match="contiguous"):
#         invalid = dataframe.copy()
#         invalid.loc[1, "position"] = 2
#         LLMPrompt.from_dataframe(invalid)

#     with pytest.raises(ValueError, match="exactly one prompt"):
#         LLMPrompt.from_dataframe(pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True))

#     with pytest.raises(ValueError, match="columns"):
#         LLMPrompt.from_dataframe(dataframe.assign(extra="x"))

#     with pytest.raises(ValueError, match="prompt first"):
#         LLMPrompt.from_dataframe(dataframe.iloc[::-1].reset_index(drop=True))


# def test_schema_identifier_and_versions_are_rejected():
#     payload = LLMPrompt().to_dict()

#     with pytest.raises(ValueError, match="identifier"):
#         LLMPrompt.from_dict(dict(payload, schema_identifier="other"))
#     with pytest.raises(ValueError, match="no migration"):
#         LLMPrompt.from_dict(dict(payload, schema_version=0))
#     with pytest.raises(ValueError, match="newer"):
#         LLMPrompt.from_dict(dict(payload, schema_version=2))


# def test_missing_local_image_fails_before_writing(tmp_path):
#     image_path = tmp_path / "image.png"
#     image_path.write_bytes(b"image")
#     prompt = LLMPrompt().add_image(image_path)
#     image_path.unlink()
#     output_path = tmp_path / "prompt.json"

#     with pytest.raises(ValueError, match="missing local image"):
#         prompt.to_json(output_path)

#     assert not output_path.exists()


# def test_unsupported_state_fails_before_writing(tmp_path):
#     class CustomResponse(ResponseGenerationMethod):
#         def get_automatic_prompt(self, questions=()):
#             return "custom"

#     prompt = LLMPrompt()
#     prompt._questions = [
#         QuestionnaireItem(
#             item_id=1,
#             question_content="Q",
#             answer_options=AnswerOptions(
#                 answer_texts=AnswerTexts(["Yes"]),
#                 response_generation_method=CustomResponse(),
#             ),
#         )
#     ]
#     output_path = tmp_path / "prompt.json"

#     with pytest.raises(TypeError, match="CustomResponse"):
#         prompt.to_json(output_path)

#     assert not output_path.exists()


# def test_non_finite_and_unknown_fields_are_rejected():
#     prompt = LLMPrompt()
#     prompt._questions = [QuestionnaireItem(item_id=float("nan"), question_content="Q")]
#     with pytest.raises(ValueError, match="finite"):
#         prompt.to_dict()

#     payload = LLMPrompt().to_dict()
#     payload["prompt_state"]["unknown"] = True
#     with pytest.raises(ValueError, match="Extra inputs"):
#         LLMPrompt.from_dict(payload)
