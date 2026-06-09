"""Versioned, lossless persistence for supported ``LLMPrompt`` state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Annotated, Any, Literal, TextIO
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .inference.multimodal import ImageInput
from .inference.response_generation import (
    ChoiceResponseGenerationMethod,
    Constraints,
    JSONItem,
    JSONObject,
    JSONReasoningResponseGenerationMethod,
    JSONResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
    LogprobResponseGenerationMethod,
)
from .utilities.survey_objects import AnswerOptions, AnswerTexts, QuestionnaireItem

FORMAT_IDENTIFIER = "qstn.llm_prompt"
SCHEMA_VERSION = 1
TABLE_COLUMNS = [
    "schema_identifier",
    "schema_version",
    "record_type",
    "position",
    "payload",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ScalarSchema(_StrictModel):
    kind: Literal["none", "bool", "int", "float", "str"]
    value: Any

    @model_validator(mode="after")
    def validate_value(self) -> ScalarSchema:
        expected_types = {
            "none": type(None),
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
        }
        expected_type = expected_types[self.kind]
        if type(self.value) is not expected_type:
            raise ValueError(
                f"Scalar kind '{self.kind}' requires {expected_type.__name__}, "
                f"got {type(self.value).__name__}."
            )
        if self.kind == "float" and not math.isfinite(self.value):
            raise ValueError("Float values must be finite.")
        return self


class BaseModelPromptTemplateSchema(_StrictModel):
    user_prefix: str | None
    assistant_prefix: str | None
    separator: str
    system_prefix: str | None


class ImageSchema(_StrictModel):
    source_type: Literal["str", "path"]
    source: str
    label: str | None


class ConstraintsSchema(_StrictModel):
    enum: list[str] | None
    ge: float | None
    le: float | None
    min_length: int | None
    max_length: int | None
    pattern: str | None
    nullable: bool

    @model_validator(mode="after")
    def validate_floats(self) -> ConstraintsSchema:
        for name in ("ge", "le"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Constraint '{name}' must be finite.")
        return self


class JSONItemSchema(_StrictModel):
    node_type: Literal["item"]
    json_field: str
    value_type: Literal["string", "float", "int", "bool"]
    explanation: str | None
    constraints: ConstraintsSchema


class JSONObjectSchema(_StrictModel):
    node_type: Literal["object"]
    json_field: str | None
    explanation: str | None
    children: list[JSONNodeSchema]


JSONNodeSchema = Annotated[
    JSONItemSchema | JSONObjectSchema,
    Field(discriminator="node_type"),
]
JSONObjectSchema.model_rebuild()


class ChoiceResponseSchema(_StrictModel):
    method_type: Literal["choice"]
    allowed_choices: list[str] | None
    allowed_choices_template: str | None
    output_template: str
    output_index_only: bool


class LogprobResponseSchema(_StrictModel):
    method_type: Literal["logprob"]
    token_position: int
    token_limit: int
    top_logprobs: int
    allowed_choices: list[str] | None
    allowed_choices_template: str | None
    ignore_reasoning: bool
    output_template: str
    output_index_only: bool


class JSONResponseSchema(_StrictModel):
    method_type: Literal["json", "json_single", "json_reasoning"]
    json_object: JSONObjectSchema
    output_template: str
    output_index_only: bool
    battery_question_key_template: str
    constrain_answer_options: bool
    response_field: str | None


class JSONDistributionResponseSchema(_StrictModel):
    method_type: Literal["json_distribution"]
    json_object: JSONObjectSchema
    output_template: str
    output_index_only: bool
    battery_question_key_template: str
    constrain_answer_options: bool
    response_field: str | None
    verbalized_options: list[str]
    option_field_template: str
    option_explanation_template: str
    explanation_prompt_placeholders_first_option_only: bool


ResponseSchema = Annotated[
    ChoiceResponseSchema
    | LogprobResponseSchema
    | JSONResponseSchema
    | JSONDistributionResponseSchema,
    Field(discriminator="method_type"),
]


class AnswerTextsSchema(_StrictModel):
    full_answers: list[str]
    answer_texts: list[str] | None
    indices: list[str] | None
    index_answer_seperator: str
    option_seperators: str
    only_scale: bool

    @model_validator(mode="after")
    def validate_lengths(self) -> AnswerTextsSchema:
        if self.answer_texts is None and self.indices is None:
            raise ValueError("Answer texts require answer_texts, indices, or both.")
        if self.answer_texts is not None and self.indices is not None:
            if len(self.answer_texts) != len(self.indices):
                raise ValueError("answer_texts and indices must have the same length.")
        expected_length = len(self.answer_texts or self.indices or [])
        if len(self.full_answers) != expected_length:
            raise ValueError("full_answers must match the materialized option count.")
        return self


class AnswerOptionsSchema(_StrictModel):
    answer_texts: AnswerTextsSchema
    from_to_scale: bool
    list_prompt_template: str | None
    scale_prompt_template: str | None
    response_generation_method: ResponseSchema | None


class QuestionSchema(_StrictModel):
    item_id: ScalarSchema
    question_content: ScalarSchema
    question_stem: str | None
    answer_options: AnswerOptionsSchema | None
    prefilled_response: str | None
    images: list[ImageSchema]


class PromptStateSchema(_StrictModel):
    questionnaire_name: str
    system_prompt: str | None
    prompt: str
    verbose: bool
    base_model_prompt_template: BaseModelPromptTemplateSchema | None
    images: list[ImageSchema]


class LLMPromptSchema(_StrictModel):
    schema_identifier: Literal["qstn.llm_prompt"]
    schema_version: Literal[1]
    prompt_state: PromptStateSchema
    questions: list[QuestionSchema]


def _require_exact_instance(value: Any, expected_type: type, location: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(
            f"{location} must be exactly {expected_type.__name__}; " f"got {type(value).__name__}."
        )


def _require_attributes(value: Any, expected: set[str], location: str) -> None:
    actual = set(vars(value))
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unexpected:
            details.append(f"unexpected attributes: {unexpected}")
        if missing:
            details.append(f"missing attributes: {missing}")
        raise ValueError(f"{location} contains unsupported state ({'; '.join(details)}).")


def _serialize_scalar(value: Any, location: str) -> ScalarSchema:
    value_type = type(value)
    kind_by_type = {
        type(None): "none",
        bool: "bool",
        int: "int",
        float: "float",
        str: "str",
    }
    if value_type not in kind_by_type:
        raise TypeError(
            f"{location} must be None, bool, int, finite float, or str; "
            f"got {value_type.__name__}."
        )
    return ScalarSchema(kind=kind_by_type[value_type], value=value)


def _deserialize_scalar(schema: ScalarSchema) -> Any:
    return schema.value


def _serialize_image(image: ImageInput, location: str) -> ImageSchema:
    _require_exact_instance(image, ImageInput, location)
    _require_attributes(image, {"source", "label"}, location)
    source_type = "path" if isinstance(image.source, Path) else "str"
    if type(image.source) is not str and not isinstance(image.source, Path):
        raise TypeError(f"{location}.source must be str or pathlib.Path.")

    source = str(image.source)
    parsed = urlparse(source)
    is_remote = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    if not is_remote and not source.startswith("data:"):
        path = Path(image.source).expanduser()
        if not path.is_file():
            raise ValueError(f"{location} references a missing local image: {path}")

    return ImageSchema(source_type=source_type, source=source, label=image.label)


def _deserialize_image(schema: ImageSchema) -> ImageInput:
    source = Path(schema.source) if schema.source_type == "path" else schema.source
    return ImageInput(source=source, label=schema.label)


def _serialize_constraints(value: Constraints, location: str) -> ConstraintsSchema:
    _require_exact_instance(value, Constraints, location)
    _require_attributes(
        value,
        {"enum", "ge", "le", "min_length", "max_length", "pattern", "nullable"},
        location,
    )
    return ConstraintsSchema(
        enum=value.enum,
        ge=value.ge,
        le=value.le,
        min_length=value.min_length,
        max_length=value.max_length,
        pattern=value.pattern,
        nullable=value.nullable,
    )


def _serialize_json_node(value: JSONItem | JSONObject, location: str) -> JSONNodeSchema:
    if type(value) is JSONItem:
        _require_attributes(
            value,
            {"json_field", "value_type", "explanation", "constraints"},
            location,
        )
        return JSONItemSchema(
            node_type="item",
            json_field=value.json_field,
            value_type=value.value_type,
            explanation=value.explanation,
            constraints=_serialize_constraints(value.constraints, f"{location}.constraints"),
        )
    if type(value) is JSONObject:
        _require_attributes(value, {"json_field", "explanation", "children"}, location)
        return JSONObjectSchema(
            node_type="object",
            json_field=value.json_field,
            explanation=value.explanation,
            children=[
                _serialize_json_node(child, f"{location}.children[{index}]")
                for index, child in enumerate(value.children)
            ],
        )
    raise TypeError(
        f"{location} must be exactly JSONItem or JSONObject; got {type(value).__name__}."
    )


def _deserialize_json_node(schema: JSONNodeSchema) -> JSONItem | JSONObject:
    if isinstance(schema, JSONItemSchema):
        return JSONItem(
            json_field=schema.json_field,
            value_type=schema.value_type,
            explanation=schema.explanation,
            constraints=Constraints(**schema.constraints.model_dump()),
        )
    return JSONObject(
        json_field=schema.json_field,
        explanation=schema.explanation,
        children=[_deserialize_json_node(child) for child in schema.children],
    )


def _serialize_response(value: Any, location: str) -> ResponseSchema:
    if type(value) is ChoiceResponseGenerationMethod:
        _require_attributes(
            value,
            {
                "allowed_choices",
                "allowed_choices_template",
                "output_template",
                "output_index_only",
            },
            location,
        )
        return ChoiceResponseSchema(
            method_type="choice",
            allowed_choices=value.allowed_choices,
            allowed_choices_template=value.allowed_choices_template,
            output_template=value.output_template,
            output_index_only=value.output_index_only,
        )
    if type(value) is LogprobResponseGenerationMethod:
        _require_attributes(
            value,
            {
                "token_position",
                "token_limit",
                "top_logprobs",
                "allowed_choices",
                "allowed_choices_template",
                "ignore_reasoning",
                "output_template",
                "output_index_only",
            },
            location,
        )
        return LogprobResponseSchema(
            method_type="logprob",
            token_position=value.token_position,
            token_limit=value.token_limit,
            top_logprobs=value.top_logprobs,
            allowed_choices=value.allowed_choices,
            allowed_choices_template=value.allowed_choices_template,
            ignore_reasoning=value.ignore_reasoning,
            output_template=value.output_template,
            output_index_only=value.output_index_only,
        )

    json_types = {
        JSONResponseGenerationMethod: "json",
        JSONSingleResponseGenerationMethod: "json_single",
        JSONReasoningResponseGenerationMethod: "json_reasoning",
    }
    if type(value) in json_types:
        _require_attributes(
            value,
            {
                "json_object",
                "output_template",
                "output_index_only",
                "battery_question_key_template",
                "constrain_answer_options",
                "response_field",
            },
            location,
        )
        json_object = _serialize_json_node(value.json_object, f"{location}.json_object")
        if not isinstance(json_object, JSONObjectSchema):
            raise TypeError(f"{location}.json_object must be exactly JSONObject.")
        return JSONResponseSchema(
            method_type=json_types[type(value)],
            json_object=json_object,
            output_template=value.output_template,
            output_index_only=value.output_index_only,
            battery_question_key_template=value.battery_question_key_template,
            constrain_answer_options=value.constrain_answer_options,
            response_field=value.response_field,
        )
    if type(value) is JSONVerbalizedDistribution:
        _require_attributes(
            value,
            {
                "verbalized_options",
                "option_field_template",
                "option_explanation_template",
                "explanation_prompt_placeholders_first_option_only",
                "json_object",
                "output_template",
                "output_index_only",
                "battery_question_key_template",
                "constrain_answer_options",
                "response_field",
            },
            location,
        )
        json_object = _serialize_json_node(value.json_object, f"{location}.json_object")
        if not isinstance(json_object, JSONObjectSchema):
            raise TypeError(f"{location}.json_object must be exactly JSONObject.")
        return JSONDistributionResponseSchema(
            method_type="json_distribution",
            json_object=json_object,
            output_template=value.output_template,
            output_index_only=value.output_index_only,
            battery_question_key_template=value.battery_question_key_template,
            constrain_answer_options=value.constrain_answer_options,
            response_field=value.response_field,
            verbalized_options=value.verbalized_options,
            option_field_template=value.option_field_template,
            option_explanation_template=value.option_explanation_template,
            explanation_prompt_placeholders_first_option_only=(
                value.explanation_prompt_placeholders_first_option_only
            ),
        )
    raise TypeError(f"{location} uses unsupported response-generation type {type(value).__name__}.")


def _set_json_response_state(
    value: JSONResponseGenerationMethod,
    schema: JSONResponseSchema | JSONDistributionResponseSchema,
) -> None:
    value.json_object = _deserialize_json_node(schema.json_object)
    value.output_template = schema.output_template
    value.output_index_only = schema.output_index_only
    value.battery_question_key_template = schema.battery_question_key_template
    value.constrain_answer_options = schema.constrain_answer_options
    value.response_field = schema.response_field


def _deserialize_response(schema: ResponseSchema) -> Any:
    if isinstance(schema, ChoiceResponseSchema):
        return ChoiceResponseGenerationMethod(
            allowed_choices=schema.allowed_choices,
            allowed_choices_template=schema.allowed_choices_template,
            output_template=schema.output_template,
            output_index_only=schema.output_index_only,
        )
    if isinstance(schema, LogprobResponseSchema):
        return LogprobResponseGenerationMethod(
            token_position=schema.token_position,
            token_limit=schema.token_limit,
            top_logprobs=schema.top_logprobs,
            allowed_choices=schema.allowed_choices,
            allowed_choices_template=schema.allowed_choices_template,
            ignore_reasoning=schema.ignore_reasoning,
            output_template=schema.output_template,
            output_index_only=schema.output_index_only,
        )
    if isinstance(schema, JSONDistributionResponseSchema):
        value = JSONVerbalizedDistribution.__new__(JSONVerbalizedDistribution)
        value.verbalized_options = list(schema.verbalized_options)
        value.option_field_template = schema.option_field_template
        value.option_explanation_template = schema.option_explanation_template
        value.explanation_prompt_placeholders_first_option_only = (
            schema.explanation_prompt_placeholders_first_option_only
        )
        _set_json_response_state(value, schema)
        return value

    response_types = {
        "json": JSONResponseGenerationMethod,
        "json_single": JSONSingleResponseGenerationMethod,
        "json_reasoning": JSONReasoningResponseGenerationMethod,
    }
    response_type = response_types[schema.method_type]
    value = response_type.__new__(response_type)
    _set_json_response_state(value, schema)
    return value


def _serialize_answer_texts(value: AnswerTexts, location: str) -> AnswerTextsSchema:
    _require_exact_instance(value, AnswerTexts, location)
    _require_attributes(
        value,
        {
            "full_answers",
            "answer_texts",
            "indices",
            "index_answer_seperator",
            "option_seperators",
            "only_scale",
        },
        location,
    )
    return AnswerTextsSchema(
        full_answers=value.full_answers,
        answer_texts=value.answer_texts,
        indices=value.indices,
        index_answer_seperator=value.index_answer_seperator,
        option_seperators=value.option_seperators,
        only_scale=value.only_scale,
    )


def _deserialize_answer_texts(schema: AnswerTextsSchema) -> AnswerTexts:
    value = AnswerTexts.__new__(AnswerTexts)
    value.full_answers = list(schema.full_answers)
    value.answer_texts = None if schema.answer_texts is None else list(schema.answer_texts)
    value.indices = None if schema.indices is None else list(schema.indices)
    value.index_answer_seperator = schema.index_answer_seperator
    value.option_seperators = schema.option_seperators
    value.only_scale = schema.only_scale
    return value


def _serialize_answer_options(value: AnswerOptions, location: str) -> AnswerOptionsSchema:
    _require_exact_instance(value, AnswerOptions, location)
    _require_attributes(
        value,
        {
            "answer_texts",
            "from_to_scale",
            "list_prompt_template",
            "scale_prompt_template",
            "_response_generation_method",
        },
        location,
    )
    response_method = value.response_generation_method
    return AnswerOptionsSchema(
        answer_texts=_serialize_answer_texts(value.answer_texts, f"{location}.answer_texts"),
        from_to_scale=value.from_to_scale,
        list_prompt_template=value.list_prompt_template,
        scale_prompt_template=value.scale_prompt_template,
        response_generation_method=(
            None
            if response_method is None
            else _serialize_response(response_method, f"{location}.response_generation_method")
        ),
    )


def _deserialize_answer_options(schema: AnswerOptionsSchema) -> AnswerOptions:
    value = AnswerOptions.__new__(AnswerOptions)
    value.answer_texts = _deserialize_answer_texts(schema.answer_texts)
    value.from_to_scale = schema.from_to_scale
    value.list_prompt_template = schema.list_prompt_template
    value.scale_prompt_template = schema.scale_prompt_template
    object.__setattr__(
        value,
        "_response_generation_method",
        (
            None
            if schema.response_generation_method is None
            else _deserialize_response(schema.response_generation_method)
        ),
    )
    return value


def _serialize_question(
    question: QuestionnaireItem,
    images: tuple[ImageInput, ...],
    position: int,
) -> QuestionSchema:
    location = f"questions[{position}]"
    _require_exact_instance(question, QuestionnaireItem, location)
    _require_attributes(
        question,
        {
            "item_id",
            "question_content",
            "question_stem",
            "answer_options",
            "prefilled_response",
        },
        location,
    )
    return QuestionSchema(
        item_id=_serialize_scalar(question.item_id, f"{location}.item_id"),
        question_content=_serialize_scalar(
            question.question_content,
            f"{location}.question_content",
        ),
        question_stem=question.question_stem,
        answer_options=(
            None
            if question.answer_options is None
            else _serialize_answer_options(question.answer_options, f"{location}.answer_options")
        ),
        prefilled_response=question.prefilled_response,
        images=[
            _serialize_image(image, f"{location}.images[{index}]")
            for index, image in enumerate(images)
        ],
    )


def serialize_prompt(prompt: Any) -> LLMPromptSchema:
    from .prompt_builder import BaseModelPromptTemplate, LLMPrompt

    _require_exact_instance(prompt, LLMPrompt, "LLMPrompt")
    _require_attributes(
        prompt,
        {
            "_questions",
            "_images",
            "_item_images",
            "verbose",
            "questionnaire_name",
            "system_prompt",
            "prompt",
            "base_model_prompt_template",
        },
        "LLMPrompt",
    )
    if type(prompt._questions) is not list:
        raise TypeError("LLMPrompt._questions must be exactly list.")
    if type(prompt._images) is not tuple:
        raise TypeError("LLMPrompt._images must be exactly tuple.")
    if type(prompt._item_images) is not dict:
        raise TypeError("LLMPrompt._item_images must be exactly dict.")

    template = prompt.base_model_prompt_template
    template_schema = None
    if template is not None:
        _require_exact_instance(template, BaseModelPromptTemplate, "base_model_prompt_template")
        _require_attributes(
            template,
            {"user_prefix", "assistant_prefix", "separator", "system_prefix"},
            "base_model_prompt_template",
        )
        template_schema = BaseModelPromptTemplateSchema(
            user_prefix=template.user_prefix,
            assistant_prefix=template.assistant_prefix,
            separator=template.separator,
            system_prefix=template.system_prefix,
        )

    valid_item_ids = {question.item_id for question in prompt._questions}
    stale_image_ids = set(prompt._item_images) - valid_item_ids
    if stale_image_ids:
        raise ValueError(
            "LLMPrompt contains images for unknown questionnaire item IDs: "
            f"{sorted(map(str, stale_image_ids))}."
        )
    for item_id, images in prompt._item_images.items():
        _serialize_scalar(item_id, "LLMPrompt._item_images key")
        if type(images) is not tuple:
            raise TypeError("Each LLMPrompt._item_images value must be exactly tuple.")

    return LLMPromptSchema(
        schema_identifier=FORMAT_IDENTIFIER,
        schema_version=SCHEMA_VERSION,
        prompt_state=PromptStateSchema(
            questionnaire_name=prompt.questionnaire_name,
            system_prompt=prompt.system_prompt,
            prompt=prompt.prompt,
            verbose=prompt.verbose,
            base_model_prompt_template=template_schema,
            images=[
                _serialize_image(image, f"images[{index}]")
                for index, image in enumerate(prompt._images)
            ],
        ),
        questions=[
            _serialize_question(
                question,
                prompt._item_images.get(question.item_id, ()),
                position,
            )
            for position, question in enumerate(prompt._questions)
        ],
    )


def _deserialize_question(
    schema: QuestionSchema,
) -> tuple[QuestionnaireItem, tuple[ImageInput, ...]]:
    question = QuestionnaireItem(
        item_id=_deserialize_scalar(schema.item_id),
        question_content=_deserialize_scalar(schema.question_content),
        question_stem=schema.question_stem,
        answer_options=(
            None
            if schema.answer_options is None
            else _deserialize_answer_options(schema.answer_options)
        ),
        prefilled_response=schema.prefilled_response,
    )
    return question, tuple(_deserialize_image(image) for image in schema.images)


def deserialize_prompt(schema: LLMPromptSchema) -> Any:
    from .prompt_builder import BaseModelPromptTemplate, LLMPrompt

    prompt = LLMPrompt.__new__(LLMPrompt)
    prompt.verbose = schema.prompt_state.verbose
    prompt.questionnaire_name = schema.prompt_state.questionnaire_name
    prompt.system_prompt = schema.prompt_state.system_prompt
    prompt.prompt = schema.prompt_state.prompt
    prompt._images = tuple(_deserialize_image(image) for image in schema.prompt_state.images)

    template = schema.prompt_state.base_model_prompt_template
    prompt.base_model_prompt_template = (
        None
        if template is None
        else BaseModelPromptTemplate(
            user_prefix=template.user_prefix,
            assistant_prefix=template.assistant_prefix,
            separator=template.separator,
            system_prefix=template.system_prefix,
        )
    )

    questions = [_deserialize_question(question) for question in schema.questions]
    prompt._questions = [question for question, _ in questions]
    prompt._item_images = {question.item_id: images for question, images in questions if images}
    return prompt


def _migrate_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    identifier = payload.get("schema_identifier")
    if identifier != FORMAT_IDENTIFIER:
        raise ValueError(
            f"Unsupported LLMPrompt schema identifier: {identifier!r}. "
            f"Expected {FORMAT_IDENTIFIER!r}."
        )
    version = payload.get("schema_version")
    if type(version) is not int:
        raise ValueError("LLMPrompt schema_version must be an integer.")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"LLMPrompt schema version {version} is newer than supported version "
            f"{SCHEMA_VERSION}."
        )
    if version < SCHEMA_VERSION:
        raise ValueError(
            f"LLMPrompt schema version {version} has no migration to supported version "
            f"{SCHEMA_VERSION}."
        )
    return payload


def prompt_to_dict(prompt: Any) -> dict[str, Any]:
    return serialize_prompt(prompt).model_dump(mode="json")


def prompt_from_dict(data: Mapping[str, Any]) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError("LLMPrompt.from_dict() requires a mapping.")
    schema = LLMPromptSchema.model_validate(_migrate_payload(data))
    return deserialize_prompt(schema)


def _write_text(path_or_buf: str | Path | TextIO, content: str) -> None:
    if hasattr(path_or_buf, "write"):
        path_or_buf.write(content)
        return
    Path(path_or_buf).write_text(content, encoding="utf-8")


def prompt_to_json(
    prompt: Any,
    path_or_buf: str | Path | TextIO | None = None,
    *,
    indent: int | None = 2,
) -> str | None:
    content = json.dumps(
        prompt_to_dict(prompt),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )
    if path_or_buf is None:
        return content
    _write_text(path_or_buf, content)
    return None


def _read_text(path_or_buf: str | Path | TextIO) -> str:
    if hasattr(path_or_buf, "read"):
        return path_or_buf.read()
    if isinstance(path_or_buf, str) and path_or_buf.lstrip().startswith("{"):
        return path_or_buf
    return Path(path_or_buf).read_text(encoding="utf-8")


def prompt_from_json(path_or_buf: str | Path | TextIO) -> Any:
    content = _read_text(path_or_buf)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Serialized LLMPrompt JSON must contain an object.")
    return prompt_from_dict(data)


def prompt_to_dataframe(prompt: Any) -> pd.DataFrame:
    schema = serialize_prompt(prompt)
    rows = [
        {
            "schema_identifier": FORMAT_IDENTIFIER,
            "schema_version": SCHEMA_VERSION,
            "record_type": "prompt",
            "position": 0,
            "payload": schema.prompt_state.model_dump(mode="json"),
        }
    ]
    rows.extend(
        {
            "schema_identifier": FORMAT_IDENTIFIER,
            "schema_version": SCHEMA_VERSION,
            "record_type": "question",
            "position": position,
            "payload": question.model_dump(mode="json"),
        }
        for position, question in enumerate(schema.questions)
    )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _parse_record_payload(value: Any, row_number: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Row {row_number} payload must be a dictionary or JSON object string.")


def prompt_from_dataframe(dataframe: pd.DataFrame) -> Any:
    if type(dataframe) is not pd.DataFrame:
        raise TypeError("LLMPrompt.from_dataframe() requires a pandas DataFrame.")
    if list(dataframe.columns) != TABLE_COLUMNS:
        raise ValueError(f"Serialized LLMPrompt DataFrame columns must be {TABLE_COLUMNS}.")
    if dataframe.empty:
        raise ValueError("Serialized LLMPrompt DataFrame must contain a prompt record.")

    identifiers = dataframe["schema_identifier"].tolist()
    versions = dataframe["schema_version"].tolist()
    if any(identifier != FORMAT_IDENTIFIER for identifier in identifiers):
        raise ValueError("Serialized LLMPrompt records contain an invalid schema identifier.")
    if any(isinstance(version, bool) or not isinstance(version, Integral) for version in versions):
        raise ValueError("Serialized LLMPrompt record versions must be integers.")
    versions = [int(version) for version in versions]
    if len(set(versions)) != 1:
        raise ValueError("Serialized LLMPrompt records must use one schema version.")

    prompt_rows = dataframe[dataframe["record_type"] == "prompt"]
    question_rows = dataframe[dataframe["record_type"] == "question"]
    if len(prompt_rows) != 1:
        raise ValueError("Serialized LLMPrompt DataFrame must contain exactly one prompt record.")
    if len(prompt_rows) + len(question_rows) != len(dataframe):
        raise ValueError("Serialized LLMPrompt DataFrame contains an unknown record type.")
    expected_record_types = ["prompt"] + ["question"] * (len(dataframe) - 1)
    if dataframe["record_type"].tolist() != expected_record_types:
        raise ValueError(
            "Serialized LLMPrompt records must contain the prompt first, followed by questions."
        )
    prompt_row = prompt_rows.iloc[0]
    if (
        isinstance(prompt_row["position"], bool)
        or not isinstance(prompt_row["position"], Integral)
        or prompt_row["position"] != 0
    ):
        raise ValueError("The prompt record position must be integer 0.")

    positions = question_rows["position"].tolist()
    if any(
        isinstance(position, bool) or not isinstance(position, Integral) for position in positions
    ):
        raise ValueError("Question record positions must be integers.")
    positions = [int(position) for position in positions]
    if positions != list(range(len(question_rows))):
        raise ValueError("Question records must be ordered with contiguous positions from 0.")

    data = {
        "schema_identifier": FORMAT_IDENTIFIER,
        "schema_version": versions[0],
        "prompt_state": _parse_record_payload(prompt_row["payload"], prompt_row.name),
        "questions": [
            _parse_record_payload(row["payload"], index) for index, row in question_rows.iterrows()
        ],
    }
    return prompt_from_dict(data)


def prompt_to_csv(prompt: Any, path_or_buf: Any, **kwargs: Any) -> None:
    dataframe = prompt_to_dataframe(prompt).copy()
    dataframe["payload"] = dataframe["payload"].map(
        lambda payload: json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    kwargs.setdefault("index", False)
    dataframe.to_csv(path_or_buf, **kwargs)


def prompt_from_csv(path_or_buf: Any, **kwargs: Any) -> Any:
    dataframe = pd.read_csv(path_or_buf, **kwargs)
    return prompt_from_dataframe(dataframe)
