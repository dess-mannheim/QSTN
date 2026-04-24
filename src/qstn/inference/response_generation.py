from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

import qstn.utilities.placeholder

from ..utilities import prompt_templates, utils

if TYPE_CHECKING:
    from ..utilities.survey_objects import QuestionnaireItem

ScalarType = Literal["string", "float", "int", "bool"]


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_optional_template(
    value: str | None,
    prompt_formatter: dict[str, str] | None = None,
    **kwargs: Any,
) -> str | None:
    if value is None:
        return None
    if prompt_formatter:
        value = utils.safe_format_with_regex(value, prompt_formatter)
    return value.format_map(_SafeFormatDict(kwargs)).strip()


@dataclass
class Constraints:
    enum: list[str] | None = None
    ge: float | None = None
    le: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    nullable: bool = False


@dataclass
class JSONItem:
    json_field: str
    value_type: ScalarType = "string"
    explanation: str | None = None
    constraints: Constraints = field(default_factory=Constraints)

    def copy_with_formatted_strings(
        self,
        prompt_formatter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> JSONItem:
        return JSONItem(
            json_field=(
                _format_optional_template(
                    self.json_field,
                    prompt_formatter=prompt_formatter,
                    **kwargs,
                )
                or self.json_field
            ),
            value_type=self.value_type,
            explanation=_format_optional_template(
                self.explanation,
                prompt_formatter=prompt_formatter,
                **kwargs,
            ),
            constraints=deepcopy(self.constraints),
        )

    def to_prompt_value(self) -> str:
        if self.explanation is not None:
            return self.explanation

        if self.constraints.enum is not None:
            enum_values = ", ".join(str(value) for value in self.constraints.enum)
            return f"one of: {enum_values}"

        return self.value_type

    def to_prompt_obj(self) -> dict[str, Any]:
        return {self.json_field: self.to_prompt_value()}


@dataclass
class JSONObject:
    json_field: str | None = None
    explanation: str | None = None
    children: list[JSONItem | JSONObject] = field(default_factory=list)

    def to_prompt_value(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for child in self.children:
            result.update(child.to_prompt_obj())
        return result

    def to_prompt_obj(self) -> dict[str, Any]:
        if self.json_field is None:
            return self.to_prompt_value()
        return {self.json_field: self.to_prompt_value()}

    def to_prompt_str(self) -> str:
        return json.dumps(self.to_prompt_obj(), indent=2, ensure_ascii=False)

    def copy_with_formatted_strings(
        self,
        prompt_formatter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> JSONObject:
        return JSONObject(
            json_field=_format_optional_template(
                self.json_field,
                prompt_formatter=prompt_formatter,
                **kwargs,
            ),
            explanation=_format_optional_template(
                self.explanation,
                prompt_formatter=prompt_formatter,
                **kwargs,
            ),
            children=[
                child.copy_with_formatted_strings(
                    prompt_formatter=prompt_formatter,
                    **kwargs,
                )
                for child in self.children
            ],
        )


# --- Answer Production Base Classes ---
class ResponseGenerationMethod(ABC):
    """Abstract base class for constraining model output for closed-ended questions."""

    @abstractmethod
    def get_automatic_prompt(self: Self, questions: list[QuestionnaireItem] = ()):
        pass


class JSONResponseGenerationMethod(ResponseGenerationMethod):
    """Base class for constraining the model output using a JSON object tree."""

    def __init__(
        self,
        json_object: JSONObject,
        output_template: str = prompt_templates.SYSTEM_JSON_DEFAULT,
        output_index_only: bool = False,
        battery_question_key_template: str = qstn.utilities.placeholder.QUESTION_CONTENT,
        constrain_answer_options: bool = True,
        response_field: str | None = None,
    ):
        super().__init__()
        self.json_object = json_object
        self.output_template = output_template
        self.output_index_only = output_index_only
        self.battery_question_key_template = battery_question_key_template
        self.constrain_answer_options = constrain_answer_options
        self.response_field = response_field

    def get_json_prompt(self: Self, questions: list[QuestionnaireItem] = ()):
        del questions
        return self.json_object.to_prompt_str()

    def get_automatic_prompt(self: Self, questions: list[QuestionnaireItem] = ()):
        formatter = {qstn.utilities.placeholder.JSON_TEMPLATE: self.get_json_prompt()}
        return utils.safe_format_with_regex(self.output_template, formatter)

    def render_battery_question_key(self, question: QuestionnaireItem) -> str:
        formatter = {
            qstn.utilities.placeholder.QUESTION_CONTENT: str(question.question_content),
        }
        return utils.safe_format_with_regex(self.battery_question_key_template, formatter)


def copy_json_response_generation_method(
    response_generation_method: JSONResponseGenerationMethod,
    json_object: JSONObject | None = None,
    prompt_formatter: dict[str, str] | None = None,
    **format_kwargs: Any,
) -> JSONResponseGenerationMethod:
    if json_object is not None and len(format_kwargs) > 0:
        raise ValueError("Provide either `json_object` or formatting kwargs, not both.")

    if json_object is None:
        json_object = response_generation_method.json_object
        if len(format_kwargs) > 0 or prompt_formatter is not None:
            json_object = json_object.copy_with_formatted_strings(
                prompt_formatter=prompt_formatter,
                **format_kwargs,
            )

    return JSONResponseGenerationMethod(
        json_object=json_object,
        output_template=response_generation_method.output_template,
        output_index_only=response_generation_method.output_index_only,
        battery_question_key_template=response_generation_method.battery_question_key_template,
        constrain_answer_options=response_generation_method.constrain_answer_options,
        response_field=response_generation_method.response_field,
    )


def _set_json_item_enum(
    json_object: JSONObject,
    response_field: str,
    options: list[str],
) -> bool:
    found = False
    for child in json_object.children:
        if isinstance(child, JSONItem):
            if child.json_field == response_field:
                child.constraints.enum = list(options)
                found = True
            continue

        if _set_json_item_enum(child, response_field, options):
            found = True

    return found


def constrain_json_response_options(
    json_object: JSONObject,
    response_field: str | None,
    options: list[str],
) -> JSONObject:
    """Return a JSON object copy with answer-option enum constraints applied."""
    constrained_json_object = deepcopy(json_object)
    if response_field is None:
        return constrained_json_object

    if not _set_json_item_enum(constrained_json_object, response_field, options):
        raise ValueError(
            "Cannot constrain answer options because JSON response field "
            f"'{response_field}' was not found."
        )

    return constrained_json_object


class ChoiceResponseGenerationMethod(ResponseGenerationMethod):
    """
    Base class for constraining the model output using a Choice between answer options

    Attributes:
        allowed_choices: List of allowed choices for choice output
        system_prompt_template: Template used for formatting the system prompt,
            e.g., from `..utilities.prompt_templates`
        output_index_only: If True, constrain output to answer option index
            rather than the full text of each answer option
    """

    def __init__(
        self,
        allowed_choices: list[str] | None = None,
        allowed_choices_template: str | None = None,
        output_template: str = prompt_templates.SYSTEM_SINGLE_ANSWER,
        output_index_only: bool = False,
    ):
        super().__init__()
        self.allowed_choices = allowed_choices
        self.allowed_choices_template = allowed_choices_template
        self.output_template = output_template
        self.output_index_only = output_index_only  # TODO: implement

    def get_automatic_prompt(self: Self, questions: list[QuestionnaireItem] = ()):
        return self.output_template


class LogprobResponseGenerationMethod(ResponseGenerationMethod):
    """
    Base class for constraining the model output by requesting token proabilities

    Attributes:
        token_position: Position in output where logprobs are captured;
            use `0` for first-token probabilities (default)
        token_limit: Number of output tokens to generate; e.g., use `1` for
            first-token probabilities (default)
        top_logprobs: How many of the logprobs to consider, OpenAI supports at most 20
        allowed_choices: If not None, restrict output additionally with `guided_choice`
        ignore_reasoning: If True, only consider tokens after the reasoning
            output, i.e., after </think>
        system_prompt_template: Template used for formatting the system prompt,
            e.g., from `..utilities.prompt_templates`
        output_index_only: If True, constrain output to answer option index
            rather than the full text of each answer option
    """

    def __init__(
        self,
        token_position: int = 0,
        token_limit: int = 1,
        # OpenAI API default; local vLLM deployments might provide more.
        top_logprobs: int = 20,
        allowed_choices: list[str] | None = None,
        allowed_choices_template: str | None = None,
        ignore_reasoning: bool = True,
        output_template: str = prompt_templates.SYSTEM_SINGLE_ANSWER,
        output_index_only: bool = False,
    ):
        super().__init__()
        self.token_position = token_position
        self.token_limit = token_limit
        self.top_logprobs = top_logprobs
        self.allowed_choices = allowed_choices
        self.allowed_choices_template = allowed_choices_template
        self.ignore_reasoning = ignore_reasoning
        self.output_template = output_template
        self.output_index_only = output_index_only

    def get_automatic_prompt(self: Self, questions: list[QuestionnaireItem] = ()):
        return self.output_template


# --- Specific Answer Production Methods ---


class JSONSingleResponseGenerationMethod(JSONResponseGenerationMethod):
    """Response Generation Method: Structured Outputs"""

    def __init__(
        self,
        output_template=prompt_templates.SYSTEM_JSON_SINGLE_ANSWER,
        output_index_only: bool = False,
        answer_field: str = "answer",
        answer_explanation: str = "choose one of: {options}",
        battery_question_key_template: str = qstn.utilities.placeholder.QUESTION_CONTENT,
        constrain_answer_options: bool = True,
    ):
        super().__init__(
            json_object=JSONObject(
                children=[
                    JSONItem(
                        json_field=answer_field,
                        explanation=answer_explanation,
                        constraints=Constraints(),
                    )
                ]
            ),
            output_template=output_template,
            output_index_only=output_index_only,
            battery_question_key_template=battery_question_key_template,
            constrain_answer_options=constrain_answer_options,
            response_field=answer_field,
        )


class JSONReasoningResponseGenerationMethod(JSONResponseGenerationMethod):
    """Response Generation Method: Structured Outputs with Reasoning"""

    def __init__(
        self,
        output_template: str = prompt_templates.SYSTEM_JSON_REASONING,
        output_index_only: bool = False,
        reasoning_field: str = "reasoning",
        reasoning_explanation: str = "your reasoning about the answer options",
        answer_field: str = "answer",
        answer_explanation: str = "choose one of: {options}",
        battery_question_key_template: str = qstn.utilities.placeholder.QUESTION_CONTENT,
        constrain_answer_options: bool = True,
    ):
        super().__init__(
            json_object=JSONObject(
                children=[
                    JSONItem(
                        json_field=reasoning_field,
                        explanation=reasoning_explanation,
                    ),
                    JSONItem(
                        json_field=answer_field,
                        explanation=answer_explanation,
                        constraints=Constraints(),
                    ),
                ]
            ),
            output_template=output_template,
            output_index_only=output_index_only,
            battery_question_key_template=battery_question_key_template,
            constrain_answer_options=constrain_answer_options,
            response_field=answer_field,
        )


class JSONVerbalizedDistribution(JSONResponseGenerationMethod):
    """Response generation method for option-wise probability distributions."""

    def __init__(
        self,
        output_template: str = prompt_templates.SYSTEM_JSON_ALL_OPTIONS,
        output_index_only: bool = False,
        option_field_template: str = "{option}",
        option_explanation_template: str = "probability for: {option}",
        explanation_prompt_placeholders_first_option_only: bool = True,
        battery_question_key_template: str = qstn.utilities.placeholder.QUESTION_CONTENT,
    ):
        self.verbalized_options: list[str] = []
        self.option_field_template = option_field_template
        self.option_explanation_template = option_explanation_template
        self.explanation_prompt_placeholders_first_option_only = (
            explanation_prompt_placeholders_first_option_only
        )

        super().__init__(
            json_object=JSONObject(),
            output_template=output_template,
            output_index_only=output_index_only,
            battery_question_key_template=battery_question_key_template,
        )

    def set_verbalized_options(
        self,
        options: list[str],
        prompt_formatter: dict[str, str] | None = None,
    ) -> None:
        """Materialize one float field per answer option."""
        self.verbalized_options = list(options)
        children: list[JSONItem] = []
        seen_field_names: set[str] = set()
        options_text = ", ".join(str(option) for option in options)
        for idx, option in enumerate(options):
            field_name = _format_optional_template(
                self.option_field_template,
                prompt_formatter=prompt_formatter,
                option=option,
                options=options_text,
            )
            if field_name is None:
                raise ValueError("`option_field_template` must produce a JSON field name.")
            if field_name in seen_field_names:
                raise ValueError(
                    "Verbalized distribution contains duplicate option labels. "
                    f"Cannot create unique JSON fields for option '{option}'."
                )
            seen_field_names.add(field_name)
            explanation_prompt_formatter = prompt_formatter
            if self.explanation_prompt_placeholders_first_option_only and idx > 0:
                explanation_prompt_formatter = (
                    {key: "" for key in prompt_formatter} if prompt_formatter else None
                )
            children.append(
                JSONItem(
                    json_field=field_name,
                    value_type="float",
                    explanation=_format_optional_template(
                        self.option_explanation_template,
                        prompt_formatter=explanation_prompt_formatter,
                        option=option,
                        options=options_text,
                    ),
                )
            )
        self.json_object = JSONObject(children=children)


def resolve_battery_response_generation_method(
    questions: list[QuestionnaireItem],
    item_position: int = 0,
) -> ResponseGenerationMethod | None:
    """Resolve the response-generation method to use for battery prompts."""
    if len(questions) == 0:
        return None

    safe_item_position = min(max(item_position, 0), len(questions) - 1)
    selected_question = questions[safe_item_position]
    fallback_method = None
    if selected_question.answer_options:
        fallback_method = selected_question.answer_options.response_generation_method

    question_method_pairs: list[tuple[QuestionnaireItem, ResponseGenerationMethod]] = []
    for question in questions:
        if question.answer_options and question.answer_options.response_generation_method:
            question_method_pairs.append(
                (question, question.answer_options.response_generation_method)
            )

    if len(question_method_pairs) == 0:
        return None

    if not all(
        isinstance(method, JSONResponseGenerationMethod) for _, method in question_method_pairs
    ):
        return fallback_method or question_method_pairs[0][1]

    json_question_method_pairs: list[tuple[QuestionnaireItem, JSONResponseGenerationMethod]] = [
        (question, method)
        for question, method in question_method_pairs
        if isinstance(method, JSONResponseGenerationMethod)
    ]

    base_method: JSONResponseGenerationMethod
    if fallback_method and isinstance(fallback_method, JSONResponseGenerationMethod):
        base_method = fallback_method
    else:
        base_method = json_question_method_pairs[0][1]

    nested_children: list[JSONObject] = []
    seen_question_keys: set[str] = set()
    for question, method in json_question_method_pairs:
        question_key = base_method.render_battery_question_key(question)
        if question_key in seen_question_keys:
            raise ValueError(
                "Battery JSON contains duplicate question keys. "
                f"Cannot create unique top-level field '{question_key}'."
            )
        seen_question_keys.add(question_key)
        nested_children.append(
            JSONObject(
                json_field=question_key,
                children=deepcopy(method.json_object.children),
            )
        )
    merged_method = JSONResponseGenerationMethod(
        json_object=JSONObject(children=nested_children),
        output_template=base_method.output_template,
        output_index_only=base_method.output_index_only,
        battery_question_key_template=base_method.battery_question_key_template,
        constrain_answer_options=base_method.constrain_answer_options,
        response_field=base_method.response_field,
    )
    return merged_method
