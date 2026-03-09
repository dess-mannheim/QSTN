import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import qstn.utilities.placeholder

from ..utilities import constants, prompt_creation, prompt_templates, utils

if TYPE_CHECKING:
    from ..utilities.survey_objects import QuestionnaireItem

# --- Answer Production Base Classes ---


class ResponseGenerationMethod(ABC):
    """Abstract base class for constraining model output for closed-ended questions."""

    @abstractmethod
    def get_automatic_prompt(self: Self, questions: list["QuestionnaireItem"] = ()):
        pass


class JSONResponseGenerationMethod(ResponseGenerationMethod):
    """
    Base class for constraining the model output using JSON Schema

    Attributes:
        json_fields: List of field names for JSON output, optionally as dicts
            of format {"field_name": "explanation"}
        constraints: Optional constraints for field values
        system_prompt_template: Template used for formatting the system prompt,
            e.g., from `..utilities.prompt_templates`
        output_index_only: If True, constrain output to answer option index
            rather than the full text of each answer option
    """

    def __init__(
        self,
        json_fields: list[str] | dict[str, str],  # required
        constraints: dict[str, list[str]] | None = None,  # remains optional
        output_template: str = prompt_templates.SYSTEM_JSON_DEFAULT,
        output_index_only: bool = False,
    ):
        super().__init__()
        if constraints is not None:
            if isinstance(json_fields, dict):
                difference = set(constraints.keys()) - set(json_fields.keys())
            else:
                difference = set(constraints.keys()) - set(json_fields)
            if len(difference) > 0:
                warnings.warn(
                    f"Constraints specified for non-existing fields: {difference}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.json_fields = json_fields
        self.constraints = constraints
        self.output_template = output_template
        self.output_index_only = output_index_only

    def _expand_multi_question_items(
        self: Self,
        questions: list["QuestionnaireItem"],
        attributes: list[str],
        explanations: list[str] | None = None,
    ) -> tuple[list[str], list[str] | None]:
        if len(questions) <= 1:
            if explanations is None:
                return list(attributes), None
            return list(attributes), list(explanations)

        expanded_attributes: list[str] = []
        expanded_explanations: list[str] | None = [] if explanations is not None else None

        for question in questions:
            for idx, attribute in enumerate(attributes):
                expanded_attributes.append(f"{attribute}_{question.question_content}")
                if expanded_explanations is not None:
                    expanded_explanations.append(explanations[idx])

        return expanded_attributes, expanded_explanations

    def _expand_multi_question_constraints(
        self: Self,
        questions: list["QuestionnaireItem"],
        constraints: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        if len(questions) <= 1:
            return dict(constraints)

        expanded_constraints: dict[str, list[str]] = {}
        keys = list(constraints.keys())
        for question in questions:
            for key in keys:
                expanded_constraints[f"{key}_{question.question_content}"] = constraints[key]
        return expanded_constraints

    def get_json_prompt(self: Self, questions: list["QuestionnaireItem"] = ()):
        num_questions = len(questions)
        if isinstance(self.json_fields, dict):
            json_attributes = list(self.json_fields.keys())
            json_explanation = list(self.json_fields.values())
        else:
            json_attributes = list(self.json_fields)
            json_explanation = None

        if num_questions > 1:
            json_attributes, json_explanation = self._expand_multi_question_items(
                questions=questions,
                attributes=json_attributes,
                explanations=json_explanation,
            )
        creator = prompt_creation.PromptCreation()
        creator.set_output_format_json(
            json_attributes=json_attributes,
            json_explanation=json_explanation,
            json_instructions=None,
        )

        return creator.get_output_prompt()

    def get_automatic_prompt(self: Self, questions: list["QuestionnaireItem"] = ()):
        formatter = {
            qstn.utilities.placeholder.JSON_TEMPLATE: self.get_json_prompt(questions=questions)
        }
        return utils.safe_format_with_regex(self.output_template, formatter)

    def create_new_rgm_with_multiple_questions(
        self: Self, questions: list["QuestionnaireItem"] = ()
    ) -> Self:
        num_questions = len(questions)
        if num_questions <= 1:
            return self

        if isinstance(self.json_fields, dict):
            original_attributes = list(self.json_fields.keys())
            original_explanations = list(self.json_fields.values())
        else:
            original_attributes = list(self.json_fields)
            original_explanations = None

        new_attributes, new_explanations = self._expand_multi_question_items(
            questions=questions,
            attributes=original_attributes,
            explanations=original_explanations,
        )

        if new_explanations is not None:
            json_fields = dict(zip(new_attributes, new_explanations))
        else:
            json_fields = new_attributes

        new_constraints = None
        if self.constraints:
            new_constraints = self._expand_multi_question_constraints(
                questions=questions,
                constraints=self.constraints,
            )

        return JSONResponseGenerationMethod(
            json_fields=json_fields,
            constraints=new_constraints,
            output_template=self.output_template,
            output_index_only=self.output_index_only,
        )


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
        allowed_choices: list[str],  # required
        output_template: str = prompt_templates.SYSTEM_SINGLE_ANSWER,
        output_index_only: bool = False,
    ):
        super().__init__()
        self.allowed_choices = allowed_choices
        self.output_template = output_template
        self.output_index_only = output_index_only  # TODO: implement

    def get_automatic_prompt(self: Self, questions: list["QuestionnaireItem"] = ()):
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
        ignore_reasoning: bool = True,
        output_template: str = prompt_templates.SYSTEM_SINGLE_ANSWER,
        output_index_only: bool = False,
    ):
        super().__init__()
        self.token_position = token_position
        self.token_limit = token_limit
        self.top_logprobs = top_logprobs
        self.allowed_choices = (
            allowed_choices  # same name enables re-using code from Choice_AnswerProductionMethod
        )
        self.ignore_reasoning = ignore_reasoning
        self.output_template = output_template
        self.output_index_only = output_index_only

    def get_automatic_prompt(self: Self, questions: list["QuestionnaireItem"] = ()):
        return self.output_template


# --- Specific Answer Production Methods ---


class JSONSingleResponseGenerationMethod(JSONResponseGenerationMethod):
    """Response Generation Method: Structured Outputs"""

    def __init__(
        self,
        output_template=prompt_templates.SYSTEM_JSON_SINGLE_ANSWER,
        output_index_only: bool = False,
    ):
        super().__init__(
            json_fields={"answer": constants.OPTIONS_ADJUST},
            constraints={"answer": constants.OPTIONS_ADJUST},
            output_template=output_template,
            output_index_only=output_index_only,
        )


class JSONReasoningResponseGenerationMethod(JSONResponseGenerationMethod):
    """Response Generation Method: Structured Outputs with Reasoning"""

    def __init__(
        self,
        output_template: str = prompt_templates.SYSTEM_JSON_REASONING,
        output_index_only: bool = False,
    ):

        json_fields = {
            "reasoning": "your reasoning about the answer options",
            "answer": constants.OPTIONS_ADJUST,
        }

        super().__init__(
            json_fields=json_fields,
            constraints={"answer": constants.OPTIONS_ADJUST},
            output_template=output_template,
            output_index_only=output_index_only,
        )


class JSONVerbalizedDistribution(JSONResponseGenerationMethod):
    """Response generation method for option-wise probability distributions.

    This method materializes one JSON float field per answer option and is
    primarily used for "verbalized distribution" tasks where the model should
    return a probability for each option.

    Field keys and explanations are configurable via templates and support the
    placeholders:
    - ``{question}``
    - ``{option}``
    - ``{question_id}``
    - ``{question_order}``
    - ``{option_index}``

    By default, the generated key pattern is
    ``"{question} | q{question_order}_o{option_index}"`` and explanations are
    ``"probability for: {option}"``.

    Question shortening is only applied when ``{question}`` is part of the
    configured field template. The first option keeps the full question text;
    following options may use a shortened prefix (e.g., ``"Wie groß..."``).

    Args:
        output_template (str): System prompt template used for automatic
            output instructions.
        output_index_only (bool): If True, use option indices instead of full
            option texts where applicable.
        json_field_template (str): Template used to generate JSON field keys.
        json_explanation_template (str): Template used for per-field
            explanation strings.
        shorten_question_in_fields (bool): Whether to shorten question text in
            fields for options after the first one.
        shorten_to (int): Number of words to keep when shortening.
    """

    def __init__(
        self,
        output_template=prompt_templates.SYSTEM_JSON_ALL_OPTIONS,
        output_index_only: bool = False,
        json_field_template: str = "{question} | q{question_order}_o{option_index}",
        json_explanation_template: str = "probability for: {option}",
        shorten_question_in_fields: bool = True,
        shorten_to: int = 2,
    ):
        self.json_field_template = json_field_template
        self.json_explanation_template = json_explanation_template
        self.shorten_question_in_fields = shorten_question_in_fields
        self.shorten_to = shorten_to
        self.verbalized_options: list[str] = []

        super().__init__(
            # will be set when given to answer options
            json_fields=None,
            constraints=None,
            # Variables
            output_template=output_template,
            output_index_only=output_index_only,
        )

    def _question_for_field(self, question: str | None = None, option_index: int = 1) -> str:
        """Return a normalized question string for field templates."""
        normalized_question = (question or "").strip()
        if not normalized_question:
            return ""
        if not self.shorten_question_in_fields or option_index <= 1:
            return normalized_question

        words = normalized_question.split()
        short_prefix = " ".join(words[: self.shorten_to])
        return f"{short_prefix}..."

    def _format_field(
        self,
        option: str,
        question: str | None = None,
        option_index: int | None = None,
        question_id: str | int | None = None,
        question_order: int | None = None,
    ) -> str:
        """Render ``json_field_template`` with the provided placeholder values."""
        return self.json_field_template.format(
            question=(question or ""),
            option=option,
            option_index=(option_index or ""),
            question_id=(question_id or ""),
            question_order=(question_order or ""),
        )

    def _format_explanation(
        self,
        option: str,
        question: str | None = None,
        option_index: int | None = None,
        question_id: str | int | None = None,
        question_order: int | None = None,
    ) -> str:
        """Render ``json_explanation_template`` with placeholder values."""
        return self.json_explanation_template.format(
            question=(question or ""),
            option=option,
            option_index=(option_index or ""),
            question_id=(question_id or ""),
            question_order=(question_order or ""),
        )

    def set_verbalized_options(
        self,
        options: list[str],
        question: str | None = None,
        question_id: str | int | None = None,
        question_order: int | None = None,
    ) -> None:
        """Materialize JSON fields/constraints for the given option set.

        Args:
            options (list[str]): Options to generate probability fields for.
            question (str | None): Optional question text used by templates.
                If provided, field keys are rendered from ``json_field_template``.
                If omitted, option strings are used as keys directly.
            question_id (str | int | None): Optional stable question id for
                template placeholders.
            question_order (int | None): Optional positional question order for
                template placeholders.

        Raises:
            ValueError: If the configured field template resolves to duplicate
                keys for different options.
        """
        self.verbalized_options = list(options)
        json_fields: dict[str, str] = {}
        constraints: dict[str, str] = {}
        for option_index, option in enumerate(options, start=1):
            if question is None:
                field_name = option
            else:
                field_name = self._format_field(
                    option=option,
                    question=self._question_for_field(
                        question=question,
                        option_index=option_index,
                    ),
                    option_index=option_index,
                    question_id=question_id,
                    question_order=question_order,
                )
            if field_name in json_fields:
                raise ValueError(
                    "The configured `json_field_template` generates duplicate field names "
                    f"for option '{option}'. Please include placeholders that make fields unique, "
                    "for example `{question_order}`, `{question_id}`, and/or `{option_index}`."
                )
            json_fields[field_name] = self._format_explanation(
                option=option,
                question=question,
                option_index=option_index,
                question_id=question_id,
                question_order=question_order,
            )
            constraints[field_name] = "float"
        self.json_fields = json_fields
        self.constraints = constraints
