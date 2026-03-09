"""Persona creation utilities for building configurable prompt-ready profiles."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ..inference.survey_inference import batch_generation
from ..prompt_builder import LLMPrompt
from ..utilities import constants
from ..utilities.survey_objects import QuestionnaireItem

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from vllm import LLM  # pyright: ignore[reportMissingImports]

_MissingPolicy = Literal["skip", "placeholder"]
_SystemPromptMode = Literal["automatic", "interview", "human_llm", "interview_llm"]

DEFAULT_AUTOMATIC_TEMPLATE = "{attribute}: {value}"
DEFAULT_INTERVIEW_QUESTION_TEMPLATE = "What is your {attribute}?"
DEFAULT_INTERVIEW_QA_TEMPLATE = "Q: {question}\nA: {value}"
DEFAULT_HUMAN_PROMPT_TEMPLATE = (
    "Write a concise third-person persona description from the attributes below.\n"
    "Do not invent facts and avoid stereotypes.\n"
    "Keep it brief (3-5 sentences).\n"
    "Attributes:\n{attributes}"
)
DEFAULT_INTERVIEW_FROM_AUTOMATIC_PROMPT_TEMPLATE = (
    "Convert the persona attributes below into interview Q/A pairs.\n"
    "Goal: only improve grammar and phrasing of the questions.\n"
    "Hard rules:\n"
    "1. Keep exactly the same number of attributes as input.\n"
    "2. Keep the same order as input.\n"
    "3. Keep each answer value exactly as provided (verbatim, no paraphrasing).\n"
    "4. Do not add, remove, infer, or generalize any information.\n"
    "5. Output only repeated lines in this exact format:\n"
    "Q: <question>\n"
    "A: <answer>\n\n"
    "Input attributes:\n{automatic_persona}"
)


@dataclass(frozen=True)
class InterviewTurn:
    """Single interview-style persona turn."""

    question: str
    answer: str


@dataclass(frozen=True)
class LLMConfig:
    """Inference configuration used by persona LLM generation helpers."""

    system_message: str | None = "You are a helpful assistant."
    client_model_name: str | None = None
    api_concurrency: int = 10
    print_conversation: bool = False
    print_progress: bool = False
    seed: int = 42
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


class PersonaBuilder:
    """Build configurable persona text variants for prompt construction.

    The builder exposes three rendering styles:
    - deterministic attribute descriptions (`render_automatic`)
    - interview-style Q/A transcripts (`render_interview_text`)
    - LLM-generated short profile descriptions (`generate_human_description`)
    """

    def __init__(
        self,
        *,
        default_attribute_order: list[str] | None = None,
        default_attribute_templates: dict[str, str] | None = None,
        default_automatic_template: str = DEFAULT_AUTOMATIC_TEMPLATE,
        default_automatic_separator: str = "\n",
        default_interview_question_map: dict[str, str] | None = None,
        default_interview_question_template: str = DEFAULT_INTERVIEW_QUESTION_TEMPLATE,
        default_interview_qa_template: str = DEFAULT_INTERVIEW_QA_TEMPLATE,
        default_interview_separator: str = "\n",
        default_missing_value_policy: _MissingPolicy = "skip",
        default_missing_value_placeholder: str = "unknown",
        default_human_prompt_template: str = DEFAULT_HUMAN_PROMPT_TEMPLATE,
        default_true_text: str = "Yes",
        default_false_text: str = "No",
        default_cast_integer_like_floats: bool = True,
    ) -> None:
        """Initialize global defaults for persona rendering.

        Args:
            default_attribute_order: Optional default order used for attribute rendering.
                Attributes not listed here are appended in input order.
            default_attribute_templates: Optional per-attribute templates used by
                `render_automatic`. Templates may reference `{attribute}` and `{value}`.
            default_automatic_template: Fallback template for attributes without a
                dedicated entry in `default_attribute_templates`.
            default_automatic_separator: Separator used to join automatic-rendered lines.
            default_interview_question_map: Optional per-attribute interview questions.
            default_interview_question_template: Fallback interview question template
                when an attribute is not in `default_interview_question_map`.
            default_interview_qa_template: Template used to render each interview turn.
                Supports `{question}`, `{value}`, and `{answer}`.
            default_interview_separator: Separator used to join interview turns.
            default_missing_value_policy: Default behavior for missing values (`"skip"` or
                `"placeholder"`).
            default_missing_value_placeholder: Placeholder text used when missing-value policy
                is `"placeholder"`.
            default_human_prompt_template: Template used by `build_human_prompt`. Supports
                `{attributes}` where each line has `- name: value`.
            default_true_text: Text used when rendering `True` attribute values.
            default_false_text: Text used when rendering `False` attribute values.
            default_cast_integer_like_floats: If `True`, float-like values that are whole
                numbers (e.g., `19.0`) are rendered as integers (`19`).
        """
        self.default_attribute_order = list(default_attribute_order or [])
        self.default_attribute_templates = dict(default_attribute_templates or {})
        self.default_automatic_template = default_automatic_template
        self.default_automatic_separator = default_automatic_separator
        self.default_interview_question_map = dict(default_interview_question_map or {})
        self.default_interview_question_template = default_interview_question_template
        self.default_interview_qa_template = default_interview_qa_template
        self.default_interview_separator = default_interview_separator
        self.default_missing_value_policy = default_missing_value_policy
        self.default_missing_value_placeholder = default_missing_value_placeholder
        self.default_human_prompt_template = default_human_prompt_template
        self.default_true_text = default_true_text
        self.default_false_text = default_false_text
        self.default_cast_integer_like_floats = default_cast_integer_like_floats

    @staticmethod
    def build_interview_from_automatic_prompt(
        automatic_persona: str,
        *,
        prompt_template: str = DEFAULT_INTERVIEW_FROM_AUTOMATIC_PROMPT_TEMPLATE,
    ) -> str:
        """Build an LLM prompt that transforms automatic text into interview Q/A text.

        Args:
            automatic_persona: Automatic persona text, e.g., from `render_automatic`.
            prompt_template: Prompt template that supports `{automatic_persona}`.

        Returns:
            Prompt string to send to an LLM.
        """
        return prompt_template.format(automatic_persona=automatic_persona)

    def render_automatic(
        self,
        attributes: Mapping[str, Any],
        *,
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        attribute_templates: Mapping[str, str] | None = None,
        automatic_template: str | None = None,
        separator: str | None = None,
        prefix: str = "",
        suffix: str = "",
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
    ) -> str:
        """Render a deterministic persona description from attributes.

        Args:
            attributes: Attribute dictionary for a single persona.
            attribute_order: Optional explicit output order for attributes.
            attribute_aliases: Optional mapping from input attribute keys to output labels.
            attribute_templates: Optional per-attribute templates overriding builder defaults.
            automatic_template: Optional fallback template overriding the builder default.
            separator: Optional separator overriding the builder default.
            prefix: Optional prefix added before the rendered body.
            suffix: Optional suffix added after the rendered body.
            missing_value_policy: Optional override for missing-value behavior.
            missing_value_placeholder: Optional override placeholder used when policy is
                `"placeholder"`.
            true_text: Optional override text for `True` values.
            false_text: Optional override text for `False` values.
            cast_integer_like_floats: Optional override for rendering whole-number floats
                as integers.

        Returns:
            Rendered automatic persona text.

        Raises:
            ValueError: If `missing_value_policy` is not `"skip"` or `"placeholder"`.
        """
        rows = self._iter_renderable_attributes(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        template_map = dict(self.default_attribute_templates)
        if attribute_templates:
            template_map.update(attribute_templates)
        default_template = automatic_template or self.default_automatic_template
        rendered = [
            self._safe_format(
                template_map.get(name, default_template),
                attribute=name,
                value=value,
            )
            for name, value in rows
        ]
        return self._join_parts(
            rendered,
            separator=separator or self.default_automatic_separator,
            prefix=prefix,
            suffix=suffix,
        )

    def render_interview_text(
        self,
        attributes: Mapping[str, Any],
        *,
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        question_map: Mapping[str, str] | None = None,
        question_template: str | None = None,
        question_answer_template: str | None = None,
        separator: str | None = None,
        prefix: str = "",
        suffix: str = "",
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
    ) -> str:
        """Render persona attributes as interview-style question/answer text.

        Args:
            attributes: Attribute dictionary for a single persona.
            attribute_order: Optional explicit output order for attributes.
            attribute_aliases: Optional mapping from input attribute keys to output labels.
            question_map: Optional per-attribute question overrides.
            question_template: Optional fallback question template override.
            question_answer_template: Optional template override for each Q/A pair.
                Supports `{question}`, `{value}`, and `{answer}`.
            separator: Optional separator for joining rendered Q/A turns.
            prefix: Optional prefix added before the rendered body.
            suffix: Optional suffix added after the rendered body.
            missing_value_policy: Optional override for missing-value behavior.
            missing_value_placeholder: Optional override placeholder used when policy is
                `"placeholder"`.
            true_text: Optional override text for `True` values.
            false_text: Optional override text for `False` values.
            cast_integer_like_floats: Optional override for rendering whole-number floats
                as integers.

        Returns:
            Rendered interview persona transcript.

        Raises:
            ValueError: If `missing_value_policy` is not `"skip"` or `"placeholder"`.
        """
        turns = self._build_interview_turns(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            question_map=question_map,
            question_template=question_template,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        qa_template = question_answer_template or self.default_interview_qa_template
        rendered = [
            self._safe_format(
                qa_template,
                question=turn.question,
                value=turn.answer,
                answer=turn.answer,
            )
            for turn in turns
        ]
        return self._join_parts(
            rendered,
            separator=separator or self.default_interview_separator,
            prefix=prefix,
            suffix=suffix,
        )

    def build_human_prompt(
        self,
        attributes: Mapping[str, Any],
        *,
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        prompt_template: str | None = None,
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
    ) -> str:
        """Build the user prompt for LLM-based persona summarization.

        Args:
            attributes: Attribute dictionary for a single persona.
            attribute_order: Optional explicit output order for attributes.
            attribute_aliases: Optional mapping from input attribute keys to output labels.
            prompt_template: Optional prompt template override. Supports `{attributes}`.
            missing_value_policy: Optional override for missing-value behavior.
            missing_value_placeholder: Optional override placeholder used when policy is
                `"placeholder"`.
            true_text: Optional override text for `True` values.
            false_text: Optional override text for `False` values.
            cast_integer_like_floats: Optional override for rendering whole-number floats
                as integers.

        Returns:
            Prompt text instructing an LLM to generate a concise third-person profile.

        Raises:
            ValueError: If `missing_value_policy` is not `"skip"` or `"placeholder"`.
        """
        rows = self._iter_renderable_attributes(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        attribute_lines = [f"- {name}: {value}" for name, value in rows]
        template = prompt_template or self.default_human_prompt_template
        return self._safe_format(template, attributes="\n".join(attribute_lines))

    def generate_human_description(
        self,
        *,
        model: LLM | AsyncOpenAI,
        attributes: Mapping[str, Any],
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        prompt_template: str | None = None,
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
        llm_config: LLMConfig | None = None,
    ) -> str:
        """Generate a concise human-readable persona description via LLM.

        This helper builds a prompt with `build_human_prompt` and executes one
        `batch_generation` call.

        Args:
            model: Inference backend (`vllm.LLM` or `openai.AsyncOpenAI`).
            attributes: Attribute dictionary for a single persona.
            attribute_order: Optional explicit output order for attributes.
            attribute_aliases: Optional mapping from input attribute keys to output labels.
            prompt_template: Optional prompt template override. Supports `{attributes}`.
            missing_value_policy: Optional override for missing-value behavior.
            missing_value_placeholder: Optional override placeholder used when policy is
                `"placeholder"`.
            true_text: Optional override text for `True` values.
            false_text: Optional override text for `False` values.
            cast_integer_like_floats: Optional override for rendering whole-number floats
                as integers.
            llm_config: Optional inference config for generation settings such as
                system message, model id hint, concurrency, and generation kwargs.

        Returns:
            Generated human-readable persona description (single string).

        Raises:
            ValueError: If `missing_value_policy` is not `"skip"` or `"placeholder"`.
        """
        human_prompt = self.build_human_prompt(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            prompt_template=prompt_template,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        resolved_llm_config = llm_config or LLMConfig()
        outputs, _, _ = batch_generation(
            model=model,
            system_messages=[resolved_llm_config.system_message],
            prompts=[human_prompt],
            client_model_name=resolved_llm_config.client_model_name,
            api_concurrency=resolved_llm_config.api_concurrency,
            print_conversation=resolved_llm_config.print_conversation,
            print_progress=resolved_llm_config.print_progress,
            seed=resolved_llm_config.seed,
            **resolved_llm_config.generation_kwargs,
        )
        return outputs[0]

    def generate_interview_from_automatic(
        self,
        *,
        model: LLM | AsyncOpenAI,
        automatic_persona: str,
        llm_config: LLMConfig | None = None,
        prompt_template: str = DEFAULT_INTERVIEW_FROM_AUTOMATIC_PROMPT_TEMPLATE,
    ) -> str:
        """Generate interview Q/A text from automatic persona text via LLM.

        Args:
            model: Inference backend (`vllm.LLM` or `openai.AsyncOpenAI`).
            automatic_persona: Automatic persona text to convert.
            llm_config: Optional inference config for generation settings.
            prompt_template: Prompt template used for conversion.

        Returns:
            Generated interview Q/A text.
        """
        prompt = self.build_interview_from_automatic_prompt(
            automatic_persona=automatic_persona,
            prompt_template=prompt_template,
        )
        resolved_llm_config = llm_config or LLMConfig()
        outputs, _, _ = batch_generation(
            model=model,
            system_messages=[resolved_llm_config.system_message],
            prompts=[prompt],
            client_model_name=resolved_llm_config.client_model_name,
            api_concurrency=resolved_llm_config.api_concurrency,
            print_conversation=resolved_llm_config.print_conversation,
            print_progress=resolved_llm_config.print_progress,
            seed=resolved_llm_config.seed,
            **resolved_llm_config.generation_kwargs,
        )
        return outputs[0]

    def build_persona_dataframe(
        self,
        personas: pd.DataFrame,
        *,
        attribute_columns: Sequence[str] | None = None,
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        attribute_templates: Mapping[str, str] | None = None,
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
        question_map: Mapping[str, str] | None = None,
        question_template: str | None = None,
        automatic_kwargs: Mapping[str, Any] | None = None,
        interview_kwargs: Mapping[str, Any] | None = None,
        human_prompt_kwargs: Mapping[str, Any] | None = None,
        include_human_llm: bool = False,
        include_interview_llm: bool = False,
        reuse_interview_questions: bool = True,
        model: LLM | AsyncOpenAI | None = None,
        human_prompt_template: str | None = None,
        interview_llm_prompt_template: str = DEFAULT_INTERVIEW_FROM_AUTOMATIC_PROMPT_TEMPLATE,
        default_system_prompt_mode: _SystemPromptMode = "automatic",
        llm_config: LLMConfig | None = None,
    ) -> pd.DataFrame:
        """Build persona outputs for each row in a dataframe.

        Created columns:
        - `persona_automatic`
        - `persona_interview`
        - `persona_human_prompt`
        - `persona_human_llm` (only if `include_human_llm=True`)
        - `persona_interview_llm` (only if `include_interview_llm=True`)
        - `system_prompt` (mapped from `default_system_prompt_mode`)

        Args:
            personas: Input dataframe where each row is one persona record.
            attribute_columns: Optional subset of columns used as persona attributes.
                Defaults to all columns from `personas`.
            attribute_order: Optional explicit output order for attributes.
            attribute_aliases: Optional mapping from input attribute keys to output labels.
            attribute_templates: Optional per-attribute templates for automatic mode.
            missing_value_policy: Optional override for missing-value behavior.
            missing_value_placeholder: Optional override placeholder used when policy is
                `"placeholder"`.
            true_text: Optional override text for `True` values.
            false_text: Optional override text for `False` values.
            cast_integer_like_floats: Optional override for rendering whole-number floats
                as integers.
            question_map: Optional per-attribute interview question overrides.
            question_template: Optional fallback interview question template override.
            automatic_kwargs: Optional kwargs bundle forwarded to `render_automatic`.
            interview_kwargs: Optional kwargs bundle forwarded to `render_interview_text`.
            human_prompt_kwargs: Optional kwargs bundle forwarded to `build_human_prompt`.
            include_human_llm: If `True`, execute LLM generation and add
                `persona_human_llm`.
            include_interview_llm: If `True`, execute LLM conversion and add
                `persona_interview_llm` from `persona_automatic`.
            reuse_interview_questions: If `True`, question templates for interview mode are
                resolved once for all rows and only answers vary by row.
            model: Inference backend required when `include_human_llm=True`.
            human_prompt_template: Optional template override for `persona_human_prompt`.
            interview_llm_prompt_template: Prompt template used to convert
                `persona_automatic` into interview Q/A text.
            default_system_prompt_mode: Persona mode used to populate `system_prompt`.
            llm_config: Optional inference config used when `include_human_llm=True`.

        Returns:
            A copy of `personas` augmented with generated persona text columns.

        Raises:
            ValueError: If an LLM-generated output is requested and `model` is not provided.
            ValueError: If `missing_value_policy` is not `"skip"` or `"placeholder"`.
            ValueError: If `default_system_prompt_mode` cannot be mapped because its
                source column is not present.
        """
        selected_columns = list(attribute_columns or personas.columns.tolist())
        output = personas.copy()

        automatic_values: list[str] = []
        interview_values: list[str] = []
        human_prompts: list[str] = []

        automatic_call_kwargs: dict[str, Any] = {
            "attribute_order": attribute_order,
            "attribute_aliases": attribute_aliases,
            "attribute_templates": attribute_templates,
            "missing_value_policy": missing_value_policy,
            "missing_value_placeholder": missing_value_placeholder,
            "true_text": true_text,
            "false_text": false_text,
            "cast_integer_like_floats": cast_integer_like_floats,
        }
        if automatic_kwargs:
            automatic_call_kwargs.update(dict(automatic_kwargs))

        interview_call_kwargs: dict[str, Any] = {
            "attribute_order": attribute_order,
            "attribute_aliases": attribute_aliases,
            "question_map": question_map,
            "question_template": question_template,
            "missing_value_policy": missing_value_policy,
            "missing_value_placeholder": missing_value_placeholder,
            "true_text": true_text,
            "false_text": false_text,
            "cast_integer_like_floats": cast_integer_like_floats,
        }
        if interview_kwargs:
            interview_call_kwargs.update(dict(interview_kwargs))

        human_prompt_call_kwargs: dict[str, Any] = {
            "attribute_order": attribute_order,
            "attribute_aliases": attribute_aliases,
            "prompt_template": human_prompt_template,
            "missing_value_policy": missing_value_policy,
            "missing_value_placeholder": missing_value_placeholder,
            "true_text": true_text,
            "false_text": false_text,
            "cast_integer_like_floats": cast_integer_like_floats,
        }
        if human_prompt_kwargs:
            human_prompt_call_kwargs.update(dict(human_prompt_kwargs))

        interview_question_templates: dict[str, str] | None = None
        if reuse_interview_questions:
            interview_question_templates = self._resolve_interview_question_templates(
                attribute_names=selected_columns,
                attribute_order=interview_call_kwargs.get("attribute_order"),
                attribute_aliases=interview_call_kwargs.get("attribute_aliases"),
                question_map=interview_call_kwargs.get("question_map"),
                question_template=interview_call_kwargs.get("question_template"),
            )

        for _, row in personas.iterrows():
            attrs = {column: row[column] for column in selected_columns}
            automatic_values.append(
                self.render_automatic(
                    attrs,
                    **automatic_call_kwargs,
                )
            )
            if interview_question_templates is None:
                interview_values.append(
                    self.render_interview_text(
                        attrs,
                        **interview_call_kwargs,
                    )
                )
            else:
                interview_values.append(
                    self._render_interview_text_with_templates(
                        attrs,
                        question_templates=interview_question_templates,
                        attribute_order=interview_call_kwargs.get("attribute_order"),
                        attribute_aliases=interview_call_kwargs.get("attribute_aliases"),
                        question_answer_template=interview_call_kwargs.get(
                            "question_answer_template"
                        ),
                        separator=interview_call_kwargs.get("separator"),
                        prefix=interview_call_kwargs.get("prefix", ""),
                        suffix=interview_call_kwargs.get("suffix", ""),
                        missing_value_policy=interview_call_kwargs.get("missing_value_policy"),
                        missing_value_placeholder=interview_call_kwargs.get(
                            "missing_value_placeholder"
                        ),
                        true_text=interview_call_kwargs.get("true_text"),
                        false_text=interview_call_kwargs.get("false_text"),
                        cast_integer_like_floats=interview_call_kwargs.get(
                            "cast_integer_like_floats"
                        ),
                    )
                )
            human_prompts.append(
                self.build_human_prompt(
                    attrs,
                    **human_prompt_call_kwargs,
                )
            )

        output["persona_automatic"] = automatic_values
        output["persona_interview"] = interview_values
        output["persona_human_prompt"] = human_prompts

        if include_human_llm:
            if model is None:
                raise ValueError("`model` must be provided when `include_human_llm=True`.")
            resolved_llm_config = llm_config or LLMConfig()
            human_llm_values, _, _ = batch_generation(
                model=model,
                system_messages=[resolved_llm_config.system_message] * len(human_prompts),
                prompts=human_prompts,
                client_model_name=resolved_llm_config.client_model_name,
                api_concurrency=resolved_llm_config.api_concurrency,
                print_conversation=resolved_llm_config.print_conversation,
                print_progress=resolved_llm_config.print_progress,
                seed=resolved_llm_config.seed,
                **resolved_llm_config.generation_kwargs,
            )
            output["persona_human_llm"] = human_llm_values

        if include_interview_llm:
            if model is None:
                raise ValueError("`model` must be provided when `include_interview_llm=True`.")
            resolved_llm_config = llm_config or LLMConfig()
            interview_prompts = [
                self.build_interview_from_automatic_prompt(
                    automatic_persona=text,
                    prompt_template=interview_llm_prompt_template,
                )
                for text in automatic_values
            ]
            interview_llm_values, _, _ = batch_generation(
                model=model,
                system_messages=[resolved_llm_config.system_message] * len(interview_prompts),
                prompts=interview_prompts,
                client_model_name=resolved_llm_config.client_model_name,
                api_concurrency=resolved_llm_config.api_concurrency,
                print_conversation=resolved_llm_config.print_conversation,
                print_progress=resolved_llm_config.print_progress,
                seed=resolved_llm_config.seed,
                **resolved_llm_config.generation_kwargs,
            )
            output["persona_interview_llm"] = interview_llm_values

        return self.apply_system_prompt_column(
            output,
            mode=default_system_prompt_mode,
            output_column=constants.SYSTEM_PROMPT_FIELD,
        )

    def apply_system_prompt_column(
        self,
        dataframe: pd.DataFrame,
        *,
        mode: _SystemPromptMode = "automatic",
        output_column: str = constants.SYSTEM_PROMPT_FIELD,
    ) -> pd.DataFrame:
        """Create or overwrite a system-prompt column from one persona mode.

        Args:
            dataframe: Dataframe containing persona mode columns.
            mode: Source mode (`automatic`, `interview`, `human_llm`, `interview_llm`).
            output_column: Name of the output column to write.

        Returns:
            Copy of `dataframe` with `output_column` set from the selected mode.

        Raises:
            ValueError: If the source persona column for `mode` is missing.
        """
        mode_column = {
            "automatic": "persona_automatic",
            "interview": "persona_interview",
            "human_llm": "persona_human_llm",
            "interview_llm": "persona_interview_llm",
        }[mode]
        if mode_column not in dataframe.columns:
            raise ValueError(
                f"Cannot set system prompt from mode '{mode}'. Missing column '{mode_column}'."
            )
        output = dataframe.copy()
        output[output_column] = output[mode_column]
        return output

    def apply_interview_as_prefilled_turns(
        self,
        llm_prompt: LLMPrompt,
        attributes: Mapping[str, Any],
        *,
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        question_map: Mapping[str, str] | None = None,
        question_template: str | None = None,
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
        item_id_prefix: str = "__persona_interview__",
    ) -> LLMPrompt:
        """Prepend interview turns as synthetic prefilled questionnaire items.

        The returned prompt is a deep copy of `llm_prompt`; the original object is
        not modified.

        Args:
            llm_prompt: Base prompt containing the actual survey questionnaire.
            attributes: Attribute dictionary used to build persona interview turns.
            attribute_order: Optional explicit output order for attributes.
            attribute_aliases: Optional mapping from input attribute keys to output labels.
            question_map: Optional per-attribute question overrides.
            question_template: Optional fallback question template override.
            missing_value_policy: Optional override for missing-value behavior.
            missing_value_placeholder: Optional override placeholder used when policy is
                `"placeholder"`.
            true_text: Optional override text for `True` values.
            false_text: Optional override text for `False` values.
            cast_integer_like_floats: Optional override for rendering whole-number floats
                as integers.
            item_id_prefix: Prefix for synthetic persona item IDs.

        Returns:
            A duplicated `LLMPrompt` with interview seed turns inserted at position `0`.

        Raises:
            ValueError: If `missing_value_policy` is not `"skip"` or `"placeholder"`.
        """
        duplicated = llm_prompt.duplicate()
        turns = self._build_interview_turns(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            question_map=question_map,
            question_template=question_template,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        existing_ids = {str(item.item_id) for item in duplicated.get_questions()}
        synthetic_items: list[QuestionnaireItem] = []
        for idx, turn in enumerate(turns):
            item_id = f"{item_id_prefix}{idx}"
            while item_id in existing_ids:
                item_id += "_"
            existing_ids.add(item_id)
            synthetic_items.append(
                QuestionnaireItem(
                    item_id=item_id,
                    question_content=turn.question,
                    prefilled_response=turn.answer,
                )
            )
        duplicated.insert_questions(synthetic_items, position=0)
        return duplicated

    def _resolve_interview_question_templates(
        self,
        *,
        attribute_names: Sequence[str],
        attribute_order: Sequence[str] | None,
        attribute_aliases: Mapping[str, str] | None,
        question_map: Mapping[str, str] | None,
        question_template: str | None,
    ) -> dict[str, str]:
        """Resolve per-attribute interview question templates once."""
        ordered_names = self._resolve_attribute_order(
            {attribute_name: None for attribute_name in attribute_names},
            attribute_order,
        )
        aliases = dict(attribute_aliases or {})
        merged_map = dict(self.default_interview_question_map)
        if question_map:
            merged_map.update(question_map)
        for original, alias in aliases.items():
            if original in merged_map and alias not in merged_map:
                merged_map[alias] = merged_map[original]
        default_template = question_template or self.default_interview_question_template
        resolved: dict[str, str] = {}
        for input_name in ordered_names:
            output_name = aliases.get(input_name, input_name)
            mapped_template = merged_map.get(input_name, default_template)
            resolved[output_name] = merged_map.get(output_name, mapped_template)
        return resolved

    def _render_interview_text_with_templates(
        self,
        attributes: Mapping[str, Any],
        *,
        question_templates: Mapping[str, str],
        attribute_order: Sequence[str] | None,
        attribute_aliases: Mapping[str, str] | None,
        question_answer_template: str | None,
        separator: str | None,
        prefix: str,
        suffix: str,
        missing_value_policy: _MissingPolicy | None,
        missing_value_placeholder: str | None,
        true_text: str | None,
        false_text: str | None,
        cast_integer_like_floats: bool | None,
    ) -> str:
        rows = self._iter_renderable_attributes(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        default_template = self.default_interview_question_template
        qa_template = question_answer_template or self.default_interview_qa_template
        rendered = []
        for attribute, value in rows:
            question_template = question_templates.get(attribute, default_template)
            question = self._safe_format(
                question_template,
                attribute=attribute,
                value=value,
            )
            rendered.append(
                self._safe_format(
                    qa_template,
                    question=question,
                    value=value,
                    answer=value,
                )
            )
        return self._join_parts(
            rendered,
            separator=separator or self.default_interview_separator,
            prefix=prefix,
            suffix=suffix,
        )

    def _build_interview_turns(
        self,
        *,
        attributes: Mapping[str, Any],
        attribute_order: Sequence[str] | None = None,
        attribute_aliases: Mapping[str, str] | None = None,
        question_map: Mapping[str, str] | None = None,
        question_template: str | None = None,
        missing_value_policy: _MissingPolicy | None = None,
        missing_value_placeholder: str | None = None,
        true_text: str | None = None,
        false_text: str | None = None,
        cast_integer_like_floats: bool | None = None,
    ) -> list[InterviewTurn]:
        rows = self._iter_renderable_attributes(
            attributes=attributes,
            attribute_order=attribute_order,
            attribute_aliases=attribute_aliases,
            missing_value_policy=missing_value_policy,
            missing_value_placeholder=missing_value_placeholder,
            true_text=true_text,
            false_text=false_text,
            cast_integer_like_floats=cast_integer_like_floats,
        )
        merged_map = dict(self.default_interview_question_map)
        if question_map:
            merged_map.update(question_map)
        if attribute_aliases:
            for original, alias in attribute_aliases.items():
                if original in merged_map and alias not in merged_map:
                    merged_map[alias] = merged_map[original]
        template = question_template or self.default_interview_question_template
        turns = []
        for attribute, value in rows:
            question = self._safe_format(
                merged_map.get(attribute, template),
                attribute=attribute,
                value=value,
            )
            turns.append(InterviewTurn(question=question, answer=str(value)))
        return turns

    def _iter_renderable_attributes(
        self,
        *,
        attributes: Mapping[str, Any],
        attribute_order: Sequence[str] | None,
        attribute_aliases: Mapping[str, str] | None,
        missing_value_policy: _MissingPolicy | None,
        missing_value_placeholder: str | None,
        true_text: str | None,
        false_text: str | None,
        cast_integer_like_floats: bool | None,
    ) -> list[tuple[str, Any]]:
        policy = missing_value_policy or self.default_missing_value_policy
        if policy not in {"skip", "placeholder"}:
            raise ValueError("`missing_value_policy` must be either 'skip' or 'placeholder'.")
        placeholder = (
            missing_value_placeholder
            if missing_value_placeholder is not None
            else self.default_missing_value_placeholder
        )
        resolved_true = self.default_true_text if true_text is None else true_text
        resolved_false = self.default_false_text if false_text is None else false_text
        cast_int_like = (
            self.default_cast_integer_like_floats
            if cast_integer_like_floats is None
            else cast_integer_like_floats
        )
        aliases = dict(attribute_aliases or {})
        resolved_order = self._resolve_attribute_order(attributes, attribute_order)
        pairs: list[tuple[str, Any]] = []
        for name in resolved_order:
            value = attributes.get(name)
            output_name = aliases.get(name, name)
            if self._is_missing(value):
                if policy == "skip":
                    continue
                value = placeholder
            else:
                value = self._normalize_attribute_value(
                    value,
                    true_text=resolved_true,
                    false_text=resolved_false,
                    cast_integer_like_floats=cast_int_like,
                )
            pairs.append((output_name, value))
        return pairs

    def _resolve_attribute_order(
        self,
        attributes: Mapping[str, Any],
        attribute_order: Sequence[str] | None,
    ) -> list[str]:
        base_order = list(attribute_order or self.default_attribute_order)
        if not base_order:
            return list(attributes.keys())
        seen = set(base_order)
        remainder = [key for key in attributes.keys() if key not in seen]
        return [key for key in base_order if key in attributes] + remainder

    @staticmethod
    def _join_parts(parts: list[str], *, separator: str, prefix: str, suffix: str) -> str:
        body = separator.join(parts)
        return f"{prefix}{body}{suffix}"

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, dict, tuple, set)):
            return False
        return bool(pd.isna(value))

    @staticmethod
    def _normalize_attribute_value(
        value: Any,
        *,
        true_text: str,
        false_text: str,
        cast_integer_like_floats: bool,
    ) -> Any:
        if isinstance(value, bool):
            return true_text if value else false_text
        if cast_integer_like_floats:
            is_real = isinstance(value, numbers.Real)
            is_integral = isinstance(value, numbers.Integral)
            if is_real and not is_integral:
                as_float = float(value)
                if as_float.is_integer():
                    return int(as_float)
        return value

    @staticmethod
    def _safe_format(template: str, **values: Any) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return template.format_map(_SafeDict(values))
