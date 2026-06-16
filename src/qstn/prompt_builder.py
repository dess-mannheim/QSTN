import copy
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from string import ascii_lowercase, ascii_uppercase
from typing import Any, Literal, Self, overload

import pandas as pd

from ._questionnaire_loader import (
    QuestionnaireLoaderColumn,
    optional_bool,
    optional_int,
    optional_list,
    optional_row_value,
    optional_template,
    row_has_value,
)
from .inference.multimodal import (
    ImageInput,
    ImageSource,
    PromptContent,
    PromptContentBlock,
    combine_prompt_content,
    format_prompt_content,
    normalize_images,
    prompt_content_text,
)
from .inference.response_generation import (
    ChoiceResponseGenerationMethod,
    JSONReasoningResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
    LogprobResponseGenerationMethod,
    ResponseGenerationMethod,
    resolve_battery_response_generation_method,
)
from .inference.utils import InferenceMode
from .utilities import constants, placeholder, prompt_templates
from .utilities.constants import QuestionnairePresentation
from .utilities.survey_objects import AnswerOptions, AnswerTexts, QuestionnaireItem
from .utilities.utils import safe_format_with_regex


class ResponseGenerationPreset(StrEnum):
    """Named response-generation methods supported by questionnaire loading."""

    NONE = "none"
    CHOICE = "choice"
    LOGPROB = "logprob"
    JSON_SINGLE = "json_single"
    JSON_REASONING = "json_reasoning"
    JSON_DISTRIBUTION = "json_distribution"


@dataclass(frozen=True)
class BaseModelPromptTemplate:
    """Template used to render chat-style turns for base-model prompts."""

    user_prefix: str | None = None
    assistant_prefix: str | None = None
    separator: str = "\n"
    system_prefix: str | None = None


def _render_prefixed(prefix: str | None, content: str) -> str:
    """Render a single prompt block, preserving empty prefixes and content."""
    if prefix is None:
        return content
    return f"{prefix}\n{content}"


def messages_to_base_model_prompt(
    messages: Sequence[dict[str, str]],
    prompt_template: BaseModelPromptTemplate | None = None,
) -> str:
    """Render chat-style messages into a plain prompt for base models."""
    template = prompt_template or BaseModelPromptTemplate()
    blocks: list[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            blocks.append(_render_prefixed(template.system_prefix, content))
        elif role == "user":
            blocks.append(_render_prefixed(template.user_prefix, content))
        elif role == "assistant":
            blocks.append(_render_prefixed(template.assistant_prefix, content))
        else:
            raise ValueError(f"Unsupported message role for base-model rendering: {role}")

    if template.assistant_prefix is not None:
        blocks.append(template.assistant_prefix)
    return template.separator.join(blocks)


def _build_response_generation_method(
    row: pd.Series,
    item_id: Any,
) -> ResponseGenerationMethod | None:
    column = QuestionnaireLoaderColumn.RESPONSE_GENERATION_METHOD
    value = optional_row_value(row, column)
    if value is None:
        return None
    if isinstance(value, ResponseGenerationMethod):
        return value

    preset_value = str(value).strip().lower()
    try:
        preset = ResponseGenerationPreset(preset_value)
    except ValueError as exc:
        supported = ", ".join(preset.value for preset in ResponseGenerationPreset)
        raise ValueError(
            f"Unsupported response_generation_method '{value}' for questionnaire_item_id "
            f"'{item_id}'. Supported presets are: {supported}."
        ) from exc

    if preset == ResponseGenerationPreset.NONE:
        return None

    output_index_only = optional_bool(
        row,
        QuestionnaireLoaderColumn.OUTPUT_INDEX_ONLY,
        item_id,
        default=False,
    )
    constrain_answer_options = optional_bool(
        row,
        QuestionnaireLoaderColumn.CONSTRAIN_ANSWER_OPTIONS,
        item_id,
        default=True,
    )

    if preset == ResponseGenerationPreset.CHOICE:
        return ChoiceResponseGenerationMethod(
            output_index_only=output_index_only,
            constrain_answer_options=constrain_answer_options,
        )
    if preset == ResponseGenerationPreset.LOGPROB:
        return LogprobResponseGenerationMethod(
            output_index_only=output_index_only,
            constrain_answer_options=constrain_answer_options,
        )
    if preset == ResponseGenerationPreset.JSON_SINGLE:
        return JSONSingleResponseGenerationMethod(
            output_index_only=output_index_only,
            constrain_answer_options=constrain_answer_options,
        )
    if preset == ResponseGenerationPreset.JSON_REASONING:
        return JSONReasoningResponseGenerationMethod(
            output_index_only=output_index_only,
            constrain_answer_options=constrain_answer_options,
        )
    if preset == ResponseGenerationPreset.JSON_DISTRIBUTION:
        return JSONVerbalizedDistribution(output_index_only=output_index_only)

    return None


def _has_likert_config(row: pd.Series) -> bool:
    return any(
        row_has_value(row, column)
        for column in QuestionnaireLoaderColumn
        if column.value.startswith("likert_")
    )


def _build_answer_options_from_row(row: pd.Series, item_id: Any) -> AnswerOptions | None:
    answer_texts = optional_list(row, QuestionnaireLoaderColumn.ANSWER_TEXTS, item_id)
    answer_codes = optional_list(row, QuestionnaireLoaderColumn.ANSWER_CODES, item_id)
    response_generation_method = _build_response_generation_method(row, item_id)
    list_prompt_template = optional_template(
        row,
        QuestionnaireLoaderColumn.LIST_PROMPT_TEMPLATE,
        prompt_templates.LIST_OPTIONS_DEFAULT,
    )
    scale_prompt_template = optional_template(
        row,
        QuestionnaireLoaderColumn.SCALE_PROMPT_TEMPLATE,
        prompt_templates.SCALE_OPTIONS_DEFAULT,
    )
    index_answer_separator = optional_template(
        row,
        QuestionnaireLoaderColumn.INDEX_ANSWER_SEPARATOR,
        ": ",
    )
    options_separator = optional_template(row, QuestionnaireLoaderColumn.OPTIONS_SEPARATOR, ", ")

    if _has_likert_config(row):
        only_from_to_scale = optional_bool(
            row,
            QuestionnaireLoaderColumn.LIKERT_ONLY_FROM_TO_SCALE,
            item_id,
            default=False,
        )
        explicit_n = optional_int(row, QuestionnaireLoaderColumn.LIKERT_N, item_id)
        if explicit_n is None:
            if only_from_to_scale:
                raise ValueError(
                    f"Column '{QuestionnaireLoaderColumn.LIKERT_N}' is required for "
                    f"from-to Likert scales on questionnaire_item_id '{item_id}'."
                )
            if answer_texts is None:
                raise ValueError(
                    f"Column '{QuestionnaireLoaderColumn.LIKERT_N}' is required when "
                    f"'{QuestionnaireLoaderColumn.ANSWER_TEXTS}' is missing for "
                    f"questionnaire_item_id '{item_id}'."
                )
            n = len(answer_texts)
        else:
            n = explicit_n

        idx_type = str(
            optional_row_value(row, QuestionnaireLoaderColumn.LIKERT_IDX_TYPE, "integer")
        )
        if idx_type not in {"char_lower", "char_upper", "integer", "no_index"}:
            raise ValueError(
                f"Column '{QuestionnaireLoaderColumn.LIKERT_IDX_TYPE}' for "
                f"questionnaire_item_id '{item_id}' must be one of: "
                "char_lower, char_upper, integer, no_index."
            )

        return generate_likert_options(
            n=n,
            answer_texts=answer_texts,
            only_from_to_scale=only_from_to_scale,
            random_order=optional_bool(
                row,
                QuestionnaireLoaderColumn.LIKERT_RANDOM_ORDER,
                item_id,
                default=False,
            ),
            reversed_order=optional_bool(
                row,
                QuestionnaireLoaderColumn.LIKERT_REVERSED_ORDER,
                item_id,
                default=False,
            ),
            even_order=optional_bool(
                row,
                QuestionnaireLoaderColumn.LIKERT_EVEN_ORDER,
                item_id,
                default=False,
            ),
            add_middle_category=optional_bool(
                row,
                QuestionnaireLoaderColumn.LIKERT_ADD_MIDDLE_CATEGORY,
                item_id,
                default=False,
            ),
            str_middle_cat=str(
                optional_row_value(
                    row,
                    QuestionnaireLoaderColumn.LIKERT_MIDDLE_CATEGORY,
                    "Neutral",
                )
            ),
            add_refusal=optional_bool(
                row,
                QuestionnaireLoaderColumn.LIKERT_ADD_REFUSAL,
                item_id,
                default=False,
            ),
            refusal_code=str(
                optional_row_value(row, QuestionnaireLoaderColumn.LIKERT_REFUSAL_CODE, "-99")
            ),
            start_idx=optional_int(
                row,
                QuestionnaireLoaderColumn.LIKERT_START_IDX,
                item_id,
                default=1,
            ),
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            index_answer_separator=index_answer_separator,
            options_separator=options_separator,
            idx_type=idx_type,
            response_generation_method=response_generation_method,
        )

    if answer_texts is None and answer_codes is None:
        if response_generation_method is not None:
            raise ValueError(
                f"questionnaire_item_id '{item_id}' defines a response_generation_method "
                "but no answer_texts or answer_codes."
            )
        return None

    if (
        answer_texts is not None
        and answer_codes is not None
        and len(answer_texts) != len(answer_codes)
    ):
        raise ValueError(
            f"answer_texts and answer_codes must have the same length for "
            f"questionnaire_item_id '{item_id}'."
        )

    answer_texts_object = AnswerTexts(
        answer_texts=answer_texts,
        indices=answer_codes,
        index_answer_seperator=index_answer_separator,
        option_seperators=options_separator,
    )
    return AnswerOptions(
        answer_texts=answer_texts_object,
        list_prompt_template=list_prompt_template,
        scale_prompt_template=scale_prompt_template,
        response_generation_method=response_generation_method,
    )


class LLMPrompt:
    """
    Main class for setting up and managing the prompt in the LLM experiment.

    This class handles loading questions
        from a predefined questionnaire, preparing prompts, managing answer options,
        and generating prompt structures for different interview types.
    """

    DEFAULT_QUESTIONNAIRE_ID: str = "Questionnaire"

    DEFAULT_SYSTEM_PROMPT: str = (
        "You will be given questions and possible answer options for each. "
        "Please reason about each question before answering."
    )
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: list[str] = ["reasoning", "answer"]

    DEFAULT_PROMPT_STRUCTURE: str = f"{placeholder.PROMPT_QUESTIONS}\n{placeholder.PROMPT_OPTIONS}"

    def __init__(
        self,
        questionnaire_source: str | pd.DataFrame = None,
        questionnaire_name: str = DEFAULT_QUESTIONNAIRE_ID,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        prompt: str = DEFAULT_PROMPT_STRUCTURE,
        verbose: bool = False,
        seed: int = 42,
    ):
        """
        Initialize an LLMPrompt instance. Either a path to a csv file
            or a pandas dataframe can be provided to structure the questionnaire.
        Question structure can later be modified with explicit methods such as
        `insert_questions`, `replace_question`, and `remove_question`.

        Args:
            questionnaire_source (str/pd.Dataframe): Path to the CSV file containing the
                questionnaire structure and questions.
            questionnaire_name (str): Name/ID for the questionnaire.
            system_prompt (str | None): System prompt for all questions.
                Set to `None` to omit a system message.
            prompt (str): Prompt for all questions.
            verbose (bool): Deprecated. Use `qstn.logger.configure_logging`
                to enable logging output.
            seed (int): Random seed for reproducibility.
        """
        if verbose:
            warnings.warn(
                "`verbose` is deprecated and will be removed in a future release. "
                "Use `qstn.logger.configure_logging` to enable logging output.",
                DeprecationWarning,
                stacklevel=2,
            )

        random.seed(seed)

        self._questions: list[QuestionnaireItem] = []
        self._images: tuple[ImageInput, ...] = ()
        self._item_images: dict[Any, tuple[ImageInput, ...]] = {}

        if self._check_valid_questionnaire(questionnaire_source):
            self.load_questionnaire_format(questionnaire_source=questionnaire_source)

        self.verbose: bool = verbose

        self.questionnaire_name: str = questionnaire_name

        self.system_prompt: str | None = system_prompt
        self.prompt: str = prompt
        self.base_model_prompt_template: BaseModelPromptTemplate | None = None

    def _check_valid_questionnaire(self, questionnaire_source: str | pd.DataFrame = None) -> bool:
        # No Object
        if questionnaire_source is None:
            return False

        # Empty String
        if isinstance(questionnaire_source, str) and not questionnaire_source:
            return False

        # Empty Dataframe
        if isinstance(questionnaire_source, pd.DataFrame):
            if questionnaire_source.empty:
                warnings.warn(
                    "The provided Dataframe is empty! No questions are created.", stacklevel=2
                )
                return False
            # Optional check if the correct columns are provided?
            # Would probably be nice to have that warning here.

        return True

    def duplicate(self):
        """
        Create a deep copy of the current interview instance.

        Returns:
            LLMQuestionnaire: A deep copy of the current object.
        """
        return copy.deepcopy(self)

    def add_image(self, image: ImageSource, *, item_id: Any = None) -> Self:
        """Add an image globally or to one questionnaire item.

        Args:
            image: Image input, URL, data URL, or local image path.
            item_id: Questionnaire item receiving the image. If omitted, the image
                applies to the full prompt.

        Returns:
            LLMPrompt: The current prompt object for fluent configuration.
        """
        normalized_image = normalize_images([image])[0]
        if item_id is None:
            self._images = (*self._images, normalized_image)
            return self

        self._validate_image_item_id(item_id)
        self._item_images[item_id] = (*self._item_images.get(item_id, ()), normalized_image)
        return self

    def set_images(
        self,
        images: Sequence[ImageSource],
        *,
        item_id: Any = None,
    ) -> Self:
        """Replace global images or the images for one questionnaire item.

        Args:
            images: Images, URLs, data URLs, or local image paths to store.
            item_id: Questionnaire item receiving the images. If omitted, replaces
                prompt-wide images.

        Returns:
            LLMPrompt: The current prompt object for fluent configuration.
        """
        normalized_image_inputs = normalize_images(images)
        if item_id is None:
            self._images = normalized_image_inputs
            return self

        self._validate_image_item_id(item_id)
        if normalized_image_inputs:
            self._item_images[item_id] = normalized_image_inputs
        else:
            self._item_images.pop(item_id, None)
        return self

    def get_images(
        self,
        *,
        item_id: Any = None,
        include_global: bool = True,
    ) -> tuple[ImageInput, ...]:
        """Return prompt-wide and optionally item-specific images.

        Args:
            item_id: Questionnaire item whose images should be included.
            include_global: Whether prompt-wide images should be returned first.

        Returns:
            tuple[ImageInput, ...]: Immutable image collection.
        """
        global_images = self._images if include_global else ()
        if item_id is None:
            return global_images
        self._validate_image_item_id(item_id)
        return (*global_images, *self._item_images.get(item_id, ()))

    def _validate_image_item_id(self, item_id: Any) -> None:
        if item_id not in {question.item_id for question in self._questions}:
            raise ValueError(
                f"Cannot attach images: questionnaire item '{item_id}' does not exist."
            )

    def _drop_stale_item_images(self) -> None:
        valid_item_ids = {question.item_id for question in self._questions}
        self._item_images = {
            item_id: images
            for item_id, images in self._item_images.items()
            if item_id in valid_item_ids
        }

    def set_base_model_prompt_template(
        self,
        template: BaseModelPromptTemplate | None = None,
        user_prefix: str | None = None,
        assistant_prefix: str | None = None,
        separator: str = "\n",
        system_prefix: str | None = None,
    ) -> Self:
        """Set the template used when rendering prompts for base-model completion mode.

        Args:
            template (BaseModelPromptTemplate | None): Existing template object to store.
            user_prefix (str | None): Prefix placed before each user turn.
            assistant_prefix (str | None): Prefix placed before assistant turns and final cue.
            separator (str): Text inserted between rendered conversation blocks.
            system_prefix (str | None): Optional prefix placed before the system prompt.

        Returns:
            LLMPrompt: The current prompt object for fluent configuration.
        """
        if template is not None:
            self.base_model_prompt_template = template
        else:
            self.base_model_prompt_template = BaseModelPromptTemplate(
                user_prefix=user_prefix,
                assistant_prefix=assistant_prefix,
                separator=separator,
                system_prefix=system_prefix,
            )
        return self

    def render_base_model_prompt(
        self,
        system_message: str | None,
        prompts: list[str],
        assistant_messages: list[str] | None = None,
    ) -> str:
        """Render chat-style turns into the exact prompt used for base-model generation.

        Args:
            system_message (str | None): Optional system text to place before the turns.
            prompts (list[str]): User turns to render.
            assistant_messages (list[str] | None): Assistant history between user turns.

        Returns:
            str: Rendered base-model prompt.
        """
        messages = []
        if system_message is not None:
            messages.append({"role": "system", "content": system_message})

        assistant_messages = assistant_messages or []
        for index, prompt in enumerate(prompts):
            messages.append({"role": "user", "content": prompt})
            if index < len(assistant_messages):
                messages.append({"role": "assistant", "content": assistant_messages[index]})

        return messages_to_base_model_prompt(messages, self.base_model_prompt_template)

    def _resolve_prompt_question(
        self,
        item_id: str | int | None,
        item_position: int | None,
    ) -> tuple[QuestionnaireItem, int]:
        """Resolve the reference question and its position."""
        question_map = {question.item_id: question for question in self._questions}
        if item_id is not None:
            if item_id not in question_map:
                raise ValueError("item_id does not exist.")
            question_item = question_map[item_id]
            position = next(
                index
                for index, question in enumerate(self._questions)
                if question.item_id == item_id
            )
            return question_item, position

        if item_position is None or item_position >= len(self._questions):
            raise ValueError("item_order_id is bigger than the number of questions")
        return self._questions[item_position], item_position

    def _build_item_prompt_format(
        self,
        question_item: QuestionnaireItem,
    ) -> dict[str, str]:
        """Build placeholder values for single-item and sequential prompts."""
        question = self.generate_question_prompt(question_item)
        if question_item.answer_options is None:
            options = ""
            automatic_output_instructions = ""
        else:
            options = question_item.answer_options.create_options_str()
            response_method = question_item.answer_options.response_generation_method
            automatic_output_instructions = (
                response_method.get_automatic_prompt() if response_method is not None else ""
            )

        return {
            placeholder.PROMPT_QUESTIONS: question,
            placeholder.PROMPT_OPTIONS: options,
            placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS: automatic_output_instructions,
        }

    def _build_battery_prompt_format(
        self,
        reference_item_position: int,
        item_separator: str,
    ) -> tuple[dict[str, str], list[str]]:
        """Build placeholder values and rendered questions for battery prompts."""
        rendered_questions: list[str] = []
        for question in self._questions:
            question_prompt = self.generate_question_prompt(question)
            options = (
                question.answer_options.create_options_str()
                if question.answer_options is not None
                else ""
            )
            rendered_questions.append(
                safe_format_with_regex(
                    question_prompt,
                    {placeholder.PROMPT_OPTIONS: options},
                )
            )

        reference_question = self._questions[reference_item_position]
        options = (
            reference_question.answer_options.create_options_str()
            if reference_question.answer_options is not None
            else ""
        )
        response_method = resolve_battery_response_generation_method(
            questions=list(self._questions),
            item_position=reference_item_position,
        )
        automatic_output_instructions = (
            response_method.get_automatic_prompt() if response_method is not None else ""
        )

        return (
            {
                placeholder.PROMPT_QUESTIONS: item_separator.join(rendered_questions),
                placeholder.PROMPT_OPTIONS: options,
                placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS: automatic_output_instructions,
            },
            rendered_questions,
        )

    def _render_prompt_templates(
        self,
        format_dict: dict[str, str],
    ) -> tuple[str | None, str]:
        """Render the configured system and user prompt templates."""
        system_prompt = (
            None
            if self.system_prompt is None
            else safe_format_with_regex(self.system_prompt, format_dict)
        )
        return system_prompt, safe_format_with_regex(self.prompt, format_dict)

    def _build_battery_prompt_content(
        self,
        prompt: str,
        rendered_questions: list[str],
        item_separator: str,
    ) -> PromptContent:
        """Interleave the exact rendered questions with their assigned images."""
        global_images = self.get_images()
        item_images = [
            self.get_images(item_id=question.item_id, include_global=False)
            for question in self._questions
        ]
        if not global_images and not any(item_images):
            return prompt

        rendered_questionnaire = item_separator.join(rendered_questions)
        if prompt.count(rendered_questionnaire) != 1:
            raise ValueError(
                "Image-bearing battery prompts must contain the rendered questionnaire "
                "items exactly once via the question placeholder."
            )
        prefix, suffix = prompt.split(rendered_questionnaire, maxsplit=1)

        blocks: list[PromptContentBlock] = []
        if prefix:
            blocks.append(prefix)
        blocks.extend(global_images)
        for index, (question_text, images) in enumerate(zip(rendered_questions, item_images)):
            separator = item_separator if index else ""
            blocks.append(f"{separator}{question_text}")
            blocks.extend(images)
        if suffix:
            blocks.append(suffix)
        return tuple(blocks)

    def _finalize_rendered_prompt(
        self,
        system_prompt: str | None,
        prompt_content: PromptContent,
        inference_mode: InferenceMode,
    ) -> tuple[str | None, PromptContent]:
        """Finalize chat or completion output after prompt construction."""
        if inference_mode == "completion":
            if not isinstance(prompt_content, str):
                raise ValueError("Image-bearing prompts are supported only in chat mode.")
            return None, self.render_base_model_prompt(system_prompt, [prompt_content])
        if inference_mode != "chat":
            raise ValueError("`inference_mode` must be either 'chat' or 'completion'.")
        return system_prompt, prompt_content

    def get_prompt_for_questionnaire_type(
        self,
        questionnaire_type: QuestionnairePresentation = QuestionnairePresentation.SINGLE_ITEM,
        item_id: str | int | None = None,
        item_position: int | None = 0,
        item_separator: str = "\n",
        inference_mode: InferenceMode = "chat",
    ) -> tuple[str | None, PromptContent]:
        """Generate the full prompt for a questionnaire presentation.

        Args:
            questionnaire_type: Presentation mode used to render the questionnaire.
            item_id: Questionnaire item ID used for item-specific presentations.
                If supplied, it takes precedence over item_position.
            item_position: Questionnaire position used when item_id is omitted.
            item_separator: Text separating rendered questions in battery mode.
            inference_mode: Return chat content or a rendered completion prompt.

        Returns:
            The system prompt and user content. Image-free user content remains a
            string; image-bearing chat content is returned as ordered text and
            ImageInput blocks.

        Raises:
            ValueError: If the requested item, presentation, or inference mode is
                invalid, or images are used with completion mode.
        """
        question_item, reference_item_position = self._resolve_prompt_question(
            item_id,
            item_position,
        )

        rendered_questions: list[str] = []
        if questionnaire_type in {
            QuestionnairePresentation.SINGLE_ITEM,
            QuestionnairePresentation.SEQUENTIAL,
        }:
            format_dict = self._build_item_prompt_format(question_item)
        elif questionnaire_type == QuestionnairePresentation.BATTERY:
            format_dict, rendered_questions = self._build_battery_prompt_format(
                reference_item_position,
                item_separator,
            )
        else:
            raise ValueError(f"Unsupported questionnaire_type: {questionnaire_type}.")

        system_prompt, prompt = self._render_prompt_templates(format_dict)
        if questionnaire_type == QuestionnairePresentation.BATTERY:
            prompt_content = self._build_battery_prompt_content(
                prompt,
                rendered_questions,
                item_separator,
            )
        else:
            prompt_content = combine_prompt_content(
                prompt,
                self.get_images(item_id=question_item.item_id),
            )

        return self._finalize_rendered_prompt(
            system_prompt,
            prompt_content,
            inference_mode,
        )

    def _get_token_counter(
        self,
        model_id: str,
        tokenizer_backend: Literal["tiktoken", "transformers"],
    ):
        if tokenizer_backend == "tiktoken":
            import tiktoken

            encoding = tiktoken.encoding_for_model(model_id)

            def count_tokens(text: str | None) -> int:
                if text is None:
                    return 0
                return len(encoding.encode(text, disallowed_special=()))

            return count_tokens

        if tokenizer_backend == "transformers":
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Token estimation with tokenizer_backend='transformers' requires "
                    "the optional 'transformers' package."
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            def count_tokens(text: str | None) -> int:
                if text is None:
                    return 0
                return len(tokenizer.encode(text, add_special_tokens=False))

            return count_tokens

        raise ValueError("`tokenizer_backend` must be either 'tiktoken' or 'transformers'.")

    @staticmethod
    def _count_chat_input_tokens(
        system_prompt: str | None,
        prompt: PromptContent,
        count_tokens,
        tokenizer_backend: Literal["tiktoken", "transformers"],
    ) -> int:
        """Count chat message content with a small OpenAI chat wrapper estimate."""
        message_count = 1 + (1 if system_prompt is not None else 0)
        content_tokens = count_tokens(system_prompt) + count_tokens(prompt_content_text(prompt))
        if tokenizer_backend == "tiktoken":
            # OpenAI chat APIs add structural tokens around each message plus a reply cue.
            return content_tokens + message_count * 3 + 3
        return content_tokens

    def calculate_input_token_estimate(
        self,
        model_id: str,
        tokenizer_backend: Literal["tiktoken", "transformers"],
        questionnaire_type: QuestionnairePresentation = QuestionnairePresentation.SINGLE_ITEM,
        inference_mode: InferenceMode = "chat",
        item_separator: str = "\n",
        previous_response_token_estimate: int = 100,
    ) -> int:
        """Estimate the largest input-token context for a questionnaire prompt.

        Args:
            model_id (str): Model identifier for the selected tokenizer backend.
            tokenizer_backend (str): Tokenizer backend, either "tiktoken" or "transformers".
            questionnaire_type (QuestionnairePresentation): Type of questionnaire prompt.
            inference_mode (str): If "chat", count chat message inputs. If "generation",
                count the rendered base-model prompt.
            item_separator (str): Separator used between items for battery prompts.
            previous_response_token_estimate (int): Estimated tokens per previous assistant
                answer in sequential presentation. Image tokens are not included in the estimate.

        Returns:
            int: Estimated largest input-token context for a single model request.
        """
        if inference_mode not in {"chat", "completion"}:
            raise ValueError("`inference_mode` must be either 'chat' or 'completion'.")
        if previous_response_token_estimate < 0:
            raise ValueError("`previous_response_token_estimate` must be non-negative.")

        count_tokens = self._get_token_counter(model_id, tokenizer_backend)

        def count_prompt(system_prompt: str | None, prompt: PromptContent) -> int:
            prompt_text = prompt_content_text(prompt)
            if inference_mode == "completion":
                return count_tokens(prompt_text)
            return self._count_chat_input_tokens(
                system_prompt,
                prompt,
                count_tokens,
                tokenizer_backend,
            )

        if questionnaire_type == QuestionnairePresentation.SINGLE_ITEM:
            return max(
                count_prompt(
                    *self.get_prompt_for_questionnaire_type(
                        questionnaire_type=QuestionnairePresentation.SINGLE_ITEM,
                        item_position=item_position,
                        inference_mode=inference_mode,
                    )
                )
                for item_position in range(len(self._questions))
            )

        if questionnaire_type == QuestionnairePresentation.BATTERY:
            return count_prompt(
                *self.get_prompt_for_questionnaire_type(
                    questionnaire_type=QuestionnairePresentation.BATTERY,
                    item_position=0,
                    item_separator=item_separator,
                    inference_mode=inference_mode,
                )
            )

        if questionnaire_type == QuestionnairePresentation.SEQUENTIAL:
            prompts: list[PromptContent] = []
            answer_count = max(len(self._questions) - 1, 0)
            system_prompt: str | None = None
            for item_position in range(len(self._questions)):
                current_system_prompt, current_prompt = self.get_prompt_for_questionnaire_type(
                    questionnaire_type=QuestionnairePresentation.SEQUENTIAL,
                    item_position=item_position,
                )
                system_prompt = current_system_prompt
                prompts.append(current_prompt)

            if inference_mode == "completion":
                rendered_prompt = self.render_base_model_prompt(
                    system_prompt,
                    [prompt_content_text(prompt) for prompt in prompts],
                )
                return count_tokens(rendered_prompt) + (
                    answer_count * previous_response_token_estimate
                )

            content_tokens = count_tokens(system_prompt) + sum(
                count_tokens(prompt_content_text(prompt)) for prompt in prompts
            )
            previous_answer_tokens = answer_count * previous_response_token_estimate
            if tokenizer_backend == "tiktoken":
                message_count = (
                    len(prompts) + answer_count + (1 if system_prompt is not None else 0)
                )
                return content_tokens + previous_answer_tokens + message_count * 3 + 3
            return content_tokens + previous_answer_tokens

        raise ValueError(f"Unsupported questionnaire_type: {questionnaire_type}.")

    def get_questions(self) -> tuple[QuestionnaireItem, ...]:
        """
        Get an immutable snapshot of loaded interview questions.

        Returns:
            Tuple[QuestionnaireItem, ...]: Loaded questions.
        """
        return tuple(self._questions)

    @property
    def questions(self) -> tuple[QuestionnaireItem, ...]:
        """Read-only view of questionnaire items."""
        return tuple(self._questions)

    def get_question(self, position: int) -> QuestionnaireItem:
        """Return a question by positional index."""
        return self._questions[position]

    def replace_question(self, position: int, questionnaire_item: QuestionnaireItem) -> None:
        """Replace the question at a given index."""
        self._questions[position] = questionnaire_item
        self._drop_stale_item_images()

    def remove_question(self, position: int) -> None:
        """Remove the question at a given index."""
        del self._questions[position]
        self._drop_stale_item_images()

    def get_question_item_id(self, position: int) -> Any:
        """Return the questionnaire item id at a given index."""
        return self._questions[position].item_id

    def load_questionnaire_format(self, questionnaire_source: str | pd.DataFrame) -> Self:
        """Load questionnaire items from a CSV file or pandas DataFrame.

        The source must include `questionnaire_item_id`. It may also include question text,
        stems, prefilled responses, answer option columns, Likert generation columns, and
        simple response-generation presets. List-like columns must contain Python lists or
        Python-list strings, for example `["No", "Yes"]`.

        Args:
            questionnaire_source (str or pd.Dataframe): Path to a CSV file or a DataFrame.

        Returns:
            Self: The updated instance with loaded questions.
        """
        questionnaire_questions: list[QuestionnaireItem] = []

        # This is a duplicate check with actual Error here,
        # because if the method is called on its own it should not run the remaining code
        if not self._check_valid_questionnaire(questionnaire_source=questionnaire_source):
            raise ValueError("Please provide a non empty DataFrame or a valid String.")

        if isinstance(questionnaire_source, pd.DataFrame):
            df = questionnaire_source
        else:
            df = pd.read_csv(questionnaire_source)

        for _, row in df.iterrows():
            questionnaire_item_id = row[constants.QUESTIONNAIRE_ITEM_ID]
            questionnaire_question_content = optional_row_value(
                row,
                constants.QUESTION_CONTENT,
            )
            question_stem = optional_row_value(row, constants.QUESTION_STEM)
            prefilled_response = optional_row_value(
                row,
                QuestionnaireLoaderColumn.PREFILLED_RESPONSE,
            )
            answer_options = _build_answer_options_from_row(row, questionnaire_item_id)

            generated_questionnaire_question = QuestionnaireItem(
                item_id=questionnaire_item_id,
                question_content=questionnaire_question_content,
                question_stem=question_stem,
                answer_options=answer_options,
                prefilled_response=prefilled_response,
            )
            questionnaire_questions.append(generated_questionnaire_question)

        self._questions = questionnaire_questions
        self._drop_stale_item_images()
        return self

    # TODO Item order could be given by ids
    @overload
    def prepare_prompt(
        self,
        question_stem: str | None = None,
        answer_options: AnswerOptions | None = None,
        prefilled_responses: dict[int, str] | None = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    @overload
    def prepare_prompt(
        self,
        question_stem: list[str] | None = None,
        answer_options: dict[str, AnswerOptions] | None = None,
        prefilled_responses: dict[int, str] | None = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    def prepare_prompt(
        self,
        question_stem: str | list[str] | None = None,
        answer_options: AnswerOptions | dict[str, AnswerOptions] | None = None,
        prefilled_responses: dict[int, str] | None = None,
        randomized_item_order: bool = False,
    ) -> Self:
        """
        Prepare the interview by assigning question stems, answer options, and prefilled responses.

        Args:
            question_stem (str or List[str], optional): Single or list of question stems.
            answer_options (AnswerOptions or Dict[int, AnswerOptions], optional):
                Answer options for all or per question.
            prefilled_responses (Dict[int, str], optional):
                If you provide prefilled responses, they will be used
            to fill the answers instead of prompting the LLM for that question.
            randomized_item_order (bool): If True, randomize the order of questions.
        Returns:
            Self: The updated instance with prepared questions.
        """
        questionnaire_questions: list[QuestionnaireItem] = self._questions

        prompt_list = isinstance(question_stem, list)
        if prompt_list:
            assert len(question_stem) == len(
                questionnaire_questions
            ), "If a list of question stems is given, length of prompt "
            "       and survey questions have to be the same"

        options_dict = False

        if isinstance(answer_options, AnswerOptions):
            # self._same_options = True # unnecessary
            options_dict = False
        elif isinstance(answer_options, dict):
            # self._same_options = False # unnecessary
            options_dict = True

        updated_questions: list[QuestionnaireItem] = []

        if not prefilled_responses:
            prefilled_responses = {}
            # for survey_question in survey_questions:
            # prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_dict:
            updated_questions = []
            for question in questionnaire_questions:
                new_questionnaire_question = replace(
                    question,
                    question_stem=(question_stem if question_stem else question.question_stem),
                    answer_options=answer_options,
                    prefilled_response=prefilled_responses.get(question.item_id),
                )
                updated_questions.append(new_questionnaire_question)

        elif not prompt_list and options_dict:
            for question in questionnaire_questions:
                new_questionnaire_question = replace(
                    question,
                    question_stem=(question_stem if question_stem else question.question_stem),
                    answer_options=answer_options.get(question.item_id),
                    prefilled_response=prefilled_responses.get(question.item_id),
                )
                updated_questions.append(new_questionnaire_question)

        elif prompt_list and not options_dict:
            for i, question in enumerate(questionnaire_questions):
                new_questionnaire_question = replace(
                    question,
                    question_stem=(question_stem[i] if question_stem else question.question_stem),
                    answer_options=answer_options,
                    prefilled_response=prefilled_responses.get(question.item_id),
                )
                updated_questions.append(new_questionnaire_question)
        elif prompt_list and options_dict:
            for i, question in enumerate(questionnaire_questions):
                new_questionnaire_question = replace(
                    question,
                    question_stem=(question_stem[i] if question_stem else question.question_stem),
                    answer_options=answer_options.get(question.item_id),
                    prefilled_response=prefilled_responses.get(question.item_id),
                )
                updated_questions.append(new_questionnaire_question)

        if randomized_item_order:
            random.shuffle(updated_questions)

        self._questions = updated_questions
        return self

    def generate_question_prompt(self, questionnaire_items: QuestionnaireItem) -> str:
        """
        Generate the prompt string for a single interview question.

        Args:
            questionnaire_items (InterviewItem): The question to prompt.

        Returns:
            str: The formatted prompt for the question.
        """

        if questionnaire_items.question_stem:
            if placeholder.QUESTION_CONTENT in questionnaire_items.question_stem:
                format_dict = {placeholder.QUESTION_CONTENT: questionnaire_items.question_content}
                question_prompt = safe_format_with_regex(
                    questionnaire_items.question_stem, format_dict
                )
            else:
                question_prompt = f"""{questionnaire_items.question_stem} {questionnaire_items.question_content}"""  # noqa: E501
        else:
            question_prompt = f"""{questionnaire_items.question_content}"""

        if questionnaire_items.answer_options:
            _options_str = questionnaire_items.answer_options.create_options_str()
            if _options_str is not None:
                safe_formatter = {placeholder.PROMPT_OPTIONS: _options_str}
                question_prompt = safe_format_with_regex(question_prompt, safe_formatter)

        return question_prompt

    def __len__(self) -> int:
        """
        Returns the number of questions in our LLMPrompt.

        Returns:
            int: The number of questions.
        """
        return len(self._questions)

    def __str__(self) -> str:
        """
        Creates a human readable display of the system prompt and prompt in default Battery format.
        """
        name_str: str = f"=== {self.questionnaire_name} ==="
        sys_prompt, prompt = self.get_prompt_for_questionnaire_type(
            questionnaire_type=QuestionnairePresentation.BATTERY
        )
        sys_str: str = f"=== SYSTEM_PROMPT ===\n{sys_prompt}"
        prompt_str: str = (
            "=== USER_PROMPT_WITH_ALL_QUESTIONS ===\n" f"{format_prompt_content(prompt)}"
        )

        full_str: str = f"{name_str}\n{sys_str}\n{prompt_str}"
        return full_str

    def insert_questions(
        self,
        items: QuestionnaireItem | list[QuestionnaireItem],
        position: int = None,
    ) -> None:
        """Inserts one or more questions into the questionnaire.

        Args:
            items (Union[QuestionnaireItem, List[QuestionnaireItem]]): A single
                QuestionnaireItem or a list of items to insert.
            position (int): The index where the questions should be inserted.
                Default [None] adds them at the end.
        """
        if position is None:
            position = len(self._questions)

        if not isinstance(items, (list, tuple)):
            items = [items]

        self._questions[position:position] = items


_IDX_TYPES = Literal["char_lower", "char_upper", "integer", "no_index"]


def generate_likert_options(
    n: int,
    answer_texts: list[str] | None,
    only_from_to_scale: bool = False,
    random_order: bool = False,
    reversed_order: bool = False,
    even_order: bool = False,
    add_middle_category: bool = False,
    str_middle_cat: str = "Neutral",
    add_refusal: bool = False,
    refusal_code: str = "-99",
    start_idx: int = 1,
    list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
    scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
    index_answer_separator: str = ": ",
    options_separator: str = ", ",
    idx_type: _IDX_TYPES = "integer",
    response_generation_method: ResponseGenerationMethod | None = None,
) -> AnswerOptions:
    """Generates a set of options and a prompt for a Likert-style scale.

    This function creates a numeric or alphabetic scale of a specified size (n),
    optionally attaching textual labels to the scale. It provides
    extensive control over ordering, formatting, and the final prompt string.

    Args:
        n (int): The number of options to generate (e.g., 5 for a 5-point scale).
        answer_texts (Optional[List[str]]): A list of text labels for each option.
            Its length must equal `n` if provided.
        only_from_to_scale (bool, optional): If True, the prompt will only show the
            min and max of the scale (e.g., "1 to 5"). Defaults to False.
        random_order (bool, optional): If True, the options are randomized. Defaults to False.
        reversed_order (bool, optional): If True, the options are in reversed input order.
            Defaults to False.
        even_order (bool, optional): If True, options the center option will be removed.
            E.g., for n=5: 1, 2, 4, 5
        add_middle_category (bool, optional): If True, a middle category will be added.
            The name can be specified,
            by default it is "Neutral". E.g., for n=4: 1, 2, 3: Neutral, 4, 5
        str_middle_cat (str, optional): The label for the middle category
            if `add_middle_category` is True.
            Defaults to "Neutral".
        add_refusal (bool, optional): If True, an additional option for
            "Don't know / Refuse to answer" will be added.
            Defaults to False.
        refusal_code (str, optional): The code assigned to the refusal option
            if `add_refusal` is True.
            Defaults to "-99".
        start_idx (int, optional): The starting index for the scale (usually 0 or 1).
            Defaults to 1.
        list_prompt_template (str, optional): The template for prompts that list all options.
        scale_prompt_template (str, optional): The template for prompts that only show the range.
        index_answer_separator (str, optional): The string used to separate an index from its
            text label (e.g., "1: Strongly Agree"). Defaults to ": ".
        options_separator (str, optional): The string used to separate options when listed
            in the prompt. Defaults to ", ".
        idx_type (_IDX_TYPES, optional): The type of index to use: "integer", "upper" (A, B, C),
            or "lower" (a, b, c). Defaults to "integer".
        response_generation_method (Optional[ResponseGenerationMethod], optional): An object
            controlling how the final response object is generated. Defaults to None.

    Raises:
        ValueError: If `answer_texts` is provided and its length does not match `n`.

    Returns:
        AnswerOptions: An object containing the generated list of option strings and the
        final formatted prompt ready for display.

    Example:
        .. code-block:: python

            # Generate a classic 5-point "Strongly Disagree" to "Strongly Agree" scale
            labels = [
                "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"
            ]
            options = SurveyOptionGenerator.generate_likert_options(n=5, answer_texts=labels)
    """

    if only_from_to_scale:
        # if len(answer_texts) != 2:
        #     raise ValueError(
        #         "From-To scales require exactly 2 descriptions, but "
        #         f"answer_texts was set to '{answer_texts}'."
        #     )
        if idx_type != "integer":
            raise ValueError(
                "From-To scales require an integer scale index, but "
                f"idx_type was set to '{idx_type}'."
            )
    else:
        if answer_texts:
            if len(answer_texts) != n:
                raise ValueError(
                    "answer_texts and n need to be the same length, but "
                    f"answer_texts has length {len(answer_texts)} "
                    f"and n was given as {n}."
                )
    if even_order:
        if n % 2 == 0:
            raise ValueError("If you want to turn a scale even, it should be odd before.")
        middle_index = n // 2
        answer_texts = answer_texts[:middle_index] + answer_texts[middle_index + 1 :]
        n = n - 1
    if add_middle_category:
        if n % 2 != 0:
            raise ValueError("If you want to add a middle category, it should be even before.")
        middle_index = n // 2
        answer_texts = answer_texts[:middle_index] + [str_middle_cat] + answer_texts[middle_index:]
        n = n + 1

    if random_order:
        if len(answer_texts) < 2:
            raise ValueError("There must be at least two answer options to reorder randomly.")
        random.shuffle(answer_texts)  # no assignment needed because shuffles already inplace
    if reversed_order:
        if len(answer_texts) < 2:
            raise ValueError("There must be at least two answer options to reorder in reverse.")
        answer_texts = answer_texts[::-1]

    if add_refusal:
        answer_texts.append("Don't know / Refuse to answer")
        n += 1

    answer_option_indices = []
    if idx_type == "no_index":
        # no index, just the answer options directly
        answer_option_indices = None
    elif idx_type == "integer":
        if add_refusal:  # if refusal is added, assign it a common code -99
            for i in range(n - 1):
                answer_code = i + start_idx
                answer_option_indices.append(str(answer_code))
            answer_option_indices.append(refusal_code)  # common code for refusal
        else:
            for i in range(n):
                answer_code = i + start_idx
                answer_option_indices.append(str(answer_code))
    else:
        # TODO @Jens add these to constants.py
        if idx_type == "char_lower":
            for i in range(n):
                answer_option_indices.append(ascii_lowercase[(i + start_idx) % 26])
        elif idx_type == "char_upper":
            for i in range(n):
                answer_option_indices.append(ascii_uppercase[(i + start_idx) % 26])

    answer_texts_object = AnswerTexts(
        answer_texts=answer_texts,
        indices=answer_option_indices,
        index_answer_seperator=index_answer_separator,
        option_seperators=options_separator,
        only_scale=only_from_to_scale,
    )

    questionnaire_options = AnswerOptions(
        answer_texts=answer_texts_object,
        from_to_scale=only_from_to_scale,
        list_prompt_template=list_prompt_template,
        scale_prompt_template=scale_prompt_template,
        response_generation_method=response_generation_method,
    )

    return questionnaire_options
