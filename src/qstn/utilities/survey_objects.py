import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from ..inference.response_generation import (
    ResponseGenerationMethod,
)
from ..utilities import placeholder, prompt_templates

if TYPE_CHECKING:
    from ..prompt_builder import LLMPrompt

from ..utilities import constants


@dataclass
class AnswerTexts:
    """Represents the answer choices for a questionnaire item.

    This class manages the different formats of answer texts, including
    lists of options and scales. It can handle answers with or without
    all_answers.

    Attributes:
        full_answers (List[str]): A list of the complete answer strings,
            including indices and separators if provided.
        answer_texts (Optional[List[str]]): The text of the answer options.
        indices (Optional[List[str]]): The indices corresponding to the
            answer options.
        index_answer_seperator (str): The separator between an index and
            its corresponding answer text. Defaults to ": ".
        option_seperators (Tuple[str, ...]): The separators used to join
            multiple answer options into a single string. Defaults to (", ",).
        only_scale (bool): If True, the answers represent a scale, and
            only the first and last answer texts are used to create a
            range of options. Defaults to False.
    """

    full_answers: list[str]
    answer_texts: list[str] | None = None
    indices: list[str] | None = None
    index_answer_seperator: str = ": "
    option_seperators: str = (", ",)
    only_scale: bool = (False,)

    def __init__(
        self,
        answer_texts: list[str],
        indices: list[str] | None = None,
        index_answer_seperator: str = ": ",
        option_seperators: str = ", ",
        only_scale: bool = False,
    ):
        """Initializes the AnswerTexts object.

        Args:
            answer_texts (List[str]): The text of the answer options.
            indices (Optional[List[str]]): The indices corresponding to the
                answer options. Defaults to None.
            index_answer_seperator (str): The separator between an index and
                its corresponding answer text. Defaults to ": ".
            option_seperators (str): The separators used to join
                multiple answer options into a single string. Defaults to ", ".
            only_scale (bool): If True, the answers represent a scale.
                Defaults to False.

        Raises:
            ValueError: If neither answer_texts nor indices are provided.
        """
        self.answer_texts = answer_texts
        self.indices = indices
        self.index_answer_seperator = index_answer_seperator
        self.option_seperators = option_seperators
        self.only_scale = only_scale

        if self.only_scale:
            full_indices = []
            dummy_answer_texts = []
            for index in range(int(self.indices[0]), int(self.indices[-1]) + 1):
                index = str(index)
                if index == self.indices[0]:
                    dummy_answer_texts.append(self.answer_texts[0])
                elif index == self.indices[-1]:
                    dummy_answer_texts.append(self.answer_texts[-1])
                else:
                    dummy_answer_texts.append("")
                full_indices.append(index)
            self.indices = full_indices
            if len(self.answer_texts) == 2:
                self.answer_texts = dummy_answer_texts
        if self.answer_texts and self.indices:
            self.full_answers = []
            for answer_text, index in zip(self.answer_texts, self.indices):
                if answer_text == "":
                    self.full_answers.append(f"{index}")
                else:
                    self.full_answers.append(f"{index}{self.index_answer_seperator}{answer_text}")
        elif self.answer_texts and self.indices is None:
            self.full_answers = [f"{answer_text}" for answer_text in self.answer_texts]
        elif self.answer_texts is None and self.indices:
            self.full_answers = [f"{index}" for index in self.indices]
        else:
            raise ValueError("Invalid Answer Text, because neither text nor indices were given.")

    def get_list_answer_texts(self):
        """Returns the answer texts as a single string, joined by the option separators.

        Returns:
            str: A string representation of the list of answers.
        """
        return self.option_seperators.join(self.full_answers)

    def get_scale_answer_texts(self):
        """Returns the first and last answer texts for a scale.

        Returns:
            Tuple[str, str]: A tuple containing the first and last answer
                texts.
        """
        return self.full_answers[0], self.full_answers[-1]


@dataclass
class AnswerOptions:
    """
    Stores answer options for a single question or a full questionnaire.

    Args:
        answer_texts (list): A list of possible answer strings.
        index (list | None): Optionally store answer option indices separately,
            e.g., for structured outputs.
        from_to_scale (bool): If True, treat answer_text as a scale [start, ..., end].
        list_prompt_template (str): A format string for list-based options.
                                    Must contain an '{options}' placeholder.
        scale_prompt_template (str): A format string for scale-based options.
                                        Must contain '{start}' and '{end}' placeholders.
    """

    answer_texts: AnswerTexts
    from_to_scale: bool = False
    list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT
    scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT
    response_generation_method: ResponseGenerationMethod | None = None
    _response_generation_method: ResponseGenerationMethod | None = None

    def _response_generation_options(
        self,
        response_generation_method: ResponseGenerationMethod | None = None,
    ) -> list[str]:
        if response_generation_method is None:
            response_generation_method = object.__getattribute__(
                self,
                "_response_generation_method",
            )
        if self.answer_texts.indices is not None and response_generation_method is not None:
            if response_generation_method.output_index_only:
                return list(self.answer_texts.indices)
        return list(self.answer_texts.full_answers)

    def _response_generation_options_text(
        self,
        response_generation_method: ResponseGenerationMethod | None = None,
    ) -> str:
        if response_generation_method is None:
            response_generation_method = object.__getattribute__(
                self,
                "_response_generation_method",
            )
        if self.from_to_scale:
            if self.scale_prompt_template is None:
                return ", ".join(self._response_generation_options(response_generation_method))

            if self.answer_texts.indices is not None and response_generation_method is not None:
                if response_generation_method.output_index_only:
                    start_option = self.answer_texts.indices[0]
                    end_option = self.answer_texts.indices[-1]
                    return self.scale_prompt_template.format(start=start_option, end=end_option)

            start_option, end_option = self.answer_texts.get_scale_answer_texts()
            return self.scale_prompt_template.format(start=start_option, end=end_option)

        if self.list_prompt_template is None:
            return ", ".join(self._response_generation_options(response_generation_method))

        return self.list_prompt_template.format(
            options=self.answer_texts.option_seperators.join(
                self._response_generation_options(response_generation_method)
            )
        )

    def _response_generation_scale_range_text(
        self,
        response_generation_method: ResponseGenerationMethod | None = None,
    ) -> str:
        if not self.from_to_scale:
            return ""
        return self._response_generation_options_text(response_generation_method)

    def _response_generation_prompt_formatter(
        self,
        response_generation_method: ResponseGenerationMethod | None = None,
    ) -> dict[str, str]:
        return {
            placeholder.PROMPT_OPTIONS: self._response_generation_options_text(
                response_generation_method
            ),
            placeholder.SCALE_RANGE: self._response_generation_scale_range_text(
                response_generation_method
            ),
        }

    def _prepare_response_generation_method(
        self,
        response_generation_method: ResponseGenerationMethod | None,
    ) -> ResponseGenerationMethod | None:
        prepared_method = copy.deepcopy(response_generation_method)

        if prepared_method is not None:
            prepared_method = prepared_method.prepare_for_answer_options(
                options=self._response_generation_options(prepared_method),
                options_text=self._response_generation_options_text(prepared_method),
                prompt_formatter=self._response_generation_prompt_formatter(prepared_method),
            )

        return prepared_method

    def __setattr__(self, name: str, value: object) -> None:
        if name == "response_generation_method":
            value = self._prepare_response_generation_method(value)
            object.__setattr__(self, "_response_generation_method", value)
            return
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> object:
        if name == "response_generation_method":
            return object.__getattribute__(self, "_response_generation_method")
        return object.__getattribute__(self, name)

    def __init__(
        self,
        answer_texts: AnswerTexts,
        from_to_scale: bool = False,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        response_generation_method: ResponseGenerationMethod | None = None,
    ):
        self.answer_texts = answer_texts
        self.from_to_scale = from_to_scale
        self.list_prompt_template = list_prompt_template
        self.scale_prompt_template = scale_prompt_template
        self.response_generation_method = response_generation_method

    def create_options_str(self) -> str:
        if self.from_to_scale:
            if self.scale_prompt_template is None:
                return None
            if len(self.answer_texts.answer_texts) < 2:
                raise ValueError(
                    "From-To scale requires at least a start and end value, "
                    f"but answer_text was set to {self.answer_texts}."
                )
            start_option, end_option = self.answer_texts.get_scale_answer_texts()
            return self.scale_prompt_template.format(start=start_option, end=end_option)
        else:
            if self.list_prompt_template is None:
                return None
            return self.list_prompt_template.format(
                options=self.answer_texts.get_list_answer_texts()
            )


class QuestionLLMResponseTuple(NamedTuple):
    """Contains the question, llm_response and optionally logprobs and built-in reasoning."""

    question: str
    llm_response: str
    logprobs: dict[str, float] | None
    reasoning: str | None


@dataclass
class InferenceResult:
    """Contains a prompt and the corresponding responses by the LLM.
    Can return results as a dataframe or return the transcript of all questions and answers.
    """

    questionnaire: "LLMPrompt"
    results: dict[int, QuestionLLMResponseTuple]

    def to_dataframe(self) -> pd.DataFrame:
        answers = []
        for item_id, question_llm_response_tuple in self.results.items():
            answers.append((item_id, *question_llm_response_tuple))
        return pd.DataFrame(
            answers,
            columns=[constants.QUESTIONNAIRE_ITEM_ID, *question_llm_response_tuple._fields],
        )

    def get_questions_transcript(self) -> str:
        parts = []

        for i, (_, question_llm_response_tuple) in enumerate(self.results.items()):
            if hasattr(self.questionnaire, "get_question"):
                question_obj = self.questionnaire.get_question(i)
            else:
                question_obj = self.questionnaire._questions[i]
            parts.append(self.questionnaire.generate_question_prompt(question_obj))
            parts.append(question_llm_response_tuple.llm_response)

        return "\n".join(parts)


@dataclass
class QuestionnaireItem:
    """Represents a single questionnaire item."""

    item_id: str
    question_content: str | int
    question_stem: str | None = None
    answer_options: AnswerOptions | None = None
    prefilled_response: str | None = None
