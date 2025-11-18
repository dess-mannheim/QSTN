from typing import List, Optional, Dict, TYPE_CHECKING, NamedTuple
from ..utilities import constants, prompt_templates

from ..inference.response_generation import (
    ResponseGenerationMethod,
    JSONResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
    LogprobResponseGenerationMethod,
    JSONAllOptionsResponseGenerationMethod,
)

import pandas as pd

from dataclasses import dataclass

if TYPE_CHECKING:
    from ..llm_interview import LLMInterview




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

    full_answers: List[str]
    answer_texts: Optional[List[str]] = None
    indices: Optional[List[str]] = None
    index_answer_seperator: str = ": "
    option_seperators: str = (", ",)
    only_scale: bool = (False,)

    def __init__(
        self,
        answer_texts: List[str],
        indices: Optional[List[str]] = None,
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
            self.answer_texts = dummy_answer_texts
        if self.answer_texts and self.indices:
            self.full_answers = [
                f"{index}{self.index_answer_seperator}{answer_text}"
                for answer_text, index in zip(self.answer_texts, self.indices)
            ]
        elif self.answer_texts and self.indices == None:
            self.full_answers = [f"{answer_text}" for answer_text in self.answer_texts]
        elif self.answer_texts == None and self.indices:
            self.full_answers = [f"{index}" for index in self.indices]
        else:
            raise ValueError(
                "Invalid Answer Text, because neither text nor indices were given."
            )

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
        index (list | None): Optionally, store answer option index separately, e.g., for structured outputs.
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
    response_generation_method: Optional[ResponseGenerationMethod] = None

    def __init__(
        self,
        answer_texts: AnswerTexts,
        from_to_scale: bool = False,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        response_generation_method: Optional[ResponseGenerationMethod] = None,
    ):
        self.answer_texts = answer_texts
        self.from_to_scale = from_to_scale
        self.list_prompt_template = list_prompt_template
        self.scale_prompt_template = scale_prompt_template
        self.response_generation_method = response_generation_method

        if self.response_generation_method:
            if isinstance(
                self.response_generation_method, JSONAllOptionsResponseGenerationMethod
            ):
                if self.response_generation_method.output_index_only:
                    self.response_generation_method.json_fields = {
                        _option: "probability" for _option in self.answer_texts.indices
                    }
                    self.response_generation_method.constraints = {
                        _option: "float" for _option in self.answer_texts.indices
                    }
                else:
                    self.response_generation_method.json_fields = {
                        _option: "probability"
                        for _option in self.answer_texts.full_answers
                    }
                    self.response_generation_method.constraints = {
                        _option: "float" for _option in self.answer_texts.full_answers
                    }

            elif isinstance(
                self.response_generation_method, JSONResponseGenerationMethod
            ):
                fields = self.response_generation_method.json_fields
                if isinstance(fields, dict):
                    for key in fields:
                        if fields[key] == constants.OPTIONS_ADJUST:
                            if self.response_generation_method.output_index_only:
                                fields[key] = ", ".join(answer_texts.indices)
                            else:
                                fields[key] = ", ".join(answer_texts.full_answers)

                constraints = self.response_generation_method.constraints
                for key in constraints:
                    if constraints[key] == constants.OPTIONS_ADJUST:
                        if self.response_generation_method.output_index_only:
                            numbers = []
                            for index in answer_texts.indices:
                                try:
                                    number = int(index)
                                except:
                                    number = index
                                numbers.append(number)
                            constraints[key] = numbers
                        else:
                            constraints[key] = answer_texts.full_answers

            elif isinstance(
                self.response_generation_method, ChoiceResponseGenerationMethod
            ) or isinstance(
                self.response_generation_method, LogprobResponseGenerationMethod
            ):
                if (
                    self.response_generation_method.allowed_choices
                    == constants.OPTIONS_ADJUST
                ):
                    if self.response_generation_method.output_index_only:
                        self.response_generation_method.allowed_choices = (
                            answer_texts.indices
                        )
                    else:
                        self.response_generation_method.allowed_choices = (
                            answer_texts.full_answers
                        )

    def create_options_str(self) -> str:
        if self.from_to_scale:
            if self.scale_prompt_template is None:
                return None
            if len(self.answer_texts.answer_texts) < 2:
                raise ValueError(
                    f"From-To scale requires at least a start and end value, but answer_text was set to {self.answer_texts}."
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
    logprobs: Optional[Dict[str, float]]
    reasoning: Optional[str]


@dataclass
class InterviewResult:
    """Contains an interview and the corresponding responses by the LLM.
    Can return results as a dataframe or return the transcript of all questions and answers.
    """

    interview: "LLMInterview"
    results: Dict[int, QuestionLLMResponseTuple]

    def to_dataframe(self) -> pd.DataFrame:
        answers = []
        for item_id, question_llm_response_tuple in self.results.items():
            answers.append((item_id, *question_llm_response_tuple))
        return pd.DataFrame(
            answers,
            columns=[constants.INTERVIEW_ITEM_ID, *question_llm_response_tuple._fields],
        )

    def get_questions_transcript(self) -> str:
        parts = []

        for i, (_, question_llm_response_tuple) in enumerate(self.results.items()):
            parts.append(
                self.interview.generate_question_prompt(self.interview._questions[i])
            )
            parts.append(question_llm_response_tuple.llm_response)

        return "\n".join(parts)


@dataclass
class InterviewItem:
    """Represents a single survey question."""

    item_id: int
    question_content: str
    question_stem: Optional[str] = None
    answer_options: Optional[AnswerOptions] = None
    prefilled_response: Optional[str] = None
