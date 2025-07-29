from typing import List, Optional, NamedTuple, Dict, Final, TYPE_CHECKING
from ..utilities import constants, prompt_templates
from ..utilities.prompt_creation import PromptCreation

import pandas as pd

from dataclasses import dataclass

if TYPE_CHECKING:
    from ..llm_interview import LLMInterview

class AnswerOptions:

    def __init__(
        self,
        answer_text: List[str],
        from_to_scale: bool,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        options_seperator: str = ", "
    ):
        """
        Initializes the AnswerOptions object.

        Args:
            answer_text (list): A list of possible answer strings.
            from_to_scale (bool): If True, treat answer_text as a scale [start, ..., end].
            list_prompt_template (str): A format string for list-based options.
                                        Must contain an '{options}' placeholder.
            scale_prompt_template (str): A format string for scale-based options.
                                         Must contain '{start}' and '{end}' placeholders.
            options_seperator (str): The seperator string used between options.
        """
        self.answer_text: List[str] = answer_text
        self.from_to_scale: bool = from_to_scale
        self.list_prompt_template: str = list_prompt_template
        self.scale_prompt_template: str = scale_prompt_template
        self.options_seperator: str = options_seperator

    def create_options_str(self) -> str:
        if not self.from_to_scale:
            joined_options = self.options_seperator.join(self.answer_text)
            return self.list_prompt_template.format(options=joined_options)
        else:
            if len(self.answer_text) < 2:
                return "Scale requires at least a start and end value."
            start_option = self.answer_text[0]
            end_option = self.answer_text[-1]
            return self.scale_prompt_template.format(start=start_option, end=end_option)


class QuestionLLMResponseTuple(NamedTuple):
    question: str
    llm_response: str


@dataclass
class InterviewResult:
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


@dataclass
class InterviewItem:
    """Represents a single survey question."""

    item_id: int
    question_content: str
    question_stem: Optional[str] = None
    answer_options: Optional[AnswerOptions] = None
    prefilled_response: Optional[str] = None

@dataclass
class InferenceOptions:
    system_prompt: str
    task_instruction: str
    question_prompts: Dict[int, str]
    answer_options: List[AnswerOptions]
    # guided_decodings: Optional[Dict[int, GuidedDecodingParams]]
    # full_guided_decoding: Optional[GuidedDecodingParams]
    # json_structure: Optional[List[str]]
    # full_json_structure: Optional[List[str]]
    order: List[int]

    def create_single_question(
        self, question_id: int, task_instruction: bool = False
    ) -> str:
        if task_instruction:
            return f"""{self.task_instruction} 
{self.question_prompts[question_id]}""".strip()
        else:
            return f"""{self.question_prompts[question_id]}"""

    def create_all_questions(self) -> str:
        default_prompt = f"{self.task_instruction}"
        all_questions_prompt = ""
        for question_prompt in self.question_prompts.values():
            all_questions_prompt = f"{all_questions_prompt}\n{question_prompt}"
        if len(default_prompt) > 0:
            all_prompt = f"{default_prompt.strip()}\n{all_questions_prompt.strip()}"
        else:
            all_prompt = all_questions_prompt.strip()
        return all_prompt

    def json_system_prompt(self, json_options: List[str]) -> str:
        creator = PromptCreation()
        creator.set_ouput_format_json(
            json_attributes=json_options, json_explanation=None
        )
        json_appendix = creator.get_output_prompt()

        system_prompt = f"""{self.system_prompt}
{json_appendix}"""
        return system_prompt
