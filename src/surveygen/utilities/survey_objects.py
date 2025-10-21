from typing import List, Optional, NamedTuple, Dict, Final, TYPE_CHECKING
from ..utilities import constants, prompt_templates
from ..utilities.prompt_creation import PromptCreation

import pandas as pd

from dataclasses import dataclass

if TYPE_CHECKING:
    from ..llm_interview import LLMInterview

@dataclass
class AnswerOptions:
    """
    Stores answer options for a single question or a full questionnaire.

    Args:
        answer_text (list): A list of possible answer strings.
        index (list | None): Optionally, store answer option index separately, e.g., for structured outputs.
        from_to_scale (bool): If True, treat answer_text as a scale [start, ..., end].
        list_prompt_template (str): A format string for list-based options.
                                    Must contain an '{options}' placeholder.
        scale_prompt_template (str): A format string for scale-based options.
                                        Must contain '{start}' and '{end}' placeholders.
        options_seperator (str): The seperator string used between options.
    """
    answer_text: List[str]
    index: Optional[List[str]] = None
    from_to_scale: bool = False
    list_prompt_template: Optional[str] = prompt_templates.LIST_OPTIONS_DEFAULT
    scale_prompt_template: Optional[str] = prompt_templates.SCALE_OPTIONS_DEFAULT
    options_seperator: str = ", "

    def create_options_str(self) -> str:
        if self.from_to_scale:
            if self.scale_prompt_template is None: return None
            if len(self.answer_text) < 2:
                raise ValueError(f"From-To scale requires at least a start and end value, but answer_text was set to {self.answer_text}.")
            start_option = self.answer_text[0]
            end_option = self.answer_text[-1]
            return self.scale_prompt_template.format(start=start_option, end=end_option)        
        else:
            if self.list_prompt_template is None: return None
            joined_options = self.options_seperator.join(self.answer_text)
            return self.list_prompt_template.format(options=joined_options)


class QuestionLLMResponseTuple(NamedTuple):
    question: str
    llm_response: str
    logprobs: Optional[Dict[str, float]]
    reasoning: Optional[str]


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

    def get_transcript(self) -> str:
        parts = [self.interview.get_prompt_structure()]

        # Use enumerate to get the index without a manual counter
        for i, (question, llm_response) in enumerate(self.results.values()):
            if i > 0:
                parts.append(f"\nQ: {question}")
            parts.append(f"\nA: {llm_response}")

        return "".join(parts)


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

    def json_system_prompt(
            self,
            json_fields: List[str] | Dict[str, str],
            json_instructions: str = prompt_templates.SYSTEM_JSON_DEFAULT
        ) -> str:
        """`json_fields` can have optional explanations in the form `{'attribute': 'explanation',...}`"""
        creator = PromptCreation()
        if isinstance(json_fields, dict):
            json_attributes = list(json_fields.keys())
            json_explanation = list(json_fields.values())
        else:
            json_attributes = json_fields
            json_explanation = None
        creator.set_output_format_json(
            json_attributes = json_attributes,
            json_explanation = json_explanation,
            json_instructions = json_instructions
        )
        json_appendix = creator.get_output_prompt()

        system_prompt = f"""{self.system_prompt}
{json_appendix}"""
        return system_prompt
