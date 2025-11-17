from typing import List, Optional, Dict, Final, TYPE_CHECKING, NamedTuple
from ..utilities import constants, prompt_templates
from ..utilities.prompt_creation import PromptCreation

from ..inference.response_generation import ResponseGenerationMethod, JSONResponseGenerationMethod, ChoiceResponseGenerationMethod, LogprobResponseGenerationMethod, JSONAllOptionsResponseGenerationMethod

import pandas as pd

from dataclasses import dataclass

if TYPE_CHECKING:
    from ..llm_interview import LLMInterview


import copy

@dataclass
class AnswerTexts:
    full_answers: List[str]
    answer_texts: Optional[List[str]] = None
    indices: Optional[List[str]] = None
    index_answer_seperator: str = ": "
    option_seperators: str = ", ",
    only_scale: bool = False,

    def __init__(
        self,
        answer_texts: List[str],
        indices: Optional[List[str]] = None,
        index_answer_seperator: str = ": ",
        option_seperators: str = ", ",
        only_scale: bool = False
    ):
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
        return self.option_seperators.join(self.full_answers)

    def get_scale_answer_texts(self):
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
            if isinstance(self.response_generation_method, JSONAllOptionsResponseGenerationMethod):
                if self.response_generation_method.output_index_only:
                    self.response_generation_method.json_fields={_option: "probability" for _option in self.answer_texts.indices}
                    self.response_generation_method.constraints={_option: "float" for _option in self.answer_texts.indices}
                else:
                    self.response_generation_method.json_fields={_option: "probability" for _option in self.answer_texts.full_answers}
                    self.response_generation_method.constraints={_option: "float" for _option in self.answer_texts.full_answers}

            elif isinstance(self.response_generation_method, JSONResponseGenerationMethod):
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
                
            elif isinstance(self.response_generation_method, ChoiceResponseGenerationMethod) or isinstance(self.response_generation_method, LogprobResponseGenerationMethod):
                if self.response_generation_method.allowed_choices == constants.OPTIONS_ADJUST:
                    if self.response_generation_method.output_index_only:
                        self.response_generation_method.allowed_choices = answer_texts.indices
                    else:
                        self.response_generation_method.allowed_choices = answer_texts.full_answers

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
    answer_options: Dict[int, AnswerOptions]
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
        json_instructions: str = prompt_templates.SYSTEM_JSON_DEFAULT,
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
            json_attributes=json_attributes,
            json_explanation=json_explanation,
            json_instructions=json_instructions,
        )
        json_appendix = creator.get_output_prompt()

        system_prompt = f"""{self.system_prompt}
{json_appendix}"""
        return system_prompt

    def get_response_generation_methods(self, question_id: int) -> ResponseGenerationMethod:
        return self.answer_options[question_id].response_generation_method


    def create_whole_response_generation_method(self) -> JSONResponseGenerationMethod:
            full_json_structure = []
            full_constraints = {}
            for i in range(len(self.answer_options)):
                response_generation_method = self.answer_options[self.order[i]].response_generation_method
                if isinstance(response_generation_method, JSONResponseGenerationMethod):
                    for json_element in response_generation_method.json_fields:
                        new_element = f"{json_element}{i}"
                        if response_generation_method.constraints:
                            constraints_element = (
                                response_generation_method.constraints.get(
                                    json_element
                                )
                            )
                            if constraints_element != None:
                                full_constraints[new_element] = constraints_element
                        full_json_structure.append(new_element)
            rgm_return = copy.deepcopy(response_generation_method)
            rgm_return.json_fields = full_json_structure
            rgm_return.constraints = full_constraints

            print(rgm_return.json_fields)
            print(rgm_return.constraints)
            return rgm_return

