from typing import List, Optional, NamedTuple, Dict

from ..utilities import prompt_templates

from dataclasses import dataclass


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
class InterviewItem:
    """Represents a single survey question."""

    item_id: int
    question_content: str
    question_stem: Optional[str] = None
    answer_options: Optional[AnswerOptions] = None
    prefilled_response: Optional[str] = None
