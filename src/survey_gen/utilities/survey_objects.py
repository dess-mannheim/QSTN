from typing import List, Optional, NamedTuple, Dict

from dataclasses import dataclass

class AnswerOptions:
    answer_text: List[str] = None
    from_to_scale: bool = False

    def __init__(self, option_descriptions: List[str], from_to_scale:bool):
        self.answer_text = option_descriptions
        self.from_to_scale = from_to_scale

    def create_options_str(self) -> str:
        #TODO ADD a number of predefined string options. Give the user the ability to dynamically adjust them.
        if not self.from_to_scale:
            options_prompt = f"""Options are: {', '.join(self.answer_text)}"""
        else:
            options_prompt = f"Options range from {self.answer_text[0]} to {self.answer_text[-1]}"
        return options_prompt

class QuestionLLMResponseTuple(NamedTuple):
    question: str
    llm_response: str

@dataclass
class SurveyItem:
    """Represents a single survey question."""
    item_id: int
    question_content: str
    question_stem: Optional[str] = None
    answer_options: Optional[AnswerOptions] = None
    prefilled_response: Optional[str] = None