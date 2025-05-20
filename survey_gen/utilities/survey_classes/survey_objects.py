from typing import List, Optional

from dataclasses import dataclass

class SurveyOptions:
    option_descriptions: List[str] = None
    from_to_scale: bool = False

    def __init__(self, option_descriptions: List[str], from_to_scale:bool):
        self.option_descriptions = option_descriptions
        self.from_to_scale = from_to_scale

    def create_options_str(self) -> str:
        #TODO ADD a number of predefined string options. Give the user the ability to dynamically adjust them.
        if not self.from_to_scale:
            options_prompt = f"""Options are: {', '.join(self.option_descriptions)}"""
        else:
            options_prompt = f"Options range from {self.option_descriptions[0]} to {self.option_descriptions[-1]}"
        return options_prompt

@dataclass
class SurveyQuestion:
    """Represents a single survey question."""
    question_id: int
    survey_question: str
    prompt: Optional[str] = None
    options: Optional[SurveyOptions] = None
    prefilled_answer: Optional[str] = None