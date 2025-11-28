from typing import Final, List, Dict

from enum import Enum

# Survey Item
SYSTEM_PROMPT_FIELD: Final[str] = "system_prompt"
QUESTIONNAIRE_INSTRUCTION_FIELD: Final[str] = "questionnaire_instruction"
QUESTIONNAIRE_ITEM_ID: Final[str] = "questionnaire_item_id"
QUESTIONNAIRE_ITEM: Final[str] = "questionnaire_item"
QUESTIONNAIRE_NAME: Final[str] = "questionnaire_name"

# Question
QUESTION_STEM: Final[str] = "question_stem"
QUESTION_CONTENT: Final[str] = "question_content"
QUESTION: Final[str] = "question"

# Answer
ANSWER_CODE: Final[str] = "answer_code"
ANSWER_TEXT: Final[str] = "answer_text"
ANSWER_OPTION: Final[str] = "answer_option"

# LLM Response
LLM_RESPONSE: Final[str] = "llm_response"
PARSED_RESPONSE: Final[str] = "parsed_response"

# Structured Output constraints
OPTIONS_ADJUST: List[str] = ["OPTIONS_ADJUST"]


class QuestionnairePresentation(Enum):
    SINGLE_ITEM: str = "questionnaire_type_single_item"
    SEQUENTIAL: str = "questionnaire_type_sequential"
    BATTERY: str = "questionnaire_type_battery"


DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You will be given questions and possible answer options for each. Please reason about each question before answering."
)
DEFAULT_TASK_INSTRUCTION: Final[str] = ""
DEFAULT_JSON_STRUCTURE: Final[List[str]] = ["reasoning", "answer"]
DEFAULT_CONSTRAINTS: Final[Dict[str, List[str]]] = {"answer": OPTIONS_ADJUST}
DEFAULT_QUESTIONNAIRE_ID: Final[str] = "Questionnaire"
