from typing import Final, List, Dict

from enum import Enum

# Survey Item
INTERVIEW_ITEM_ID: Final[str] = "interview_item_id"
INTERVIEW_ITEM: Final[str] = "interview_item"
INTERVIEW_NAME: Final[str] = "interview_name"

# Question
QUESTION_STEM: Final[str] = "question_stem"
QUESTION_CONTENT: Final[str] = "question_content"
QUESTION_CONTENT_PLACEHOLDER: Final[str] = "QUESTION_CONTENT_PLACEHOLDER"
QUESTION: Final[str] = "question"

# Answer
ANSWER_CODE: Final[str] = "answer_code"
ANSWER_TEXT: Final[str] = "answer_text"
ANSWER_OPTION: Final[str] = "answer_option"

# LLM Response
LLM_RESPONSE: Final[str] = "llm_response"
PARSED_RESPONSE: Final[str] = "parsed_response"

#Structured Output constraints
OPTIONS_ADJUST: List[str] = ["OPTIONS_ADJUST"]


class InterviewType(Enum):
    QUESTION: str = "interview_type_question"
    CONTEXT: str = "interview_type_context"
    ONE_PROMPT: str = "interview_type_one_prompt"


DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You will be given questions and possible answer options for each. Please reason about each question before answering."
)
DEFAULT_TASK_INSTRUCTION: Final[str] = ""
DEFAULT_JSON_STRUCTURE: Final[List[str]] = ["reasoning", "answer"]
DEFAULT_CONSTRAINTS: Final[Dict[str, List[str]]] = {"answer": OPTIONS_ADJUST}
DEFAULT_INTERVIEW_ID: Final[str] = "Interview"