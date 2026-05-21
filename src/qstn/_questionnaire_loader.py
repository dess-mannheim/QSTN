import ast
from enum import StrEnum
from typing import Any

import pandas as pd


class QuestionnaireLoaderColumn(StrEnum):
    """Column names understood by `LLMPrompt.load_questionnaire_format`."""

    ANSWER_CODES = "answer_codes"
    ANSWER_TEXTS = "answer_texts"
    CONSTRAIN_ANSWER_OPTIONS = "constrain_answer_options"
    INDEX_ANSWER_SEPARATOR = "index_answer_separator"
    LIKERT_ADD_MIDDLE_CATEGORY = "likert_add_middle_category"
    LIKERT_ADD_REFUSAL = "likert_add_refusal"
    LIKERT_EVEN_ORDER = "likert_even_order"
    LIKERT_IDX_TYPE = "likert_idx_type"
    LIKERT_MIDDLE_CATEGORY = "likert_middle_category"
    LIKERT_N = "likert_n"
    LIKERT_ONLY_FROM_TO_SCALE = "likert_only_from_to_scale"
    LIKERT_RANDOM_ORDER = "likert_random_order"
    LIKERT_REFUSAL_CODE = "likert_refusal_code"
    LIKERT_REVERSED_ORDER = "likert_reversed_order"
    LIKERT_START_IDX = "likert_start_idx"
    LIST_PROMPT_TEMPLATE = "list_prompt_template"
    OPTIONS_SEPARATOR = "options_separator"
    OUTPUT_INDEX_ONLY = "output_index_only"
    PREFILLED_RESPONSE = "prefilled_response"
    RESPONSE_GENERATION_METHOD = "response_generation_method"
    SCALE_PROMPT_TEMPLATE = "scale_prompt_template"


def is_missing_value(value: Any) -> bool:
    """Return True for empty spreadsheet cells without rejecting list objects."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def row_has_value(row: pd.Series, column: str) -> bool:
    return column in row.index and not is_missing_value(row[column])


def optional_row_value(row: pd.Series, column: str, default: Any = None) -> Any:
    if not row_has_value(row, column):
        return default
    return row[column]


def parse_bool(value: Any, column: str, item_id: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Column '{column}' for questionnaire_item_id '{item_id}' must be "
                "a Python bool literal: True or False."
            ) from exc
        if isinstance(parsed, bool):
            return parsed
    raise ValueError(
        f"Column '{column}' for questionnaire_item_id '{item_id}' must be "
        "a Python bool literal: True or False."
    )


def optional_bool(row: pd.Series, column: str, item_id: Any, default: bool = False) -> bool:
    if not row_has_value(row, column):
        return default
    return parse_bool(row[column], column, item_id)


def parse_int(value: Any, column: str, item_id: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"Column '{column}' for questionnaire_item_id '{item_id}' must be an integer."
        )
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Column '{column}' for questionnaire_item_id '{item_id}' must be an integer."
        ) from exc


def optional_int(
    row: pd.Series,
    column: str,
    item_id: Any,
    default: int | None = None,
) -> int | None:
    if not row_has_value(row, column):
        return default
    return parse_int(row[column], column, item_id)


def parse_python_list(value: Any, column: str, item_id: Any) -> list[str]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, tuple):
        parsed = list(value)
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Column '{column}' for questionnaire_item_id '{item_id}' must be a "
                "Python list literal."
            ) from exc
    else:
        raise ValueError(
            f"Column '{column}' for questionnaire_item_id '{item_id}' must be a Python list."
        )

    if not isinstance(parsed, list):
        raise ValueError(
            f"Column '{column}' for questionnaire_item_id '{item_id}' must be a Python list."
        )
    return [str(item) for item in parsed]


def optional_list(row: pd.Series, column: str, item_id: Any) -> list[str] | None:
    if not row_has_value(row, column):
        return None
    return parse_python_list(row[column], column, item_id)


def optional_template(row: pd.Series, column: str, default: str) -> str:
    value = optional_row_value(row, column, default)
    return default if value is None else str(value)
