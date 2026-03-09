"""Helpers for resolving and merging response-generation methods in battery mode."""

import copy
from typing import Any

from ..utilities.survey_objects import QuestionnaireItem
from .response_generation import (
    JSONResponseGenerationMethod,
    JSONVerbalizedDistribution,
    ResponseGenerationMethod,
)


def _scope_key_for_question(
    key: str,
    question: QuestionnaireItem,
    total_questions: int,
) -> str:
    if total_questions <= 1:
        return key
    qid_suffix = f"__qid_{question.item_id}"
    if key.endswith(qid_suffix):
        return key
    return f"{key}{qid_suffix}"


def merge_battery_json_response_generation_methods(
    questions: list[QuestionnaireItem],
    question_method_pairs: list[tuple[QuestionnaireItem, JSONResponseGenerationMethod]],
    fallback_method: JSONResponseGenerationMethod,
) -> JSONResponseGenerationMethod:
    """Merge per-question JSON methods into one deterministic battery JSON method."""
    question_order_by_id = {
        question.item_id: question_order
        for question_order, question in enumerate(questions, start=1)
    }

    scoped_fields_per_question: list[list[tuple[str, str | None]]] = []
    scoped_constraints_per_question: list[list[tuple[str, Any]]] = []

    for question, method in question_method_pairs:
        question_order = question_order_by_id.get(question.item_id)
        if isinstance(method, JSONVerbalizedDistribution):
            options = list(method.verbalized_options)
            if len(options) == 0 and isinstance(method.json_fields, dict):
                options = list(method.json_fields.keys())

            scoped_fields: list[tuple[str, str | None]] = []
            scoped_constraints: list[tuple[str, Any]] = []
            for option_index, option in enumerate(options, start=1):
                field_key = method._format_field(
                    option=option,
                    question=method._question_for_field(
                        question=str(question.question_content),
                        option_index=option_index,
                    ),
                    option_index=option_index,
                    question_id=question.item_id,
                    question_order=question_order,
                )
                explanation = method._format_explanation(
                    option=option,
                    question=str(question.question_content),
                    option_index=option_index,
                    question_id=question.item_id,
                    question_order=question_order,
                )
                scoped_fields.append((field_key, explanation))
                scoped_constraints.append((field_key, "float"))
            scoped_fields_per_question.append(scoped_fields)
            scoped_constraints_per_question.append(scoped_constraints)
            continue

        if isinstance(method.json_fields, dict):
            raw_fields: list[tuple[str, str | None]] = list(method.json_fields.items())
        else:
            raw_fields = [(field, None) for field in method.json_fields]

        scoped_fields: list[tuple[str, str | None]] = []
        for key, explanation in raw_fields:
            scoped_key = _scope_key_for_question(
                key=key,
                question=question,
                total_questions=len(questions),
            )
            scoped_fields.append((scoped_key, explanation))
        scoped_fields_per_question.append(scoped_fields)

        raw_constraints = method.constraints or {}
        scoped_constraints: list[tuple[str, Any]] = []
        for key, value in raw_constraints.items():
            scoped_key = _scope_key_for_question(
                key=key,
                question=question,
                total_questions=len(questions),
            )
            scoped_constraints.append((scoped_key, copy.deepcopy(value)))
        scoped_constraints_per_question.append(scoped_constraints)

    ordered_fields = [
        entry for scoped_fields in scoped_fields_per_question for entry in scoped_fields
    ]
    ordered_constraints = [
        entry
        for scoped_constraints in scoped_constraints_per_question
        for entry in scoped_constraints
    ]

    field_keys = [key for key, _ in ordered_fields]
    if len(field_keys) != len(set(field_keys)):
        raise ValueError(
            "Merged battery JSON fields contain duplicate keys. "
            "Please ensure each field template resolves to unique keys; "
            "for verbalized distributions include `{question_order}`, `{question_id}`, "
            "and/or `{option_index}`."
        )

    if all(explanation is not None for _, explanation in ordered_fields):
        json_fields: list[str] | dict[str, str] = {
            key: explanation for key, explanation in ordered_fields
        }
    else:
        json_fields = [key for key, _ in ordered_fields]

    constraints: dict[str, Any] | None = None
    if len(ordered_constraints) > 0:
        constraints = {key: value for key, value in ordered_constraints}

    return JSONResponseGenerationMethod(
        json_fields=json_fields,
        constraints=constraints,
        output_template=fallback_method.output_template,
        output_index_only=fallback_method.output_index_only,
    )


def resolve_battery_response_generation_method(
    questions: list[QuestionnaireItem],
    item_position: int = 0,
) -> tuple[ResponseGenerationMethod | None, bool]:
    """
    Resolve the response-generation method to use for battery prompts.

    Returns:
        Tuple[ResponseGenerationMethod | None, bool]:
            The resolved method and whether it is already merged/scoped
            across all questions.
    """
    if len(questions) == 0:
        return None, False

    safe_item_position = min(max(item_position, 0), len(questions) - 1)
    selected_question = questions[safe_item_position]
    fallback_method = None
    if selected_question.answer_options:
        fallback_method = selected_question.answer_options.response_generation_method

    question_method_pairs: list[tuple[QuestionnaireItem, ResponseGenerationMethod]] = []
    for question in questions:
        if question.answer_options and question.answer_options.response_generation_method:
            question_method_pairs.append(
                (question, question.answer_options.response_generation_method)
            )

    if len(question_method_pairs) == 0:
        return None, False

    if not all(
        isinstance(method, JSONResponseGenerationMethod) for _, method in question_method_pairs
    ):
        return fallback_method or question_method_pairs[0][1], False

    json_question_method_pairs: list[tuple[QuestionnaireItem, JSONResponseGenerationMethod]] = [
        (question, method)
        for question, method in question_method_pairs
        if isinstance(method, JSONResponseGenerationMethod)
    ]

    if fallback_method and isinstance(fallback_method, JSONResponseGenerationMethod):
        base_method = fallback_method
    else:
        base_method = json_question_method_pairs[0][1]

    merged_method = merge_battery_json_response_generation_methods(
        questions=questions,
        question_method_pairs=json_question_method_pairs,
        fallback_method=base_method,
    )
    return merged_method, True
