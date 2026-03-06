import json
import warnings
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import json_repair
import numpy as np
import pandas as pd

from ..inference.response_generation import (
    JSONResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    ResponseGenerationMethod,
)
from ..inference.survey_inference import batch_generation
from ..prompt_builder import LLMPrompt
from ..utilities import constants
from ..utilities.survey_objects import (
    AnswerOptions,
    InferenceResult,
    QuestionLLMResponseTuple,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from vllm import LLM  # pyright: ignore[reportMissingImports]

DEFAULT_LLM_AS_A_JUDGE_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_LLM_AS_A_JUDGE_PROMPT: str = (
    "Your task is to parse the correct answer option from an open text "
    + "answer a LLM has given to survey questions. "
    + "You will be provided with the survey question, "
    + "possible answer options and the LLM answer.\n"
    + "{automatic_output_instructions}\n"
    + "{no_answer_option_instruction}"
    + "Question: {question}\n"
    + "Possible answer options: {answer_options}\n"
    + "Response by LLM: {llm_response}"
)
DEFAULT_LLM_AS_A_JUDGE_BATTERY_PROMPT: str = (
    "Your task is to parse answer options for multiple survey questions from one "
    + "aggregated LLM response. "
    + "{automatic_output_instructions}\n"
    + "{no_answer_option_instruction}"
    + "Questions: {question}\n"
    + "Possible answer options per question:\n{answer_options}\n"
    + "Aggregated response by LLM: {llm_response}"
)
SOURCE_LLM_RESPONSE_COLUMN: str = "source_llm_response"


def parse_json_str(answer: str) -> dict[str, str] | None:
    try:
        result_json = json.loads(answer)
    except Exception:
        try:
            result_json = json_repair.loads(answer, skip_json_loads=True)
        except Exception:
            return None

    return result_json


def _is_battery_like_result(survey_result: InferenceResult) -> bool:
    return len(survey_result.results) == 1 and -1 in survey_result.results


def _split_battery_results(
    survey_results: list[InferenceResult],
) -> tuple[list[InferenceResult], list[InferenceResult]]:
    battery_results: list[InferenceResult] = []
    non_battery_results: list[InferenceResult] = []
    for survey_result in survey_results:
        if _is_battery_like_result(survey_result):
            battery_results.append(survey_result)
        else:
            non_battery_results.append(survey_result)
    return battery_results, non_battery_results


def _parse_json_non_battery(
    survey_results: list[InferenceResult],
) -> dict[LLMPrompt, pd.DataFrame]:
    """Internal JSON parser for single-item/sequential style survey results."""
    final_result = {}

    for survey_result in survey_results:
        answers: list[pd.DataFrame] = []
        for key, value in survey_result.results.items():
            parsed_llm_response = parse_json_str(value.llm_response)
            reasoning = value.reasoning
            logprobs = value.logprobs
            if isinstance(parsed_llm_response, dict):
                for reserved_key in [constants.QUESTIONNAIRE_ITEM_ID, constants.QUESTION]:
                    if reserved_key in parsed_llm_response:
                        parsed_llm_response.pop(reserved_key)
                answer_format = parsed_llm_response.keys()

                row_data = [key, value.question, *parsed_llm_response.values()]
                row_columns = [
                    constants.QUESTIONNAIRE_ITEM_ID,
                    constants.QUESTION,
                    *answer_format,
                ]

                if reasoning is not None:
                    row_data.append(reasoning)
                    row_columns.append("built_in_reasoning")

                if logprobs is not None:
                    row_data.append(logprobs)
                    row_columns.append("logprobs")

                answers.append(pd.DataFrame(data=[row_data], columns=row_columns, index=[0]))
            else:
                answers.append(
                    pd.DataFrame(
                        data=[(key, value.question, value.llm_response, "ERROR: Parsing")],
                        columns=[
                            constants.QUESTIONNAIRE_ITEM_ID,
                            constants.QUESTION,
                            constants.LLM_RESPONSE,
                            "error_col",
                        ],
                        index=[0],
                    )
                )
        final_result[survey_result.questionnaire] = pd.concat(answers, ignore_index=True)

    return final_result


def parse_json(
    survey_results: list[InferenceResult],
) -> dict[LLMPrompt, pd.DataFrame]:
    """
    Parse JSON outputs of survey results with automatic battery routing.

    Battery-style results (`questionnaire_item_id == -1` as a single aggregated
    row) are routed to `parse_json_battery`; all others use standard JSON parsing.

    Args:
        survey_results (List[InferenceResult]): Survey results returned by
            survey conduction methods.

    Returns:
        Dict[LLMPrompt, pd.DataFrame]: Mapping from questionnaire to parsed
            dataframe. Battery-style inputs are returned in expanded
            per-question row format.
    """
    battery_results, non_battery_results = _split_battery_results(survey_results)
    all_results: dict[LLMPrompt, pd.DataFrame] = {}

    if non_battery_results:
        all_results.update(_parse_json_non_battery(non_battery_results))
    if battery_results:
        all_results.update(parse_json_battery(battery_results))

    return all_results


def parse_json_battery(
    survey_results: list[InferenceResult],
) -> dict[LLMPrompt, pd.DataFrame]:
    """
    Parse JSON outputs of battery-style survey results.

    Expects one aggregated response row per questionnaire (`item_id == -1`) with
    JSON keys ending in `_<question_content>`, and expands this into one row per
    questionnaire item.

    Args:
        survey_results (List[InferenceResult]): Battery-style survey results with
            one aggregated response (`item_id == -1`) per questionnaire.

    Returns:
        Dict[LLMPrompt, pd.DataFrame]: Mapping from questionnaire to expanded
            per-question dataframe.
    """
    parsed_results: dict[LLMPrompt, pd.DataFrame] = _parse_json_non_battery(survey_results)

    all_results = {}

    for survey, df in parsed_results.items():

        if "error_col" in df.columns:
            all_results[survey] = df
            continue

        source_row = df.iloc[0]

        grouped_items = {}

        for col_name, cell_value in source_row.items():
            survey_questions = survey.get_questions()
            for i in range(len(survey_questions)):
                current_question = survey_questions[i]
                current_id = current_question.item_id
                if col_name.endswith(f"_{current_question.question_content}"):
                    new_col_name = col_name.removesuffix(f"_{current_question.question_content}")
                    if current_id not in grouped_items:
                        grouped_items[current_id] = {constants.QUESTIONNAIRE_ITEM_ID: current_id}
                    grouped_items[current_id][constants.QUESTION] = survey.generate_question_prompt(
                        current_question
                    )
                    grouped_items[current_id][new_col_name] = cell_value

            final_data_list = list(grouped_items.values())

        all_results[survey] = pd.DataFrame(final_data_list)

        # long_df.loc[0:minimum_rows, constants.INTERVIEW_ITEM_ID] = [
        #     survey_question.item_id
        #     for survey_question in survey._questions[0:minimum_rows]
        # ]
        # long_df.loc[0:minimum_rows, constants.QUESTION] = [
        #     survey.generate_question_prompt(survey_question)
        #     for survey_question in survey._questions[0:minimum_rows]
        # ]
        # long_df = long_df.drop(columns=constants.INTERVIEW_ITEM_ID).rename(
        #     columns={"new_survey_item_id": constants.INTERVIEW_ITEM_ID}
        # )
        # all_results[survey] = long_df

    return all_results


def raw_responses(
    survey_results: list[InferenceResult],
) -> dict[LLMPrompt, pd.DataFrame]:
    """Organizes the questions and answers of a survey in a pandas Dataframe.
    Args:
        survey_results List[InterviewResult]: All results for all interviews.

    Returns:
        Dict[LLMInterview, pd.Dataframe]: A dictionary where the keys are the
            LLMInterviews and the values are a Dataframe with questions/answers.
    """

    all_results = {}
    for survey_result in survey_results:
        all_results[survey_result.questionnaire] = survey_result.to_dataframe()
    return all_results


def _validate_generation_output_lengths(
    output: list[str],
    logprobs: list[Any] | None,
    reasoning_output: list[Any] | None,
    expected_size: int,
) -> tuple[list[Any], list[Any]]:
    if len(output) != expected_size:
        raise ValueError(
            f"`generation_fn` returned {len(output)} outputs, expected {expected_size}."
        )

    if logprobs is None or len(logprobs) == 0:
        normalized_logprobs = [None] * expected_size
    else:
        if len(logprobs) != expected_size:
            raise ValueError(
                f"`generation_fn` returned {len(logprobs)} logprobs, expected {expected_size}."
            )
        normalized_logprobs = list(logprobs)

    if reasoning_output is None:
        normalized_reasoning_output = [None] * expected_size
    else:
        if len(reasoning_output) != expected_size:
            raise ValueError(
                f"`generation_fn` returned {len(reasoning_output)} reasoning outputs, "
                f"expected {expected_size}."
            )
        normalized_reasoning_output = list(reasoning_output)

    return normalized_logprobs, normalized_reasoning_output


def _is_questionnaire_answer_options_map(answer_options: dict[Any, Any]) -> bool:
    return any(isinstance(key, LLMPrompt) for key in answer_options)


def _resolve_answer_options_override_for_questionnaire(
    questionnaire: LLMPrompt,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ),
) -> AnswerOptions | dict[Any, AnswerOptions] | None:
    if answer_options is None or isinstance(answer_options, AnswerOptions):
        return answer_options

    if not isinstance(answer_options, dict):
        raise TypeError(
            "`answer_options` must be `None`, an `AnswerOptions`, a dict keyed by "
            "`questionnaire_item_id`, or a dict keyed by `LLMPrompt`."
        )

    if _is_questionnaire_answer_options_map(answer_options):
        scoped_answer_options = answer_options.get(questionnaire)
        if scoped_answer_options is None or isinstance(scoped_answer_options, AnswerOptions):
            return scoped_answer_options
        if isinstance(scoped_answer_options, dict):
            return scoped_answer_options
        raise TypeError(
            "Values in questionnaire-scoped `answer_options` must be `AnswerOptions`, "
            "dicts keyed by `questionnaire_item_id`, or `None`."
        )

    return answer_options


def _resolve_answer_options_for_question(
    item_id: Any,
    item_answer_options: AnswerOptions | None,
    scoped_answer_options: AnswerOptions | dict[Any, AnswerOptions] | None,
) -> AnswerOptions | None:
    if scoped_answer_options is None:
        return item_answer_options

    if isinstance(scoped_answer_options, AnswerOptions):
        return scoped_answer_options

    if isinstance(scoped_answer_options, dict):
        override = scoped_answer_options.get(item_id)
        if override is None:
            return item_answer_options
        if isinstance(override, AnswerOptions):
            return override
        raise TypeError(
            "Item-scoped `answer_options` values must be `AnswerOptions` instances or `None`."
        )

    raise TypeError(
        "`answer_options` must be `None`, an `AnswerOptions`, a dict keyed by "
        "`questionnaire_item_id`, or a dict keyed by `LLMPrompt`."
    )


def _build_answer_options_lookup(
    questionnaire: LLMPrompt,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
) -> dict[Any, str]:
    scoped_answer_options = _resolve_answer_options_override_for_questionnaire(
        questionnaire=questionnaire,
        answer_options=answer_options,
    )
    answer_options_lookup = {}
    for question in questionnaire.get_questions():
        options = ""
        effective_answer_options = _resolve_answer_options_for_question(
            item_id=question.item_id,
            item_answer_options=question.answer_options,
            scoped_answer_options=scoped_answer_options,
        )
        if effective_answer_options:
            options_str = effective_answer_options.create_options_str()
            options = options_str if options_str else ""
        answer_options_lookup[question.item_id] = options
    return answer_options_lookup


def _build_answer_choices_lookup(
    questionnaire: LLMPrompt,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
) -> dict[Any, list[str]]:
    scoped_answer_options = _resolve_answer_options_override_for_questionnaire(
        questionnaire=questionnaire,
        answer_options=answer_options,
    )
    answer_choices_lookup: dict[Any, list[str]] = {}
    for question in questionnaire.get_questions():
        choices: list[str] = []
        effective_answer_options = _resolve_answer_options_for_question(
            item_id=question.item_id,
            item_answer_options=question.answer_options,
            scoped_answer_options=scoped_answer_options,
        )
        if effective_answer_options:
            choices = list(effective_answer_options.answer_texts.full_answers)
        answer_choices_lookup[question.item_id] = choices
    return answer_choices_lookup


def _with_optional_no_answer_choice(
    choices: list[str],
    no_answer_option: str | None,
) -> list[str]:
    adjusted_choices = list(choices)
    if no_answer_option and no_answer_option not in adjusted_choices:
        adjusted_choices.append(no_answer_option)
    return adjusted_choices


def _is_options_adjust_placeholder(value: Any) -> bool:
    return value == constants.OPTIONS_ADJUST


def _build_no_answer_option_instruction(no_answer_option: str | None) -> str:
    if no_answer_option is None:
        return ""
    return (
        f'If no valid answer can be extracted, return exactly "{no_answer_option}" '
        + "for that question.\n"
    )


def _resolve_choices_for_json_key(
    key: str,
    questionnaire: LLMPrompt,
    answer_choices_lookup: dict[Any, list[str]],
) -> list[str]:
    questions = list(questionnaire.get_questions())

    for question in questions:
        suffix = f"_{question.question_content}"
        if key.endswith(suffix):
            return list(answer_choices_lookup.get(question.item_id, []))

    if key == "answer" and len(questions) == 1:
        return list(answer_choices_lookup.get(questions[0].item_id, []))

    all_choices: list[str] = []
    for question in questions:
        for choice in answer_choices_lookup.get(question.item_id, []):
            if choice not in all_choices:
                all_choices.append(choice)
    return all_choices


def _materialize_single_question_response_generation_method(
    response_generation_method: ResponseGenerationMethod | None,
    answer_choices: list[str],
    no_answer_option: str | None,
) -> ResponseGenerationMethod | None:
    if response_generation_method is None:
        return None

    if not isinstance(response_generation_method, JSONResponseGenerationMethod):
        return response_generation_method
    adjusted_method = deepcopy(response_generation_method)

    adjusted_choices = _with_optional_no_answer_choice(answer_choices, no_answer_option)

    if isinstance(adjusted_method.json_fields, dict):
        for key in adjusted_method.json_fields:
            if _is_options_adjust_placeholder(adjusted_method.json_fields[key]):
                adjusted_method.json_fields[key] = (
                    ", ".join(adjusted_choices) if adjusted_choices else "parsed answer text"
                )

    if adjusted_method.constraints:
        placeholder_keys_to_drop: list[str] = []
        for key in list(adjusted_method.constraints.keys()):
            if _is_options_adjust_placeholder(adjusted_method.constraints[key]):
                if adjusted_choices:
                    adjusted_method.constraints[key] = list(adjusted_choices)
                else:
                    placeholder_keys_to_drop.append(key)

        for key in placeholder_keys_to_drop:
            adjusted_method.constraints.pop(key, None)

        if len(adjusted_method.constraints) == 0:
            adjusted_method.constraints = None

    return adjusted_method


def _materialize_battery_response_generation_method(
    response_generation_method: ResponseGenerationMethod | None,
    questionnaire: LLMPrompt,
    no_answer_option: str | None,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
) -> ResponseGenerationMethod | None:
    if response_generation_method is None:
        return None

    if not isinstance(response_generation_method, JSONResponseGenerationMethod):
        return response_generation_method
    adjusted_method = deepcopy(response_generation_method)

    questions = list(questionnaire.get_questions())
    if isinstance(adjusted_method.json_fields, dict):
        json_keys = list(adjusted_method.json_fields.keys())
    else:
        json_keys = list(adjusted_method.json_fields)

    if len(questions) > 1:
        already_question_scoped = all(
            any(key.endswith(f"_{question.question_content}") for question in questions)
            for key in json_keys
        )
        if not already_question_scoped:
            adjusted_method = adjusted_method.create_new_rgm_with_multiple_questions(
                questions=questions
            )

    answer_choices_lookup = _build_answer_choices_lookup(
        questionnaire=questionnaire,
        answer_options=answer_options,
    )

    if isinstance(adjusted_method.json_fields, dict):
        for key in adjusted_method.json_fields:
            if _is_options_adjust_placeholder(adjusted_method.json_fields[key]):
                choices = _resolve_choices_for_json_key(key, questionnaire, answer_choices_lookup)
                adjusted_choices = _with_optional_no_answer_choice(choices, no_answer_option)
                adjusted_method.json_fields[key] = (
                    ", ".join(adjusted_choices) if adjusted_choices else "parsed answer text"
                )

    if adjusted_method.constraints:
        placeholder_keys_to_drop: list[str] = []
        for key in list(adjusted_method.constraints.keys()):
            if _is_options_adjust_placeholder(adjusted_method.constraints[key]):
                choices = _resolve_choices_for_json_key(key, questionnaire, answer_choices_lookup)
                adjusted_choices = _with_optional_no_answer_choice(choices, no_answer_option)
                if adjusted_choices:
                    adjusted_method.constraints[key] = adjusted_choices
                else:
                    placeholder_keys_to_drop.append(key)

        for key in placeholder_keys_to_drop:
            adjusted_method.constraints.pop(key, None)

        if len(adjusted_method.constraints) == 0:
            adjusted_method.constraints = None

    return adjusted_method


def _finalize_response_generation_methods(
    response_generation_methods: list[ResponseGenerationMethod | None],
) -> ResponseGenerationMethod | list[ResponseGenerationMethod] | None:
    if len(response_generation_methods) == 0:
        return None
    if all(method is None for method in response_generation_methods):
        return None
    return response_generation_methods


def _automatic_output_instructions(
    response_generation_method: ResponseGenerationMethod | None,
) -> str:
    if response_generation_method is None:
        return ""
    if not hasattr(response_generation_method, "get_automatic_prompt"):
        return ""
    return response_generation_method.get_automatic_prompt()


def _build_battery_answer_options_summary(
    questionnaire: LLMPrompt,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
) -> str:
    options_lookup = _build_answer_options_lookup(
        questionnaire=questionnaire,
        answer_options=answer_options,
    )
    option_lines: list[str] = []
    for question in questionnaire.get_questions():
        options = options_lookup.get(question.item_id, "")
        if options:
            option_lines.append(f"{question.question_content}: {options}")
    return "\n".join(option_lines)


def _build_default_battery_response_generation_methods(
    survey_results: list[InferenceResult],
    no_answer_option: str | None = None,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
) -> list[ResponseGenerationMethod]:
    methods: list[ResponseGenerationMethod] = []
    for survey_result in survey_results:
        answer_choices_lookup = _build_answer_choices_lookup(
            questionnaire=survey_result.questionnaire,
            answer_options=answer_options,
        )
        json_fields = {
            f"answer_{question.question_content}": "selected answer option"
            for question in survey_result.questionnaire.get_questions()
        }
        constraints: dict[str, list[str]] = {}
        for question in survey_result.questionnaire.get_questions():
            key = f"answer_{question.question_content}"
            adjusted_choices = _with_optional_no_answer_choice(
                answer_choices_lookup.get(question.item_id, []),
                no_answer_option,
            )
            if adjusted_choices:
                constraints[key] = adjusted_choices

        methods.append(
            JSONResponseGenerationMethod(
                json_fields=json_fields,
                constraints=constraints or None,
            )
        )
    return methods


def _run_llm_parser_jobs(
    model: "LLM | AsyncOpenAI",
    parser_jobs: list[dict[str, Any]],
    response_generation_method: ResponseGenerationMethod | list[ResponseGenerationMethod] | None,
    generation_fn: Callable[..., tuple[list[str], list[Any] | None, list[Any] | None]],
    system_prompt: str,
    client_model_name: str | None,
    api_concurrency: int,
    print_conversation: bool,
    print_progress: bool,
    seed: int,
    **generation_kwargs: Any,
) -> list[InferenceResult]:
    if len(parser_jobs) == 0:
        return []

    output, logprobs, reasoning_output = generation_fn(
        model=model,
        system_messages=[system_prompt] * len(parser_jobs),
        prompts=[item["prompt"] for item in parser_jobs],
        response_generation_method=response_generation_method,
        client_model_name=client_model_name,
        api_concurrency=api_concurrency,
        print_conversation=print_conversation,
        print_progress=print_progress,
        seed=seed,
        **generation_kwargs,
    )

    normalized_logprobs, normalized_reasoning_output = _validate_generation_output_lengths(
        output=output,
        logprobs=logprobs,
        reasoning_output=reasoning_output,
        expected_size=len(parser_jobs),
    )

    regrouped_results: dict[LLMPrompt, dict[Any, QuestionLLMResponseTuple]] = defaultdict(dict)
    for item, parsed_answer, item_logprobs, item_reasoning in zip(
        parser_jobs,
        output,
        normalized_logprobs,
        normalized_reasoning_output,
    ):
        regrouped_results[item["questionnaire"]][item["item_id"]] = QuestionLLMResponseTuple(
            question=item["question"],
            llm_response=parsed_answer,
            logprobs=item_logprobs,
            reasoning=item_reasoning,
        )

    return [
        InferenceResult(questionnaire=questionnaire, results=results)
        for questionnaire, results in regrouped_results.items()
    ]


def _add_source_response_column_by_item_id(
    parsed_results: dict[LLMPrompt, pd.DataFrame],
    source_response_map: dict[LLMPrompt, dict[Any, str]],
) -> None:
    for questionnaire, parsed_df in parsed_results.items():
        parsed_df[SOURCE_LLM_RESPONSE_COLUMN] = parsed_df[constants.QUESTIONNAIRE_ITEM_ID].map(
            source_response_map.get(questionnaire, {})
        )


def _add_source_response_column_by_questionnaire(
    parsed_results: dict[LLMPrompt, pd.DataFrame],
    source_response_map: dict[LLMPrompt, str],
) -> None:
    for questionnaire, parsed_df in parsed_results.items():
        parsed_df[SOURCE_LLM_RESPONSE_COLUMN] = source_response_map.get(questionnaire)


def _parse_with_llm_non_battery(
    model: "LLM | AsyncOpenAI",
    survey_results: list[InferenceResult],
    system_prompt: str = DEFAULT_LLM_AS_A_JUDGE_SYSTEM_PROMPT,
    prompt: str = DEFAULT_LLM_AS_A_JUDGE_PROMPT,
    response_generation_method: (
        ResponseGenerationMethod | list[ResponseGenerationMethod] | None
    ) = None,
    generation_fn: Callable[..., tuple[list[str], list[Any] | None, list[Any] | None]] = (
        batch_generation
    ),
    client_model_name: str | None = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    use_parser: bool = True,
    no_answer_option: str | None = None,
    seed: int = 42,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
    **generation_kwargs: Any,
) -> dict[LLMPrompt, pd.DataFrame]:
    parser_jobs: list[dict[str, Any]] = []
    source_response_map: dict[LLMPrompt, dict[Any, str]] = defaultdict(dict)
    resolved_response_generation_methods: list[ResponseGenerationMethod | None] = []
    options_lookup_cache: dict[LLMPrompt, dict[Any, str]] = {}
    answer_choices_lookup_cache: dict[LLMPrompt, dict[Any, list[str]]] = {}

    flat_items: list[tuple[LLMPrompt, Any, QuestionLLMResponseTuple]] = []
    for survey_result in survey_results:
        for item_id, qa_tuple in survey_result.results.items():
            flat_items.append((survey_result.questionnaire, item_id, qa_tuple))

    if isinstance(response_generation_method, list):
        if len(response_generation_method) != len(flat_items):
            raise ValueError(
                "`response_generation_method` must have the same length as the number "
                "of parser jobs."
            )
        source_methods: list[ResponseGenerationMethod | None] = list(response_generation_method)
    elif response_generation_method is None and use_parser:
        source_methods = [JSONSingleResponseGenerationMethod() for _ in flat_items]
    else:
        source_methods = [response_generation_method for _ in flat_items]

    for i, (questionnaire, item_id, qa_tuple) in enumerate(flat_items):
        if questionnaire not in options_lookup_cache:
            options_lookup_cache[questionnaire] = _build_answer_options_lookup(
                questionnaire=questionnaire,
                answer_options=answer_options,
            )
            answer_choices_lookup_cache[questionnaire] = _build_answer_choices_lookup(
                questionnaire=questionnaire,
                answer_options=answer_options,
            )

        options_lookup = options_lookup_cache[questionnaire]
        answer_choices_lookup = answer_choices_lookup_cache[questionnaire]
        current_method = _materialize_single_question_response_generation_method(
            source_methods[i],
            answer_choices=answer_choices_lookup.get(item_id, []),
            no_answer_option=no_answer_option,
        )
        parser_prompt = prompt.format(
            question=qa_tuple.question,
            llm_response=qa_tuple.llm_response,
            answer_options=options_lookup.get(item_id, ""),
            no_answer_option_instruction=_build_no_answer_option_instruction(no_answer_option),
            automatic_output_instructions=_automatic_output_instructions(current_method),
        )
        parser_jobs.append(
            {
                "questionnaire": questionnaire,
                "item_id": item_id,
                "question": qa_tuple.question,
                "prompt": parser_prompt,
            }
        )
        resolved_response_generation_methods.append(current_method)
        source_response_map[questionnaire][item_id] = qa_tuple.llm_response

    parser_survey_results = _run_llm_parser_jobs(
        model=model,
        parser_jobs=parser_jobs,
        response_generation_method=_finalize_response_generation_methods(
            resolved_response_generation_methods
        ),
        generation_fn=generation_fn,
        system_prompt=system_prompt,
        client_model_name=client_model_name,
        api_concurrency=api_concurrency,
        print_conversation=print_conversation,
        print_progress=print_progress,
        seed=seed,
        **generation_kwargs,
    )

    if use_parser:
        parsed_results = _parse_json_non_battery(parser_survey_results)
    else:
        parsed_results = raw_responses(parser_survey_results)

    _add_source_response_column_by_item_id(parsed_results, source_response_map)

    return parsed_results


def parse_with_llm(
    model: "LLM | AsyncOpenAI",
    survey_results: list[InferenceResult],
    system_prompt: str = DEFAULT_LLM_AS_A_JUDGE_SYSTEM_PROMPT,
    prompt: str = DEFAULT_LLM_AS_A_JUDGE_PROMPT,
    battery_prompt: str = DEFAULT_LLM_AS_A_JUDGE_BATTERY_PROMPT,
    response_generation_method: (
        ResponseGenerationMethod | list[ResponseGenerationMethod] | None
    ) = None,
    generation_fn: Callable[..., tuple[list[str], list[Any] | None, list[Any] | None]] = (
        batch_generation
    ),
    client_model_name: str | None = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    use_parser: bool = True,
    no_answer_option: str | None = None,
    seed: int = 42,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
    **generation_kwargs: Any,
) -> dict[LLMPrompt, pd.DataFrame]:
    """
    Parse free-text survey answers using LLM-as-a-judge with automatic battery routing.

    Battery-style results are routed to `parse_with_llm_battery`. Non-battery
    results use the regular single-item/sequential parser flow.

    Args:
        model (LLM or AsyncOpenAI): vLLM model or AsyncOpenAI client used for
            parser inference.
        survey_results (List[InferenceResult]): Survey results to parse.
        system_prompt (str): System prompt passed to parser inference.
        prompt (str): Prompt template for parser inference. Supports
            `{question}`, `{llm_response}`, `{answer_options}`,
            `{automatic_output_instructions}`, and
            `{no_answer_option_instruction}` placeholders.
        battery_prompt (str): Prompt template used for battery-style parser
            routing. Supports `{question}`, `{llm_response}`, `{answer_options}`,
            `{automatic_output_instructions}`, and
            `{no_answer_option_instruction}` placeholders.
        response_generation_method (
            ResponseGenerationMethod | List[ResponseGenerationMethod], optional
        ): Constraint for parser output. If `use_parser=True` and this is `None`,
            default JSON constraints are applied. If no answer options are
            available for a question, constraints for that field are dropped to
            avoid unsatisfiable schemas (free-text JSON value fallback).
        generation_fn (Callable): Generation function following the
            `batch_generation` output contract.
        client_model_name (str, optional): Model name for OpenAI client calls.
        api_concurrency (int): Max concurrent API requests for OpenAI calls.
        print_conversation (bool): Whether parser conversations are printed.
        print_progress (bool): Whether parser progress bars are shown.
        use_parser (bool): If `True`, parser outputs are post-processed into
            structured dataframes (`parse_json` / `parse_json_battery`). If
            `False`, raw parser model outputs are returned.
        no_answer_option (str, optional): Optional additional answer label that
            allows parser output to mark unanswered/unparseable cases.
        seed (int): Random seed for parser inference.
        answer_options (AnswerOptions | Dict[int, AnswerOptions] |
            Dict[LLMPrompt, AnswerOptions | Dict[int, AnswerOptions]], optional):
            Optional override for answer options used by parser prompts and
            JSON constraints. This is useful when original survey questions
            were run without embedded answer options.
        generation_kwargs: Additional generation kwargs passed to `generation_fn`.

    Returns:
        Dict[LLMPrompt, pd.DataFrame]: Mapping from questionnaire to parsed (or
            raw) dataframe. Includes `source_llm_response` for traceability.
    """
    battery_results, non_battery_results = _split_battery_results(survey_results)
    all_results: dict[LLMPrompt, pd.DataFrame] = {}

    if non_battery_results:
        all_results.update(
            _parse_with_llm_non_battery(
                model=model,
                survey_results=non_battery_results,
                system_prompt=system_prompt,
                prompt=prompt,
                response_generation_method=response_generation_method,
                generation_fn=generation_fn,
                client_model_name=client_model_name,
                api_concurrency=api_concurrency,
                print_conversation=print_conversation,
                print_progress=print_progress,
                use_parser=use_parser,
                no_answer_option=no_answer_option,
                seed=seed,
                answer_options=answer_options,
                **generation_kwargs,
            )
        )

    if battery_results:
        selected_battery_prompt = battery_prompt
        if (
            prompt != DEFAULT_LLM_AS_A_JUDGE_PROMPT
            and battery_prompt == DEFAULT_LLM_AS_A_JUDGE_BATTERY_PROMPT
        ):
            selected_battery_prompt = prompt

        all_results.update(
            parse_with_llm_battery(
                model=model,
                survey_results=battery_results,
                system_prompt=system_prompt,
                prompt=selected_battery_prompt,
                response_generation_method=response_generation_method,
                generation_fn=generation_fn,
                client_model_name=client_model_name,
                api_concurrency=api_concurrency,
                print_conversation=print_conversation,
                print_progress=print_progress,
                use_parser=use_parser,
                no_answer_option=no_answer_option,
                seed=seed,
                answer_options=answer_options,
                **generation_kwargs,
            )
        )

    return all_results


def parse_with_llm_battery(
    model: "LLM | AsyncOpenAI",
    survey_results: list[InferenceResult],
    system_prompt: str = DEFAULT_LLM_AS_A_JUDGE_SYSTEM_PROMPT,
    prompt: str = DEFAULT_LLM_AS_A_JUDGE_BATTERY_PROMPT,
    response_generation_method: (
        ResponseGenerationMethod | list[ResponseGenerationMethod] | None
    ) = None,
    generation_fn: Callable[..., tuple[list[str], list[Any] | None, list[Any] | None]] = (
        batch_generation
    ),
    client_model_name: str | None = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    use_parser: bool = True,
    no_answer_option: str | None = None,
    seed: int = 42,
    answer_options: (
        AnswerOptions | dict[Any, AnswerOptions | dict[Any, AnswerOptions] | None] | None
    ) = None,
    **generation_kwargs: Any,
) -> dict[LLMPrompt, pd.DataFrame]:
    """
    Parse battery-style aggregated survey answers using LLM-as-a-judge.

    If `use_parser=True`, outputs are expanded to per-question rows via
    `parse_json_battery`. If `use_parser=False`, the single aggregated row is
    returned (raw parser output).

    Args:
        model (LLM or AsyncOpenAI): vLLM model or AsyncOpenAI client used for
            parser inference.
        survey_results (List[InferenceResult]): Battery-style survey results with
            one aggregated response (`item_id == -1`) per questionnaire.
        system_prompt (str): System prompt passed to parser inference.
        prompt (str): Prompt template for parser inference. Supports
            `{question}`, `{llm_response}`, `{answer_options}`,
            `{automatic_output_instructions}`, and
            `{no_answer_option_instruction}` placeholders.
        response_generation_method (
            ResponseGenerationMethod | List[ResponseGenerationMethod], optional
        ): Constraint for parser output. If `use_parser=True` and this is `None`,
            a default battery-aware JSON constraint is created. If no answer
            options are available for a question, constraints for that field are
            dropped to avoid unsatisfiable schemas (free-text JSON value
            fallback).
        generation_fn (Callable): Generation function following the
            `batch_generation` output contract.
        client_model_name (str, optional): Model name for OpenAI client calls.
        api_concurrency (int): Max concurrent API requests for OpenAI calls.
        print_conversation (bool): Whether parser conversations are printed.
        print_progress (bool): Whether parser progress bars are shown.
        use_parser (bool): If `True`, parser outputs are expanded with
            `parse_json_battery`. If `False`, raw aggregated parser output is
            returned.
        no_answer_option (str, optional): Optional additional answer label that
            allows parser output to mark unanswered/unparseable cases.
        seed (int): Random seed for parser inference.
        answer_options (AnswerOptions | Dict[int, AnswerOptions] |
            Dict[LLMPrompt, AnswerOptions | Dict[int, AnswerOptions]], optional):
            Optional override for answer options used by parser prompts and
            JSON constraints. This is useful when original survey questions
            were run without embedded answer options.
        generation_kwargs: Additional generation kwargs passed to `generation_fn`.

    Returns:
        Dict[LLMPrompt, pd.DataFrame]: Mapping from questionnaire to parsed (or
            raw) dataframe. Includes `source_llm_response`.

    Raises:
        ValueError: If any input result is not battery-style.
    """
    if any(not _is_battery_like_result(survey_result) for survey_result in survey_results):
        raise ValueError(
            "`parse_with_llm_battery` expects battery-style survey results with one "
            "aggregated response (`questionnaire_item_id == -1`) per questionnaire."
        )

    parser_jobs: list[dict[str, Any]] = []
    source_response_map: dict[LLMPrompt, str] = {}
    resolved_response_generation_methods: list[ResponseGenerationMethod | None] = []

    if isinstance(response_generation_method, list):
        if len(response_generation_method) != len(survey_results):
            raise ValueError(
                "`response_generation_method` must have the same length as battery parser jobs."
            )
        source_methods: list[ResponseGenerationMethod | None] = list(response_generation_method)
    elif response_generation_method is None and use_parser:
        source_methods = _build_default_battery_response_generation_methods(
            survey_results,
            no_answer_option=no_answer_option,
            answer_options=answer_options,
        )
    else:
        source_methods = [response_generation_method for _ in survey_results]

    for i, survey_result in enumerate(survey_results):
        qa_tuple = survey_result.results[-1]
        current_method = _materialize_battery_response_generation_method(
            source_methods[i],
            questionnaire=survey_result.questionnaire,
            no_answer_option=no_answer_option,
            answer_options=answer_options,
        )
        parser_prompt = prompt.format(
            question=qa_tuple.question,
            llm_response=qa_tuple.llm_response,
            answer_options=_build_battery_answer_options_summary(
                questionnaire=survey_result.questionnaire,
                answer_options=answer_options,
            ),
            no_answer_option_instruction=_build_no_answer_option_instruction(no_answer_option),
            automatic_output_instructions=_automatic_output_instructions(current_method),
        )
        parser_jobs.append(
            {
                "questionnaire": survey_result.questionnaire,
                "item_id": -1,
                "question": qa_tuple.question,
                "prompt": parser_prompt,
            }
        )
        resolved_response_generation_methods.append(current_method)
        source_response_map[survey_result.questionnaire] = qa_tuple.llm_response

    parser_survey_results = _run_llm_parser_jobs(
        model=model,
        parser_jobs=parser_jobs,
        response_generation_method=_finalize_response_generation_methods(
            resolved_response_generation_methods
        ),
        generation_fn=generation_fn,
        system_prompt=system_prompt,
        client_model_name=client_model_name,
        api_concurrency=api_concurrency,
        print_conversation=print_conversation,
        print_progress=print_progress,
        seed=seed,
        **generation_kwargs,
    )

    if use_parser:
        parsed_results = parse_json_battery(parser_survey_results)
    else:
        parsed_results = raw_responses(parser_survey_results)

    _add_source_response_column_by_questionnaire(parsed_results, source_response_map)

    return parsed_results


def _filter_logprobs_by_choices(logprob_df: pd.DataFrame, choices: pd.Series) -> pd.DataFrame:

    matches_found = []

    # check for each output token whether any of the choices start with this token
    for token in logprob_df["token"]:
        boolean_index = choices.str.startswith(token)
        # if len(choices[boolean_index]) > 1:
        #    warnings.warn(
        #        "Multiple allowed_choices "
        #        f"({list(choices[boolean_index])}) match "
        #        f"the same output token: {token}",
        #        stacklevel=2
        #    )
        matches_found.append(boolean_index.any())

    return logprob_df[matches_found]


def _logprobs_filter(
    logprobs: dict[str, float], allowed_choices: dict[str, list[str]]
) -> dict[str, float]:

    # normalize logprobs
    logprob_df = pd.DataFrame({"token": logprobs.keys(), "prob": logprobs.values()})
    logprob_df["prob"] = logprob_df.prob.apply(np.exp)
    logprob_df = logprob_df[logprob_df.prob > 0]

    # flatten to check for collisions between answer options
    # TODO: implement this properly.
    # Only collisions between answer options matter, not, e.g., TRUMP vs. trump!
    # all_valid_outputs = [output for choices in allowed_choices.values() for output in choices]
    # _ = _filter_logprobs_by_choices(logprob_df, pd.Series(all_valid_outputs))

    # filter the individual survey answers
    choice_results = {}
    for choice, valid_outputs in allowed_choices.items():
        valid_logprobs = _filter_logprobs_by_choices(logprob_df, pd.Series(valid_outputs))
        if len(valid_logprobs) == 0:
            warnings.warn(
                "Could not find logprobs for answer option "
                f"'{choice}' with possible outputs {valid_outputs}",
                stacklevel=2,
            )
            choice_results[choice] = np.nan
        else:
            choice_results[choice] = valid_logprobs["prob"].sum()

    # normalize so that probs sum up to 1
    overall_sum = sum(
        [_result for _result in choice_results.values() if not np.isnan(_result)]
    )  # only consider values != nan
    if not np.isnan(overall_sum) and overall_sum > 0:
        choice_results = {
            choice: token_sum / overall_sum for choice, token_sum in choice_results.items()
        }

    return choice_results


def parse_logprobs(
    survey_results: list[InferenceResult],
    allowed_choices: list[str] | dict[str, list[str]],
) -> dict[LLMPrompt, pd.DataFrame]:
    """
    Filter and aggregate logprobs returned by Logprob_AnswerProductionMethod.

    Args:
        survey_results: List of InterviewResult that is returned from running a survey
        allowed_choices: List of possible answer options OR dictionary mapping
            options to multiple tokens that encode each option

    Returns:
        Dict[LLMInterview, pd.Dataframe]: A dictionary where the keys are the
            LLMInterviews and the values are a Dataframe with questions/answers.
    """
    final_result = {}

    # if each choice only maps to one token
    if isinstance(allowed_choices, list):
        allowed_choices = {c: [c] for c in allowed_choices}
    answer_format = list(allowed_choices.keys())

    for survey_result in survey_results:
        answers = []
        missing_logprobs = False
        for item_id, qa_tuple in survey_result.results.items():
            if qa_tuple.logprobs is None:
                warnings.warn(
                    "No logprobs found in InterviewResult. "
                    + "Make sure to use Logprob_AnswerProductionMethod to generate logprobs.",
                    stacklevel=2,
                )
                missing_logprobs = True
                answers.append((item_id, qa_tuple.question, *([np.nan] * len(answer_format))))
            else:
                filtered_logprobs = _logprobs_filter(qa_tuple.logprobs, allowed_choices)
                answers.append(
                    (
                        item_id,
                        qa_tuple.question,
                        *[filtered_logprobs.get(choice, np.nan) for choice in answer_format],
                    )
                )

        df = pd.DataFrame(
            answers,
            columns=[
                constants.QUESTIONNAIRE_ITEM_ID,
                constants.QUESTION,
                *answer_format,
            ],
        )
        if missing_logprobs:
            missing_mask = df[answer_format].isna().all(axis=1)
            df["error_col"] = [
                "MISSING_LOGPROBS" if is_missing else None for is_missing in missing_mask
            ]

        final_result[survey_result.questionnaire] = df

    return final_result
