"""
Module for managing and conducting surveys using LLM models.

This module provides functions to conduct surveys in different ways:
- Single-item
- battery
- sequential

Usage example:
--------------

.. code-block:: python

    from qstn import survey_manager
    from qstn.prompt_builder import LLMPrompt
    from qstn.utilities import placeholder
    from vllm import LLM

    import pandas as pd

    questionnaire = [
        {"questionnaire_item_id": 1, "question_content": "The Democratic Party?"},
        {"questionnaire_item_id": 2, "question_content": "The Republican Party?"},
    ]
    party_questionnaire = pd.DataFrame(questionnaire)


    system_prompt = (
        "Act as if you were a black middle aged man from New York! "
        "Answer in a single short sentence!"
    )
    prompt = (
        "Please tell us how you feel about the following parties: "
        + placeholder.PROMPT_QUESTIONS
    )

    questionnaire = LLMPrompt(
        questionnaire_name="political_parties",
        questionnaire_source=party_questionnaire,
        system_prompt=system_prompt,
        prompt=prompt,
    )

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    chat_generator = LLM(model_id, max_model_len=5000, seed=42)

    results = survey_manager.conduct_survey_single_item(
        chat_generator,
        questionnaire,
        client_model_name=model_id,
        print_conversation=True,
        # We can use the same inference arguments for inference, as we would for vllm or OpenAI
        temperature=0.8,
        max_tokens=5000,
    )
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from .inference.response_generation import (
    JSONResponseGenerationMethod,
    ResponseGenerationMethod,
)
from .inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from .parser.llm_answer_parser import raw_responses
from .prompt_builder import LLMPrompt, QuestionnairePresentation
from .utilities import constants, utils
from .utilities.survey_objects import (
    InferenceResult,
    QuestionLLMResponseTuple,
)

if TYPE_CHECKING:
    from vllm import LLM

# @dataclass
# class GenerationFailure:
#     batch_index: int
#     error_type: str
#     error_message: str


def _normalize_llm_prompts(
    llm_prompts: LLMPrompt | list[LLMPrompt],
) -> list[LLMPrompt]:
    """Normalize a single prompt or list of prompts to a list."""
    if isinstance(llm_prompts, LLMPrompt):
        return [llm_prompts]
    return llm_prompts


def _initialize_question_response_pairs(
    llm_prompts: list[LLMPrompt],
) -> list[dict[int, QuestionLLMResponseTuple]]:
    """Initialize mutable per-questionnaire response maps."""
    return [{} for _ in llm_prompts]


def _iter_survey_steps(max_survey_length: int, print_progress: bool):
    """Yield step indices, optionally wrapped in tqdm."""
    if print_progress:
        return tqdm(range(max_survey_length), desc="Processing questionnaires")
    return range(max_survey_length)


def _get_current_batch(llm_prompts: list[LLMPrompt], i: int) -> dict[int, LLMPrompt]:
    """Return questionnaire batch that still has a question at step index i."""
    return {
        pos: questionnaire
        for pos, questionnaire in enumerate(llm_prompts)
        if len(questionnaire) > i
    }


def _normalize_generation_outputs(
    output: list[str],
    logprobs: list[Any] | None,
    reasoning_output: list[Any] | None,
    expected_size: int,
) -> tuple[list[str], list[Any], list[Any]]:
    """Normalize optional generation outputs to zip-safe lists."""
    if logprobs is None:
        logprobs = [None] * expected_size
    if reasoning_output is None:
        reasoning_output = [None] * expected_size
    return output, logprobs, reasoning_output


def _run_batch_generation(
    model: Union["LLM", AsyncOpenAI],
    system_messages: list[str],
    prompts: list[str],
    response_generation_methods: list[ResponseGenerationMethod] | None,
    client_model_name: str | None,
    api_concurrency: int,
    print_conversation: bool,
    print_progress: bool,
    seed: int,
    **generation_kwargs: Any,
) -> tuple[list[str], list[Any], list[Any]]:
    """Thin wrapper around `batch_generation` with normalized optional outputs."""
    output, logprobs, reasoning_output = batch_generation(
        model=model,
        system_messages=system_messages,
        prompts=prompts,
        response_generation_method=response_generation_methods,
        client_model_name=client_model_name,
        api_concurrency=api_concurrency,
        print_conversation=print_conversation,
        print_progress=print_progress,
        seed=seed,
        **generation_kwargs,
    )
    return _normalize_generation_outputs(
        output=output,
        logprobs=logprobs,
        reasoning_output=reasoning_output,
        expected_size=len(system_messages),
    )


def _finalize_survey_results(
    llm_prompts: list[LLMPrompt],
    question_llm_response_pairs: list[dict[int, QuestionLLMResponseTuple]],
) -> list[InferenceResult]:
    """Convert internal response maps to public `InferenceResult` objects."""
    return [
        InferenceResult(survey, question_llm_response_pairs[i])
        for i, survey in enumerate(llm_prompts)
    ]


def _store_question_responses(
    question_llm_response_pairs: list[dict[int, QuestionLLMResponseTuple]],
    survey_ids: Iterable[int],
    questions: list[str],
    answers: list[str],
    logprobs: list[Any],
    reasoning_output: list[Any],
    item_ids: Iterable[Any],
) -> None:
    """Store question-level answers in the mutable response map."""
    for survey_id, question, answer, logprob_answer, reasoning, item_id in zip(
        survey_ids,
        questions,
        answers,
        logprobs,
        reasoning_output,
        item_ids,
    ):
        question_llm_response_pairs[survey_id].update(
            {item_id: QuestionLLMResponseTuple(question, answer, logprob_answer, reasoning)}
        )


def _prepare_single_item_batch(
    current_batch: dict[int, LLMPrompt], i: int
) -> tuple[list[str], list[str], list[str], list[ResponseGenerationMethod | None]]:
    """Prepare messages and metadata for a single-item survey step."""
    system_messages, prompts = zip(
        *[
            questionnaire.get_prompt_for_questionnaire_type(
                QuestionnairePresentation.SINGLE_ITEM,
                questionnaire.get_question_item_id(i),
            )
            for questionnaire in current_batch.values()
        ]
    )

    questions = [
        questionnaire.generate_question_prompt(questionnaire_items=questionnaire.get_question(i))
        for questionnaire in current_batch.values()
    ]
    response_generation_methods = [
        (
            questionnaire.get_question(i).answer_options.response_generation_method
            if questionnaire.get_question(i).answer_options
            else None
        )
        for questionnaire in current_batch.values()
    ]

    return list(system_messages), list(prompts), questions, response_generation_methods


def _prepare_battery_batch(
    current_batch: dict[int, LLMPrompt], i: int, item_separator: str
) -> tuple[list[str], list[str], list[ResponseGenerationMethod | None]]:
    """Prepare messages and response-generation methods for battery mode."""
    system_messages, prompts = zip(
        *[
            questionnaire.get_prompt_for_questionnaire_type(
                QuestionnairePresentation.BATTERY,
                questionnaire.get_question_item_id(i),
                item_separator=item_separator,
            )
            for questionnaire in current_batch.values()
        ]
    )

    response_generation_methods: list[ResponseGenerationMethod | None] = []
    for questionnaire in current_batch.values():
        response_generation_method = None
        if questionnaire.get_question(i).answer_options:
            response_generation_method = questionnaire.get_question(
                i
            ).answer_options.response_generation_method
            if isinstance(response_generation_method, JSONResponseGenerationMethod):
                response_generation_method = (
                    response_generation_method.create_new_rgm_with_multiple_questions(
                        questions=list(questionnaire.questions)
                    )
                )
        response_generation_methods.append(response_generation_method)

    return list(system_messages), list(prompts), response_generation_methods


def _prepare_sequential_step(
    current_batch: dict[int, LLMPrompt], i: int
) -> tuple[list[str], list[str], list[str], list[ResponseGenerationMethod | None]]:
    """Prepare per-step prompts/questions/methods for sequential mode."""
    first_question: bool = i == 0

    if first_question:
        system_messages, prompts = zip(
            *[
                questionnaire.get_prompt_for_questionnaire_type(
                    QuestionnairePresentation.SEQUENTIAL,
                    questionnaire.get_question_item_id(i),
                )
                for questionnaire in current_batch.values()
            ]
        )
        questions = [
            questionnaire.generate_question_prompt(
                questionnaire_items=questionnaire.get_question(i)
            )
            for questionnaire in current_batch.values()
        ]
    else:
        system_messages, _ = zip(
            *[
                questionnaire.get_prompt_for_questionnaire_type(
                    QuestionnairePresentation.SEQUENTIAL,
                    questionnaire.get_question_item_id(i),
                )
                for questionnaire in current_batch.values()
            ]
        )
        prompts = [
            questionnaire.generate_question_prompt(
                questionnaire_items=questionnaire.get_question(i)
            )
            for questionnaire in current_batch.values()
        ]
        questions = prompts

    response_generation_methods = [
        (
            questionnaire.get_question(i).answer_options.response_generation_method
            if questionnaire.get_question(i).answer_options
            else None
        )
        for questionnaire in current_batch.values()
    ]

    return list(system_messages), list(prompts), questions, response_generation_methods


def conduct_survey_single_item(
    model: Union["LLM", AsyncOpenAI],
    llm_prompts: LLMPrompt | list[LLMPrompt],
    client_model_name: str | None = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    n_save_step: int | None = None,
    intermediate_save_file: str | None = None,
    seed: int = 42,
    **generation_kwargs: Any,
) -> list[InferenceResult]:
    """
    Conducts a survey by asking each question in a new context (single item presentation).

    System Prompt -> User Prompt with one question -> LLM Answer for one question
    -> Reset Context -> New instance with System Prompt

    Args:
        model (LLM or AsyncOpenAI): vllm.LLM instance or AsyncOpenAI client.
        llm_prompts (LLMPrompt or List(LLMPrompt)): Single LLMPrompt or list of LLMPrompt objects
            to conduct as a survey.
        client_model_name (str, optional): Name of model when using OpenAI client.
        api_concurrency (int): Number of concurrent API requests. Defaults to 10.
        print_conversation (bool): If True, prints all conversations to stdout. Default False.
        print_progress (bool): If True, shows a tqdm progress bar. Default True.
        n_save_step (int, optional): Save intermediate results every n steps.
        intermediate_save_file (str, optional): Path to save intermediate results.
            Has to be provided if n_save_step.
        seed (int): Random seed for reproducibility. Defaults to 42.
        generation_kwargs: Additional generation parameters that will be given to vllm.chat(),
            vllm.SamplingParams, or client.chat.completions.create().

    Returns:
        List(InferenceResult): A list of results containing the survey data and
            LLM responses for each provided prompt.
    """

    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    llm_prompts = _normalize_llm_prompts(llm_prompts)

    max_survey_length: int = max(len(questionnaire) for questionnaire in llm_prompts)
    question_llm_response_pairs = _initialize_question_response_pairs(llm_prompts)

    for i in _iter_survey_steps(max_survey_length, print_progress):
        current_batch = _get_current_batch(llm_prompts, i)

        (
            system_messages,
            prompts,
            questions,
            response_generation_methods,
        ) = _prepare_single_item_batch(current_batch, i)

        # TODO Implement Retrying for errors.
        # try:
        output, logprobs, reasoning_output = _run_batch_generation(
            model=model,
            system_messages=system_messages,
            prompts=prompts,
            response_generation_methods=response_generation_methods,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )
        # except Exception as e:
        #     warnings.warn(
        #         "Questions at position {i} could not be processed, because "
        #         "an error occured: {e}. Output is set to None"
        #     )
        #     output = [None] * len(current_batch)
        #     logprobs = None
        #     reasoning_output = [None] * len(current_batch)

        _store_question_responses(
            question_llm_response_pairs=question_llm_response_pairs,
            survey_ids=current_batch.keys(),
            questions=questions,
            answers=output,
            logprobs=logprobs,
            reasoning_output=reasoning_output,
            item_ids=[item.get_question_item_id(i) for item in current_batch.values()],
        )

        _intermediate_saves(
            llm_prompts,
            n_save_step,
            intermediate_save_file,
            question_llm_response_pairs,
            i,
        )

    return _finalize_survey_results(llm_prompts, question_llm_response_pairs)


def _intermediate_saves(
    questionnaires: list[LLMPrompt],
    n_save_step: int,
    intermediate_save_file: str,
    question_llm_response_pairs: QuestionLLMResponseTuple,
    i: int,
):
    """
    Internal helper to save intermediate survey results.

    Args:
        questionnaires: List of questionnaires being conducted.
        n_save_step: Save frequency in steps.
        intermediate_save_file: Path to save file.
        question_llm_response_pairs: Current responses.
        i: Current step number.
    """
    if n_save_step:
        if i % n_save_step == 0:
            intermediate_survey_results: list[InferenceResult] = []
            for j, questionnaire in enumerate(questionnaires):
                intermediate_survey_results.append(
                    InferenceResult(questionnaire, question_llm_response_pairs[j])
                )
            parsed_results = raw_responses(intermediate_survey_results)
            utils.create_one_dataframe(parsed_results).to_csv(intermediate_save_file)


def _intermediate_save_path_check(n_save_step: int, intermediate_save_path: str):
    """
    Internal helper to validate intermediate save path.

    Args:
        n_save_step: Save frequency in steps.
        intermediate_save_path: Path to check.
    """
    if n_save_step:
        if not isinstance(n_save_step, int) or n_save_step <= 0:
            raise ValueError("`n_save_step` must be a positive integer.")

        if not intermediate_save_path:
            raise ValueError("`intermediate_save_file` must be provided if saving is enabled.")

        if not intermediate_save_path.endswith(".csv"):
            raise ValueError("`intermediate_save_file` should be a .csv file.")

        # Ensure it's a directory that exists or can be created
        parent_dir = Path(intermediate_save_path).parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Invalid intermediate save path: {intermediate_save_path}. Error: {e}"
                ) from e

        # Optional: Check it's writable
        if not os.access(parent_dir, os.W_OK):
            raise ValueError(f"Save path '{intermediate_save_path}' is not writable.")


def conduct_survey_battery(
    model: Union["LLM", AsyncOpenAI],
    llm_prompts: LLMPrompt | list[LLMPrompt],
    client_model_name: str | None = None,
    api_concurrency: int = 10,
    n_save_step: int | None = None,
    intermediate_save_file: str | None = None,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    item_separator: str = "\n",
    **generation_kwargs: Any,
) -> list[InferenceResult]:
    """
    Conducts the entire survey in one single LLM prompt (battery presentation).

    System Prompt -> User Prompt with all questions -> LLM Answers all questions

    Args:
        model (LLM or AsyncOpenAI): vllm.LLM instance or AsyncOpenAI client.
        llm_prompts (LLMPrompt or List(LLMPrompt)): Single LLMPrompt or list
            of LLMPrompt objects to conduct as a survey.
        client_model_name (str, optional): Name of model when using OpenAI client.
        api_concurrency (int): Number of concurrent API requests. Defaults to 10.
        print_conversation (bool): If True, prints all conversations to stdout. Default False.
        print_progress (bool): If True, shows a tqdm progress bar. Default True.
        n_save_step (int, optional): Save intermediate results every n steps.
        intermediate_save_file (str, optional): Path to save intermediate
            results. Has to be provided if n_save_step.
        seed (int): Random seed for reproducibility. Defaults to 42.
        item_separator (str): The str that separates each question. Defaults to a newline.
        generation_kwargs: Additional generation parameters that will be given
            to vllm.chat(), vllm.SamplingParams, or
            client.chat.completions.create().

    Returns:
        List(InferenceResult): A list of results containing the survey data and
            LLM responses for each provided prompt.
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    llm_prompts = _normalize_llm_prompts(llm_prompts)
    # inference_options: List[InferenceOptions] = []

    # We always conduct the survey in one prompt
    max_survey_length: int = 1

    question_llm_response_pairs = _initialize_question_response_pairs(llm_prompts)

    for i in _iter_survey_steps(max_survey_length, print_progress):
        current_batch = _get_current_batch(llm_prompts, i)
        (
            system_messages,
            prompts,
            response_generation_methods,
        ) = _prepare_battery_batch(current_batch, i, item_separator)

        # TODO Implement Retrying for errors.
        # try:
        output, logprobs, reasoning_output = _run_batch_generation(
            model=model,
            system_messages=system_messages,
            prompts=prompts,
            response_generation_methods=response_generation_methods,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )
        # except Exception as e:
        #     warnings.warn(
        #         "Questions at position {i} could not be processed, because "
        #         "an error occured: {e}. Output is set to None"
        #     )
        #     output = [None] * len(current_batch)
        #     logprobs = None
        #     reasoning_output = [None] * len(current_batch)

        _store_question_responses(
            question_llm_response_pairs=question_llm_response_pairs,
            survey_ids=current_batch.keys(),
            questions=prompts,
            answers=output,
            logprobs=logprobs,
            reasoning_output=reasoning_output,
            item_ids=[-1] * len(output),
        )

        _intermediate_saves(
            llm_prompts,
            n_save_step,
            intermediate_save_file,
            question_llm_response_pairs,
            i,
        )

    return _finalize_survey_results(llm_prompts, question_llm_response_pairs)


def conduct_survey_sequential(
    model: Union["LLM", AsyncOpenAI],
    llm_prompts: LLMPrompt | list[LLMPrompt],
    client_model_name: str | None = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    n_save_step: int | None = None,
    intermediate_save_file: str | None = None,
    seed: int = 42,
    **generation_kwargs: Any,
) -> list[InferenceResult]:
    """
    Conducts the survey in multiple chat calls, where all questions and answers
    are kept in context (sequential presentation).

    System Prompt -> User Prompt with first question -> LLM Answer to first
    question -> User Prompt with second question -> ....

    Args:
        model (LLM or AsyncOpenAI): vllm.LLM instance or AsyncOpenAI client.
        llm_prompts (LLMPrompt or List(LLMPrompt)): Single LLMPrompt or list
            of LLMPrompt objects to conduct as a survey.
        client_model_name (str, optional): Name of model when using OpenAI client.
        api_concurrency (int): Number of concurrent API requests. Defaults to 10.
        print_conversation (bool): If True, prints all conversations to stdout. Default False.
        print_progress (bool): If True, shows a tqdm progress bar. Default True.
        n_save_step (int, optional): Save intermediate results every n steps.
        intermediate_save_file (str, optional): Path to save intermediate
            results. Has to be provided if n_save_step.
        seed (int): Random seed for reproducibility. Defaults to 42.
        generation_kwargs: Additional generation parameters that will be given
            to vllm.chat(), vllm.SamplingParams, or
            client.chat.completions.create().

    Returns:
        List(InferenceResult): A list of results containing the survey data and
            LLM responses for each provided prompt.
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)
    llm_prompts = _normalize_llm_prompts(llm_prompts)

    max_survey_length: int = max(len(questionnaire) for questionnaire in llm_prompts)

    question_llm_response = _initialize_question_response_pairs(llm_prompts)

    all_prompts: list[list[str]] = []
    assistant_messages: list[list[str]] = []

    for _ in llm_prompts:
        assistant_messages.append([])
        all_prompts.append([])

    for i in _iter_survey_steps(max_survey_length, print_progress):
        current_batch = _get_current_batch(llm_prompts, i)
        (
            system_messages,
            prompts,
            questions,
            response_generation_methods,
        ) = _prepare_sequential_step(current_batch, i)

        for c in range(len(current_batch.values())):
            all_prompts[c].append(prompts[c])

        current_assistant_messages: list[str] = []

        missing_indeces = []

        for index, surv in enumerate(current_batch.values()):
            prefilled_answer = surv.get_question(i).prefilled_response
            if prefilled_answer is not None:
                current_assistant_messages.append(prefilled_answer)
                missing_indeces.append(index)

        needed_batch = [
            item for a, item in enumerate(current_batch.values()) if a not in missing_indeces
        ]

        if len(needed_batch) == 0:
            for c in range(len(current_batch.values())):
                assistant_messages[c].append(current_assistant_messages[c])

            logprobs = [None] * len(current_batch.values())
            reasoning_output = [None] * len(current_batch.values())
            _store_question_responses(
                question_llm_response_pairs=question_llm_response,
                survey_ids=current_batch.keys(),
                questions=questions,
                answers=current_assistant_messages,
                logprobs=logprobs,
                reasoning_output=reasoning_output,
                item_ids=[item.get_question_item_id(i) for item in current_batch.values()],
            )
            continue
            # TODO: add support for automatic system prompt for other answer production methods

        # TODO Implement Retrying for errors.
        # try:
        output, logprobs, reasoning_output = batch_turn_by_turn_generation(
            model=model,
            system_messages=system_messages,
            prompts=all_prompts,
            assistant_messages=assistant_messages,
            response_generation_method=response_generation_methods,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )
        # except Exception as e:
        #     warnings.warn(
        #         "Questions at position {i} could not be processed, because "
        #         "an error occured: {e}. Output is set to None"
        #     )
        #     output = [None] * len(current_batch)
        #     logprobs = None
        #     reasoning_output = [None] * len(current_batch)

        if reasoning_output is None:
            reasoning_output = [None] * len(needed_batch)
        if logprobs is None or len(logprobs) == 0:
            logprobs = [None] * len(needed_batch)

        for num, index in enumerate(missing_indeces):
            output.insert(index, current_assistant_messages[num])
        _store_question_responses(
            question_llm_response_pairs=question_llm_response,
            survey_ids=current_batch.keys(),
            questions=questions,
            answers=output,
            logprobs=logprobs,
            reasoning_output=reasoning_output,
            item_ids=[item.get_question_item_id(i) for item in needed_batch],
        )

        for o, _ in enumerate(output):
            assistant_messages[o].append(output[o])
        # assistant_messages.append(output)

        _intermediate_saves(
            llm_prompts, n_save_step, intermediate_save_file, question_llm_response, i
        )

    return _finalize_survey_results(llm_prompts, question_llm_response)


class SurveyCreator:
    """Helper class to create LLM prompts from a population CSV/DataFrame and questionnaire."""

    @classmethod
    def from_path(cls, survey_path: str, questionnaire_path: str) -> list[LLMPrompt]:
        """
        Generates LLMPrompt objects from two CSV files (population/survey and questionnaire).
        Args:
            survey_path (str): The path to the survey CSV file.

        Returns:
            A list of LLMQuestionnaire objects.
        """
        df = pd.read_csv(survey_path)
        df_questionnaire = pd.read_csv(questionnaire_path)
        return cls._from_dataframe(df, df_questionnaire)

    @classmethod
    def from_dataframe(
        cls, survey_dataframe: pd.DataFrame, questionnaire_dataframe: pd.DataFrame
    ) -> list[LLMPrompt]:
        """
        Generates LLMPrompt objects from two pandas DataFrames.

        Args:
            survey_dataframe (pandas.DataFrame): A DataFrame containing survey
                data (questionnaire_name, system_prompt, and
                questionnaire_instruction).
            questionnaire_dataframe (pandas.DataFrame): A DataFrame containing the questions.

        Returns:
            A list of LLMQuestionnaire objects.
        """
        return cls._from_dataframe(survey_dataframe, questionnaire_dataframe)

    @classmethod
    def _create_questionnaire(cls, row: pd.Series, df_questionnaire) -> LLMPrompt:
        """
        Internal helper method to create the LLM Prompts.
        """
        return LLMPrompt(
            questionnaire_source=df_questionnaire,
            questionnaire_name=row[constants.QUESTIONNAIRE_NAME],
            system_prompt=row[constants.SYSTEM_PROMPT_FIELD],
            prompt=row[constants.QUESTIONNAIRE_INSTRUCTION_FIELD],
        )

    @classmethod
    def _from_dataframe(cls, df: pd.DataFrame, df_questionnaire: pd.DataFrame) -> list[LLMPrompt]:
        """
        Internal helper method to process the DataFrame.
        """
        questionnaires = df.apply(
            lambda row: cls._create_questionnaire(row, df_questionnaire), axis=1
        )
        return questionnaires.to_list()


if __name__ == "__main__":
    pass
