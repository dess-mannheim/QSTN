"""
Module for managing and conducting surveys using LLM models.

This module provides functions to conduct surveys in different ways:
- Single-item
- battery
- sequential

Usage example:
-------------
```python
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


system_prompt = "Act as if you were a black middle aged man from New York! Answer in a single short sentence!"
prompt = f"Please tell us how you feel about the following parties:\n{placeholder.PROMPT_QUESTIONS}"

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
```
"""

from typing import (
    List,
    Dict,
    Optional,
    Union,
    Any,
)

from .utilities.survey_objects import (
    QuestionLLMResponseTuple,
)
from .utilities import constants
from .utilities import utils

from .parser.llm_answer_parser import raw_responses

from .inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from .inference.response_generation import (
    ResponseGenerationMethod,
    JSONResponseGenerationMethod,
)

from .prompt_builder import LLMPrompt, QuestionnairePresentation

from .utilities.survey_objects import InferenceResult

from vllm import LLM

from openai import AsyncOpenAI

from pathlib import Path
import os

import pandas as pd


from tqdm.auto import tqdm


def conduct_survey_single_item(
    model: Union[LLM, AsyncOpenAI],
    llm_prompts: Union[LLMPrompt, List[LLMPrompt]],
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    seed: int = 42,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Dict[str, Any] = {},
    **generation_kwargs: Any,
) -> List[InferenceResult]:
    """
    Conducts a survey by asking each question in a new context (Single Item presentation).

    Args:
        model: vllm.LLM instance or AsyncOpenAI client.
        llm_prompts: Single LLMPrompt or list of LLMPrompt objects to conduct as a survey.
        client_model_name: Name of model when using OpenAI client.
        api_concurrency: Number of concurrent API requests.
        print_conversation: If True, prints all conversations to stdout.
        print_progress: If True, shows a tqdm progress bar.
        n_save_step: Save intermediate results every n steps.
        intermediate_save_file: Path to save intermediate results.
        seed: Random seed for reproducibility.
        chat_template: Optionally pass a specific chat template string.
        chat_template_kwargs: Arguments to pass to the chat template (e.g., to disable reasoning).
        **generation_kwargs: Additional generation parameters that will be given to vllm.chat() or client.chat.completions.create().

    Returns:
        List[InferenceResult]: A list of results containing the survey data and LLM responses for each provided prompt.
    """

    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    if isinstance(llm_prompts, LLMPrompt):
        llm_prompts = [llm_prompts]

    max_survey_length: int = max(len(questionnaire._questions) for questionnaire in llm_prompts)
    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    for i in range(len(llm_prompts)):
        # inference_option = interviews[i]._generate_inference_options()
        # inference_options.append(inference_opti
        question_llm_response_pairs.append({})

    survey_results: List[InferenceResult] = []

    for i in (
        tqdm(range(max_survey_length), desc="Processing questionnaires")
        if print_progress
        else range(max_survey_length)
    ):
        current_batch: List[LLMPrompt] = [
            questionnaire for questionnaire in llm_prompts if len(questionnaire._questions) > i
        ]

        system_messages, prompts = zip(
            *[
                questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.SINGLE_ITEM, i)
                for questionnaire in current_batch
            ]
        )

        questions = [
            questionnaire.generate_question_prompt(
                questionnaire_items=questionnaire._questions[i]
            )
            for questionnaire in current_batch
        ]
        response_generation_methods = [
            (
                questionnaire._questions[i].answer_options.response_generation_method
                if questionnaire._questions[i].answer_options
                else None
            )
            for questionnaire in current_batch
        ]

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
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            **generation_kwargs,
        )

        # avoid errors when zipping
        if logprobs is None:
            logprobs = [None] * len(current_batch)

        for survey_id, question, answer, logprob_answer, reasoning, item in zip(
            range(len(current_batch)),
            questions,
            output,
            logprobs,
            reasoning_output,
            current_batch,
        ):
            question_llm_response_pairs[survey_id].update(
                {
                    item._questions[i].item_id: QuestionLLMResponseTuple(
                        question, answer, logprob_answer, reasoning
                    )
                }
            )

        _intermediate_saves(
            llm_prompts,
            n_save_step,
            intermediate_save_file,
            question_llm_response_pairs,
            i,
        )

    for i, survey in enumerate(llm_prompts):
        survey_results.append(InferenceResult(survey, question_llm_response_pairs[i]))

    return survey_results


def _intermediate_saves(
    questionnaires: List[LLMPrompt],
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
            intermediate_survey_results: List[InferenceResult] = []
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
            raise ValueError(
                "`intermediate_save_file` must be provided if saving is enabled."
            )

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
                )

        # Optional: Check it's writable
        if not os.access(parent_dir, os.W_OK):
            raise ValueError(f"Save path '{intermediate_save_path}' is not writable.")


def conduct_survey_battery(
    model: Union[LLM, AsyncOpenAI],
    llm_prompts: Union[LLMPrompt, List[LLMPrompt]],
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Dict[str, Any] = {},
    item_separator: str = "\n",
    **generation_kwargs: Any,
) -> List[InferenceResult]:
    """
    Conducts the entire survey in one single LLM prompt (battery presentation).

    Args:
        model: vllm.LLM instance or AsyncOpenAI client.
        llm_prompts: Single LLMPrompt or list of LLMPrompt objects to conduct as a survey.
        client_model_name: Name of model when using OpenAI client.
        api_concurrency: Number of concurrent API requests.
        print_conversation: If True, prints all conversations to stdout.
        print_progress: If True, shows a tqdm progress bar.
        n_save_step: Save intermediate results every n steps.
        intermediate_save_file: Path to save intermediate results.
        seed: Random seed for reproducibility.
        chat_template: Optionally pass a specific chat template string.
        chat_template_kwargs: Arguments to pass to the chat template (e.g., to disable reasoning).
        item_seperator: Which String should separate the items, defaults to "\n".
        **generation_kwargs: Additional generation parameters that will be given to vllm.chat() or client.chat.completions.create().

    Returns:
        List[InferenceResult]: A list of results containing the survey data and LLM responses for each provided prompt.
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    if isinstance(llm_prompts, LLMPrompt):
        llm_prompts = [llm_prompts]
    # inference_options: List[InferenceOptions] = []

    # We always conduct the survey in one prompt
    max_survey_length: int = 1

    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(llm_prompts)):
        question_llm_response_pairs.append({})

    survey_results: List[InferenceResult] = []

    for i in (
        tqdm(range(max_survey_length), desc="Processing questionnaires")
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            questionnaire for questionnaire in llm_prompts if len(questionnaire._questions) > i
        ]

        system_messages, prompts = zip(
            *[
                questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.BATTERY, i, item_separator=item_separator)
                for questionnaire in current_batch
            ]
        )
        # questions = [interview.generate_question_prompt(interview_question=interview._questions[i]) for interview in current_batch]
        response_generation_methods: List[ResponseGenerationMethod] = []
        for questionnaire in current_batch:
            if questionnaire._questions[i].answer_options:
                response_generation_method = questionnaire._questions[
                    i
                ].answer_options.response_generation_method
                if isinstance(response_generation_method, JSONResponseGenerationMethod):
                    response_generation_method = response_generation_method.create_new_rgm_with_multiple_questions(
                        questions=questionnaire._questions
                    )
                response_generation_methods.append(response_generation_method)

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
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            **generation_kwargs,
        )

        # avoid errors when zipping
        if logprobs is None:
            logprobs = [None] * len(current_batch)

        for survey_id, prompt, answer, logprob_answer, reasoning in zip(
            range(len(current_batch)), prompts, output, logprobs, reasoning_output
        ):
            question_llm_response_pairs[survey_id].update(
                {
                    -1: QuestionLLMResponseTuple(
                        prompt, answer, logprob_answer, reasoning
                    )
                }
            )

        _intermediate_saves(
            llm_prompts,
            n_save_step,
            intermediate_save_file,
            question_llm_response_pairs,
            i,
        )

    for i, survey in enumerate(llm_prompts):
        survey_results.append(InferenceResult(survey, question_llm_response_pairs[i]))

    return survey_results


def conduct_survey_sequential(
    model: Union[LLM, AsyncOpenAI],
    llm_prompts: Union[LLMPrompt, List[LLMPrompt]],
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    seed: int = 42,
    **generation_kwargs: Any,
) -> List[InferenceResult]:
    """
    Conducts the survey in multiple chat calls, where all questions and answers are kept in context (sequential presentation).

    Args:
        model: vllm.LLM instance or AsyncOpenAI client.
        llm_prompts: Single LLMPrompt or list of LLMPrompt objects to conduct as a survey.
        client_model_name: Name of model when using OpenAI client.
        api_concurrency: Number of concurrent API requests.
        print_conversation: If True, prints all conversations to stdout.
        print_progress: If True, shows a tqdm progress bar.
        n_save_step: Save intermediate results every n steps.
        intermediate_save_file: Path to save intermediate results.
        seed: Random seed for reproducibility.
        chat_template: Optionally pass a specific chat template string.
        chat_template_kwargs: Arguments to pass to the chat template (e.g., to disable reasoning).
        item_seperator: Which String should separate the items, defaults to "\n".
        **generation_kwargs: Additional generation parameters that will be given to vllm.chat() or client.chat.completions.create().

    Returns:
        List[InferenceResult]: A list of results containing the survey data and LLM responses for each provided prompt.
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)
    if isinstance(llm_prompts, LLMPrompt):
        llm_prompts = [llm_prompts]

    max_survey_length: int = max(len(questionnaire._questions) for questionnaire in llm_prompts)

    question_llm_response: List[Dict[int, QuestionLLMResponseTuple]] = []

    for i in range(len(llm_prompts)):
        question_llm_response.append({})

    survey_results: List[InferenceResult] = []

    all_prompts: List[List[str]] = []
    assistant_messages: List[List[str]] = []

    for i in range(len(llm_prompts)):
        assistant_messages.append([])
        all_prompts.append([])

    for i in (
        tqdm(range(max_survey_length), desc="Processing questionnaires")
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            questionnaire for questionnaire in llm_prompts if len(questionnaire._questions) > i
        ]

        first_question: bool = i == 0

        if first_question:
            system_messages, prompts = zip(
                *[
                    questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.SEQUENTIAL, i)
                    for questionnaire in current_batch
                ]
            )
            questions = [
                questionnaire.generate_question_prompt(
                    questionnaire_items=questionnaire._questions[i]
                )
                for questionnaire in current_batch
            ]
        else:
            system_messages, _ = zip(
                *[
                    questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.SEQUENTIAL, i)
                    for questionnaire in current_batch
                ]
            )
            prompts = [
                questionnaire.generate_question_prompt(
                    questionnaire_items=questionnaire._questions[i]
                )
                for questionnaire in current_batch
            ]
            questions = prompts

        response_generation_methods = [
            (
                questionnaire._questions[i].answer_options.response_generation_method
                if questionnaire._questions[i].answer_options
                else None
            )
            for questionnaire in current_batch
        ]

        for c in range(len(current_batch)):
            all_prompts[c].append(prompts[c])

        current_assistant_messages: List[str] = []

        missing_indeces = []

        for index, surv in enumerate(current_batch):
            prefilled_answer = surv._questions[i].prefilled_response
            if prefilled_answer:
                current_assistant_messages.append(prefilled_answer)
                missing_indeces.append(index)

        current_batch = [
            item for a, item in enumerate(current_batch) if a not in missing_indeces
        ]

        if len(current_batch) == 0:
            for c in range(len(current_batch)):
                assistant_messages[c].append(current_assistant_messages[c])
            for (
                survey_id,
                question,
                llm_response,
                logprob_answer,
                reasoning,
                item,
            ) in zip(
                range(len(current_batch)),
                questions,
                current_assistant_messages,
                logprobs,
                reasoning_output,
                current_batch,
            ):
                question_llm_response[survey_id].update(
                    {
                        item._questions[i].item_id: QuestionLLMResponseTuple(
                            question, llm_response, logprob_answer, reasoning
                        )
                    }
                )
            continue
            # TODO: add support for automatic system prompt for other answer production methods

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

        # avoid errors when zipping
        if logprobs is None or len(logprobs) == 0:
            logprobs = [None] * len(current_batch)

        for num, index in enumerate(missing_indeces):
            output.insert(index, current_assistant_messages[num])
        for survey_id, question, llm_response, logprob_answer, reasoning, item in zip(
            range(len(current_batch)),
            questions,
            output,
            logprobs,
            reasoning_output,
            current_batch,
        ):
            question_llm_response[survey_id].update(
                {
                    item._questions[i].item_id: QuestionLLMResponseTuple(
                        question, llm_response, logprob_answer, reasoning
                    )
                }
            )

        for o in range(len(output)):
            assistant_messages[o].append(output[o])
        # assistant_messages.append(output)

        _intermediate_saves(
            llm_prompts, n_save_step, intermediate_save_file, question_llm_response, i
        )

    for i, survey in enumerate(llm_prompts):
        survey_results.append(InferenceResult(survey, question_llm_response[i]))

    return survey_results


class SurveyCreator:
    @classmethod
    def from_path(
        self, survey_path: str, questionnaire_path: str
    ) -> List[LLMPrompt]:
        """
        Generates LLMQuestionnaire objects from a CSV file path.

        Args:
            survey_path: The path to the CSV file.

        Returns:
            A list of LLMQuestionnaire objects.
        """
        df = pd.read_csv(survey_path)
        df_questionnaire = pd.read_csv(questionnaire_path)
        return self._from_dataframe(df, df_questionnaire)

    @classmethod
    def from_dataframe(
        self, survey_dataframe: pd.DataFrame, questionnaire_dataframe: pd.DataFrame
    ) -> List[LLMPrompt]:
        """
        Generates LLMQuestionnaire objects from a pandas DataFrame.

        Args:
            survey_dataframe: A DataFrame containing the survey data.

        Returns:
            A list of LLMQuestionnaire objects.
        """
        return self._from_dataframe(survey_dataframe, questionnaire_dataframe)

    @classmethod
    def _create_questionnaire(self, row: pd.Series, df_questionnaire) -> LLMPrompt:
        """
        Internal helper method to process the DataFrame.
        """
        return LLMPrompt(
            questionnaire_source=df_questionnaire,
            questionnaire_name=row[constants.QUESTIONNAIRE_NAME],
            system_prompt=row[constants.SYSTEM_PROMPT_FIELD],
            prompt=row[constants.QUESTIONNAIRE_INSTRUCTION_FIELD],
        )

    @classmethod
    def _from_dataframe(
        self, df: pd.DataFrame, df_questionnaire: pd.DataFrame
    ) -> List[LLMPrompt]:
        """
        Internal helper method to process the DataFrame.
        """
        questionnaires = df.apply(
            lambda row: self._create_questionnaire(row, df_questionnaire), axis=1
        )
        return questionnaires.to_list()


if __name__ == "__main__":
    pass
