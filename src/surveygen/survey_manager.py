"""
Module for managing and conducting surveys using LLM models.

This module provides functions to conduct surveys in different ways:
- Question by question
- Whole survey in one prompt
- In-context learning

Usage example:
-------------
```python
from surveygen import LLMInterview, conduct_survey_question_by_question
from surveygen.parser.llm_answer_parser import raw_responses
from vllm import LLM

# Initialize model and interview
model = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
interview = LLMInterview(interview_path="questions.csv")
interview.prepare_interview(question_stem="How do you feel towards {QUESTION_CONTENT_PLACEHOLDER}?")

# Conduct survey
results = conduct_survey_question_by_question(
    model=model,
    interviews=interview,
    print_progress=True
)

# Access results
for result in results:
    raw_responses = raw_responses(survey_answers)
```
"""

from typing import (
    List,
    Dict,
    Optional,
    Union,
    Any,
    overload,
    Literal,
)
from string import ascii_lowercase, ascii_uppercase

from .utilities.survey_objects import (
    AnswerOptions,
    QuestionLLMResponseTuple,
    AnswerTexts,
)
from .utilities import prompt_templates
from .utilities import constants
from .utilities import utils

from .parser.llm_answer_parser import raw_responses

from .inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from .inference.response_generation import (
    ResponseGenerationMethod,
    JSONResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
)

from .llm_interview import LLMInterview, InterviewType

from .utilities.survey_objects import AnswerOptions, InferenceOptions, InterviewResult

from vllm import LLM

from openai import AsyncOpenAI

from pathlib import Path
import os

import pandas as pd

import random

from tqdm.auto import tqdm


class SurveyOptionGenerator:
    """
    A class responsible for preparing optional answers for the LLM.
    """

    LIKERT_5: List[str] = [
        "disagree strongly",
        "disagree a little",
        "neither agree nor disagree",
        "agree a little",
        "agree strongly",
    ]
    LIKERT_NO_MIDDLE: List[str] = [
        "disagree strongly",
        "disagree a little",
        "agree a little",
        "agree strongly",
    ]

    LIKERT_IMPORTANCE_FROM_TO: List[str] = ["Not at all important", "Very Important"]
    LIKERT_JUSTIFIABLE_FROM_TO: List[str] = ["Never justifiable", "Always justifiable"]
    _IDX_TYPES = Literal["char_lower", "char_upper", "integer", "no_index"]

    @staticmethod
    def generate_likert_options(
        n: int,
        answer_texts: Optional[List[str]],
        only_from_to_scale: bool = False,
        random_order: bool = False,
        reversed_order: bool = False,
        even_order: bool = False,
        start_idx: int = 1,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        index_answer_separator: str = ": ",
        options_separator: str = ", ",
        idx_type: _IDX_TYPES = "integer",
        response_generation_method: Optional[ResponseGenerationMethod] = None,
    ) -> AnswerOptions:
        """Generates a set of options and a prompt for a Likert-style scale.

        This function creates a numeric or alphabetic scale of a specified size (n),
        optionally attaching textual labels to the scale. It provides
        extensive control over ordering, formatting, and the final prompt string.

        :param n: The number of points on the scale (e.g., 5 for a 1-5 scale).
        :type n: int
        :param answer_texts: A list of strings to use as labels for points on the scale.
                             For example, ["Agree", "Disagree", "Neither"] for a 3-point scale.
        :type answer_texts: Optional[List[str]]
        :param only_from_to_scale: If True, forces a 'range' style prompt (e.g., "from 1 to 5")
                                using `scale_prompt_template`.
        :type only_from_to_scale: bool
        :param random_order: If True, shuffles the final list of generated options.
                            Mutually exclusive with `reversed_order`.
        :type random_order: bool
        :param reversed_order: If True, reverses the natural order of the scale (e.g., 5, 4, 3, 2, 1).
                            Mutually exclusive with `random_order`.
        :type reversed_order: bool
        :param even_order: Removes the middle answer option of the scale, if the scale is odd.
        :type even_order: bool
        :param start_idx: The starting number for an character and integer based scales. Defaults to 1. If e.g. 2 is given then (0 -> 1) and (A -> B).
        :type start_idx: int
        :param list_prompt_template: The format string for a list-style prompt. Must contain `{options}`.
        :type list_prompt_template: str
        :param scale_prompt_template: The format string for a range-style prompt. Must contain `{start}`
                                    and `{end}`.
        :type scale_prompt_template: str
        :param idx_type: The type of index for the scale: "char_lower", "char_upper", "integer", or "no_index".
        :type idx_type: _IDX_TYPES

        :raises ValueError: If `n` is less than 2, if `random_order` and `reversed_order` are both True,
                            or if `len(anchor_labels)` > `n`.

        :return: An object containing `options` (the final list of option strings) and `prompt`
                (the formatted prompt string).
        :rtype: AnswerOptions"""

        if only_from_to_scale:
            if len(answer_texts) != 2:
                raise ValueError(
                    f"From-To scales require exactly 2 descriptions, but answer_texts was set to '{answer_texts}'."
                )
            if idx_type != "integer":
                raise ValueError(
                    f"From-To scales require an integer scale index, but idx_type was set to '{idx_type}'."
                )
        else:
            if answer_texts:
                if len(answer_texts) != n:
                    raise ValueError(
                        f"answer_texts and n need to be the same length, but answer_texts has length {len(answer_texts)} and n was given as {n}."
                    )
        if even_order:
            if n % 2 != 0:
                raise ValueError(
                    "If you want to turn a scale even, it should be odd before."
                )
            middle_index = n // 2
            answer_texts = (
                answer_texts[:middle_index] + answer_texts[middle_index + 1 :]
            )
            n = n - 1
        if random_order:
            if len(answer_texts) < 2:
                raise ValueError(
                    "There must be at least two answer options to reorder randomly."
                )
            random.shuffle(
                answer_texts
            )  # no assignment needed because shuffles already inplace
        if reversed_order:
            if len(answer_texts) < 2:
                raise ValueError(
                    "There must be at least two answer options to reorder in reverse."
                )
            answer_options = answer_options[::-1]

        answer_option_indices = []
        if idx_type == "no_index":
            # no index, just the answer options directly
            answer_option_indices = None
        elif idx_type == "integer":
            for i in range(n):
                answer_code = i + start_idx
                answer_option_indices.append(str(answer_code))
        else:
            # TODO @Jens add these to constants.py
            if idx_type == "char_lower":
                for i in range(n):
                    answer_option_indices.append(ascii_lowercase[(i + start_idx) % 26])
            elif idx_type == "char_upper":
                for i in range(n):
                    answer_option_indices.append(ascii_uppercase[(i + start_idx) % 26])

        answer_texts_object = AnswerTexts(
            answer_texts=answer_texts,
            indices=answer_option_indices,
            index_answer_seperator=index_answer_separator,
            option_seperators=options_separator,
            only_scale=only_from_to_scale
        )

        interview_option = AnswerOptions(
            answer_texts=answer_texts_object,
            from_to_scale=only_from_to_scale,
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            response_generation_method=response_generation_method,
        )

        return interview_option

    # #TODO: It seems to me like this method and the one above could be merged? (Georg)
    # @staticmethod
    # def generate_generic_options(
    #     answer_texts: Dict,
    #     only_from_to_scale: bool = False,
    #     random_order: bool = False,
    #     reversed_order: bool = False,
    #     even_order: bool = False,
    #     idx_type: Optional[_IDX_TYPES] = None, # uses the answer_texts.keys() as an index by default
    #     list_prompt_template: Optional[str] = prompt_templates.LIST_OPTIONS_DEFAULT,
    #     scale_prompt_template: Optional[str] = prompt_templates.SCALE_OPTIONS_DEFAULT,
    #     options_separator: str = ", ",
    # ):

    #     n = len(answer_texts.values())
    #     answer_codes = answer_texts.keys()
    #     answer_texts = answer_texts.values()
    #     # answer_options = descriptions

    #     if idx_type == 'char_lower':
    #         if all(isinstance(item, int) for item in answer_codes):
    #             new_codes = []
    #             for i in answer_codes:
    #                 code = ascii_lowercase[i - 1]
    #                 new_codes.append(code)
    #             answer_codes = new_codes
    #         else:
    #             answer_codes = [s.lower() for s in answer_codes]
    #     elif idx_type == 'char_upper':
    #         if all(isinstance(item, int) for item in answer_codes):
    #             new_codes = []
    #             for i in answer_codes:
    #                 code = ascii_uppercase[i - 1]
    #                 new_codes.append(code)
    #             answer_codes = new_codes
    #         else:
    #             answer_codes = [s.upper() for s in answer_codes]
    #     elif idx_type == 'integer':
    #         answer_codes = range(1, len(answer_codes) + 1)

    #     answer_options = dict(zip(answer_codes, answer_texts))

    #     if only_from_to_scale:
    #         assert all(
    #             isinstance(item, int) for item in answer_codes
    #         ), "To use from-to scale you must have integer answer codes."

    #     if random_order:
    #         assert (
    #             n >= 2
    #         ), "There must be at least two answer options to reorder randomly."
    #         temp = list(answer_texts)
    #         random.shuffle(temp)
    #         # reassigning to keys
    #         answer_options = dict(zip(answer_codes, temp))
    #     if reversed_order:
    #         assert (
    #             n >= 2
    #         ), "There must be at least two answer options to reverse options."
    #         reversed_values = list(answer_texts)[::-1]
    #         answer_options = dict(zip(answer_codes, reversed_values))
    #     if even_order:
    #         assert n % 2 != 0, "There must be a odd number of options!"
    #         middle_index = n // 2
    #         # Get the key of the item to be removed
    #         key_to_remove = list(answer_codes)[middle_index]
    #         # Create a new dictionary, excluding the item with key_to_remove
    #         # This uses a dictionary comprehension.
    #         answer_options = {
    #             key: value
    #             for key, value in answer_options.items()
    #             if key != key_to_remove
    #         }
    #         if all(isinstance(element, int) for element in list(answer_codes)):
    #             first_part = list(answer_codes)[:middle_index]
    #             last_part = list(answer_codes)[middle_index + 1 :]
    #             last_part = [x - 1 for x in last_part]
    #             answer_options = dict(
    #                 zip(first_part + last_part, answer_options.values())
    #             )
    #         elif set(list(answer_codes)).issubset(list(ascii_lowercase)):
    #             first_part = list(answer_codes)[:middle_index]
    #             # print("First part:", first_part)
    #             last_part = list(answer_codes)[middle_index + 1 :]
    #             # print("Last part:", last_part)
    #             last_parts = []
    #             for i in range(middle_index + 1, len(list(answer_codes))):
    #                 part = list(ascii_lowercase)[i - 1]
    #                 last_parts.append(part)
    #                 # print("Last parts:", last_parts)
    #             answer_options = dict(
    #                 zip(first_part + last_parts, answer_options.values())
    #             )
    #         elif set(list(answer_codes)).issubset(list(ascii_uppercase)):
    #             first_part = list(answer_codes)[:middle_index]
    #             # print("First part:", first_part)
    #             last_part = list(answer_codes)[middle_index + 1 :]
    #             # print("Last part:", last_part)
    #             last_parts = []
    #             for i in range(middle_index + 1, len(list(answer_codes))):
    #                 part = list(ascii_uppercase)[i - 1]
    #                 last_parts.append(part)
    #                 # print("Last parts:", last_parts)
    #             answer_options = dict(
    #                 zip(first_part + last_parts, answer_options.values())
    #             )

    #     if idx_type == 'no_index':
    #         answer_option_strings = list(answer_options.values())
    #         answer_option_index = None
    #     else:
    #         answer_option_strings = [f"{key}: {val}" for key, val in answer_options.items()]
    #         answer_option_index = list(answer_options.keys())
    #     #print(answer_options)

    #     interview_option = AnswerOptions(
    #         answer_text = answer_option_strings,
    #         index = answer_option_index,
    #         from_to_scale = only_from_to_scale,
    #         list_prompt_template = list_prompt_template,
    #         scale_prompt_template = scale_prompt_template,
    #         options_seperator = options_separator,
    #     )

    #     return interview_option


def conduct_survey_question_by_question(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    seed: int = 42,
    chat_template: Optional[str] = str,
    chat_template_kwargs: Dict[str, Any] = {},
    **generation_kwargs: Any,
) -> List[InterviewResult]:
    """
    Conducts a survey by asking questions one at a time.

    Args:
        model: LLM instance or AsyncOpenAI client.
        interviews: Single interview or list of interviews to conduct as a survey.
        answer_production_method: Options for structured output format.
        client_model_name: Name of model when using OpenAI client.
        api_concurrency: Number of concurrent API requests.
        print_conversation: If True, prints all conversations.
        print_progress: If True, shows progress bar.
        n_save_step: Save intermediate results every n steps.
        intermediate_save_file: Path to save intermediate results.
        seed: Random seed for reproducibility.
        chat_template: Optionally pass a specific chat template
        chat_template_kwargs: Arguments to pass to the chat template, e.g., to disable reasoning
        **generation_kwargs: Additional generation parameters that will be given to vllm.chat() or  client.chat.completions.create().

    Returns:
        List[InterviewResult]: Results for each interview.
    """

    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    if isinstance(interviews, LLMInterview):
        interviews = [interviews]

    max_survey_length: int = max(len(interview._questions) for interview in interviews)
    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    for i in range(len(interviews)):
        #inference_option = interviews[i]._generate_inference_options()
        #inference_options.append(inference_opti
        question_llm_response_pairs.append({})

    survey_results: List[InterviewResult] = []

    for i in (
        tqdm(range(max_survey_length), desc="Processing interviews")
        if print_progress
        else range(max_survey_length)
    ):
        current_batch: List[LLMInterview] = [
            interview
            for interview in interviews
            if len(interview._questions) > i
        ]

        system_messages, prompts = zip(*[
            interview.get_prompt_for_interview_type(InterviewType.QUESTION, i)
            for interview in current_batch
        ])

        questions = [interview.generate_question_prompt(interview_question=interview._questions[i]) for interview in current_batch]
        response_generation_methods = [interview._questions[i].answer_options.response_generation_method for interview in current_batch]

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
            current_batch
        ):
            question_llm_response_pairs[survey_id].update(
                {
                    item._questions[i].item_id: QuestionLLMResponseTuple(
                        question, answer, logprob_answer, reasoning
                    )
                }
            )

        # TODO: check that this works with logprobs
        _intermediate_saves(
            interviews,
            n_save_step,
            intermediate_save_file,
            question_llm_response_pairs,
            i,
        )

    for i, survey in enumerate(interviews):
        survey_results.append(InterviewResult(survey, question_llm_response_pairs[i]))

    return survey_results


def _intermediate_saves(
    interviews: List[LLMInterview],
    n_save_step: int,
    intermediate_save_file: str,
    question_llm_response_pairs: QuestionLLMResponseTuple,
    i: int,
):
    """
    Internal helper to save intermediate survey results.

    Args:
        interviews: List of interviews being conducted.
        n_save_step: Save frequency in steps.
        intermediate_save_file: Path to save file.
        question_llm_response_pairs: Current responses.
        i: Current step number.
    """
    if n_save_step:
        if i % n_save_step == 0:
            intermediate_survey_results: List[InterviewResult] = []
            for j, interview in enumerate(interviews):
                intermediate_survey_results.append(
                    InterviewResult(interview, question_llm_response_pairs[j])
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


def conduct_whole_survey_one_prompt(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Dict[str, Any] = {},
    item_separator:str = "\n",
    **generation_kwargs: Any,
) -> List[InterviewResult]:
    """
    Conducts the entire survey in one single LLM prompt.

    Args:
        model: LLM instance or AsyncOpenAI client.
        interviews: Single interview or list of interviews to conduct.
        answer_production_method: Options for structured output format.
        client_model_name: Name of model when using OpenAI client.
        api_concurrency: Number of concurrent API requests.
        n_save_step: Save intermediate results every n steps.
        intermediate_save_file: Path to save intermediate results.
        print_conversation: If True, prints the conversation.
        print_progress: If True, shows progress bar.
        seed: Random seed for reproducibility.
        chat_template: Optionally pass a specific chat template
        chat_template_kwargs: Arguments to pass to the chat template, e.g., to disable reasoning
        **generation_kwargs: Additional generation parameters that will be given to vllm.chat() or  client.chat.completions.create().

    Returns:
        List[InterviewResult]: Results for each interview.
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    if isinstance(interviews, LLMInterview):
        interviews = [interviews]
    #inference_options: List[InferenceOptions] = []

    # We always conduct the survey in one prompt
    max_survey_length: int = 1

    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        question_llm_response_pairs.append({})

    survey_results: List[InterviewResult] = []

    for i in (
        tqdm(range(max_survey_length), desc="Processing interviews")
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            interview
            for interview in interviews
            if len(interview._questions) > i
        ]

        system_messages, prompts = zip(*[
            interview.get_prompt_for_interview_type(InterviewType.ONE_PROMPT, i)
            for interview in current_batch
        ])
        #questions = [interview.generate_question_prompt(interview_question=interview._questions[i]) for interview in current_batch]
        response_generation_methods: List[ResponseGenerationMethod] = []
        for interview in current_batch:
            response_generation_method = interview._questions[i].answer_options.response_generation_method
            if isinstance(response_generation_method, JSONResponseGenerationMethod):    
                response_generation_method = response_generation_method.create_new_rgm_with_multiple_questions(questions=interview._questions)
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
            interviews,
            n_save_step,
            intermediate_save_file,
            question_llm_response_pairs,
            i,
        )

    for i, survey in enumerate(interviews):
        survey_results.append(InterviewResult(survey, question_llm_response_pairs[i]))

    return survey_results


def conduct_survey_in_context(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    seed: int = 42,
    **generation_kwargs: Any,
) -> List[InterviewResult]:
    """
    Conducts surveys using in-context learning approach.

    Args:
        model: LLM instance or AsyncOpenAI client.
        interviews: Single interview or list of interviews to conduct.
        answer_production_method: Options for structured output format.
        client_model_name: Name of model when using OpenAI client.
        api_concurrency: Number of concurrent API requests.
        print_conversation: If True, prints the conversation.
        print_progress: If True, shows progress bar.
        n_save_step: Save intermediate results every n steps.
        intermediate_save_file: Path to save intermediate results.
        seed: Random seed for reproducibility.
        **generation_kwargs: Additional generation parameters that will be given to vllm.chat() or  client.chat.completions.create().

    Returns:
        List[InterviewResult]: Results for each interview.
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)
    if isinstance(interviews, LLMInterview):
        interviews = [interviews]
    
    max_survey_length: int = max(len(interview._questions) for interview in interviews)

    question_llm_response: List[Dict[int, QuestionLLMResponseTuple]] = []

    for i in range(len(interviews)):
        question_llm_response.append({})

    survey_results: List[InterviewResult] = []

    all_prompts: List[List[str]] = []
    assistant_messages: List[List[str]] = []

    for i in range(len(interviews)):
        assistant_messages.append([])
        all_prompts.append([])

    for i in (
        tqdm(range(max_survey_length), desc="Processing interviews")
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            interview
            for interview in interviews
            if len(interview._questions) > i
        ]

        first_question: bool = i == 0

        if first_question:
            system_messages, prompts = zip(*[
            interview.get_prompt_for_interview_type(InterviewType.CONTEXT, i)
            for interview in current_batch
        ])
            questions = [interview.generate_question_prompt(interview_question=interview._questions[i]) for interview in current_batch]
        else: 
            system_messages, _ = zip(*[
                interview.get_prompt_for_interview_type(InterviewType.CONTEXT, i)
                for interview in current_batch
            ])
            prompts = [interview.generate_question_prompt(interview_question=interview._questions[i]) for interview in current_batch]
            questions = prompts

        response_generation_methods = [interview._questions[i].answer_options.response_generation_method for interview in current_batch]


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
            for survey_id, question, llm_response, logprob_answer, reasoning, item in zip(
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
            range(len(current_batch)), questions, output, logprobs, reasoning_output, current_batch
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
            interviews, n_save_step, intermediate_save_file, question_llm_response, i
        )

    for i, survey in enumerate(interviews):
        survey_results.append(InterviewResult(survey, question_llm_response[i]))

    return survey_results


class SurveyCreator:
    @classmethod
    def from_path(
        self, survey_path: str, questionnaire_path: str
    ) -> List[LLMInterview]:
        """
        Generates LLMInterview objects from a CSV file path.

        Args:
            survey_path: The path to the CSV file.

        Returns:
            A list of LLMInterview objects.
        """
        df = pd.read_csv(survey_path)
        df_questionnaire = pd.read_csv(questionnaire_path)
        return self._from_dataframe(df, df_questionnaire)

    @classmethod
    def from_dataframe(
        self, survey_dataframe: pd.DataFrame, questionnaire_dataframe: pd.DataFrame
    ) -> List[LLMInterview]:
        """
        Generates LLMInterview objects from a pandas DataFrame.

        Args:
            survey_dataframe: A DataFrame containing the survey data.

        Returns:
            A list of LLMInterview objects.
        """
        return self._from_dataframe(survey_dataframe, questionnaire_dataframe)

    @classmethod
    def _create_interview(self, row: pd.Series, df_questionnaire) -> LLMInterview:
        """
        Internal helper method to process the DataFrame.
        """
        return LLMInterview(
            interview_source=df_questionnaire,
            interview_name=row[constants.INTERVIEW_NAME],
            system_prompt=row[constants.SYSTEM_PROMPT_FIELD],
            prompt=row[constants.INTERVIEW_INSTRUCTION_FIELD],
        )

    @classmethod
    def _from_dataframe(
        self, df: pd.DataFrame, df_questionnaire: pd.DataFrame
    ) -> List[LLMInterview]:
        """
        Internal helper method to process the DataFrame.
        """
        interviews = df.apply(
            lambda row: self._create_interview(row, df_questionnaire), axis=1
        )
        return interviews.to_list()


if __name__ == "__main__":
    pass
