from typing import (
    List,
    Dict,
    Optional,
    Union,
    Any,
    overload,
    Self,
    Literal,
    Final,
)
from enum import Enum

from dataclasses import dataclass, replace
from string import ascii_lowercase, ascii_uppercase

from .utilities.prompt_creation import PromptCreation
from .utilities.survey_objects import (
    AnswerOptions,
    InterviewItem,
    QuestionLLMResponseTuple,
)
from .utilities import prompt_templates
from .utilities import constants
from .utilities import utils

from .parser.llm_answer_parser import raw_responses

from .inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from .inference.dynamic_pydantic import generate_pydantic_model

from .llm_interview import LLMInterview

from .utilities.survey_objects import AnswerOptions, InferenceOptions, InterviewResult

from vllm import LLM
from vllm.sampling_params import GuidedDecodingParams

from openai import AsyncOpenAI

from pathlib import Path
import os

import pandas as pd

import random

import copy

import tqdm


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
    _IDX_TYPES = Literal["char_low", "char_upper", "integer"]

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
        options_separator: str = ", ",
        idx_type: _IDX_TYPES = "integer",
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
        :param start_idx: The starting number for an integer-based scale (e.g., 1 or 0).
        :type start_idx: int
        :param list_prompt_template: The format string for a list-style prompt. Must contain `{options}`.
        :type list_prompt_template: str
        :param scale_prompt_template: The format string for a range-style prompt. Must contain `{start}`
                                    and `{end}`.
        :type scale_prompt_template: str
        :param idx_type: The type of index for the scale: "char_low", "char_upper" or "integer".
        :type idx_type: _IDX_TYPES

        :raises ValueError: If `n` is less than 2, if `random_order` and `reversed_order` are both True,
                            or if `len(anchor_labels)` > `n`.

        :return: An object containing `options` (the final list of option strings) and `prompt`
                (the formatted prompt string).
        :rtype: AnswerOptions"""

        # @TODO @Jens Instead of assertions we should probably raise Value errors
        if only_from_to_scale:
            assert (
                len(answer_texts) == 2
            ), "If from to scale, provide exactly two descriptions"
            assert (
                idx_type == "integer"
            ), "Index type must be integer, not lower/uppercase characters."
        else:
            if answer_texts:
                assert (
                    len(answer_texts) == n
                ), "Description list must be the same length as options"
        if even_order:
            assert n % 2 != 0, "There must be a odd number of options!"
            middle_index = n // 2
            answer_texts = (
                answer_texts[:middle_index] + answer_texts[middle_index + 1 :]
            )
            n = n - 1
        if random_order:
            assert (
                len(answer_texts) >= 2
            ), "There must be at least two answer options to reorder randomly."
            random.shuffle(
                answer_texts
            )  # no assignment needed because shuffles already inplace
        if reversed_order:
            assert (
                len(answer_option) >= 2
            ), "There must be at least two answer options to reverse options."
            answer_options = answer_options[::-1]

        answer_options = []
        if idx_type == "integer":
            for i in range(n):
                answer_code = i + start_idx
                answer_option = f"{answer_code}"
                if only_from_to_scale:
                    if i == 0:
                        answer_option = f"{answer_code}: {answer_texts[0]}"
                    elif i == (n - 1):
                        answer_option = f"{answer_code}: {answer_texts[1]}"
                elif answer_texts:
                    answer_option = f"{answer_code}: {answer_texts[i]}"
                answer_options.append(answer_option)
        else:
            # TODO @Jens add these to constants.py
            if idx_type == "char_low":
                for i in range(n):
                    answer_option = f"{ascii_lowercase[i]}: {answer_texts[i]}"
                    answer_options.append(answer_option)
            elif idx_type == "char_upper":
                for i in range(n):
                    answer_option = f"{ascii_uppercase[i]}: {answer_texts[i]}"
                    answer_options.append(answer_option)

        interview_option = AnswerOptions(
            answer_options,
            from_to_scale=only_from_to_scale,
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            options_seperator=options_separator,
        )

        return interview_option

    @staticmethod
    def generate_generic_options(
        answer_texts: Dict,
        only_from_to_scale: bool = False,
        random_order: bool = False,
        reversed_order: bool = False,
        even_order: bool = False,
        to_lowercase: bool = False,
        to_uppercase: bool = False,
        to_integer: bool = False,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        options_separator: str = ", ",
    ):

        n = len(answer_texts.values())
        answer_codes = answer_texts.keys()
        answer_texts = answer_texts.values()
        # answer_options = descriptions

        if to_lowercase:
            if all(isinstance(item, int) for item in answer_codes):
                new_codes = []
                for i in answer_codes:
                    code = ascii_lowercase[i - 1]
                    new_codes.append(code)
                answer_codes = new_codes
            else:
                answer_codes = [s.lower() for s in answer_codes]
        if to_uppercase:
            if all(isinstance(item, int) for item in answer_codes):
                new_codes = []
                for i in answer_codes:
                    code = ascii_uppercase[i - 1]
                    new_codes.append(code)
                answer_codes = new_codes
            else:
                answer_codes = [s.upper() for s in answer_codes]
        if to_integer:
            answer_codes = range(1, len(answer_codes) + 1)

        answer_options = dict(zip(answer_codes, answer_texts))

        if only_from_to_scale:
            assert all(
                isinstance(item, int) for item in answer_codes
            ), "To use from-to scale you must have integer answer codes."

        if random_order:
            assert (
                n >= 2
            ), "There must be at least two answer options to reorder randomly."
            temp = list(answer_texts)
            random.shuffle(temp)
            # reassigning to keys
            answer_options = dict(zip(answer_codes, temp))
        if reversed_order:
            assert (
                n >= 2
            ), "There must be at least two answer options to reverse options."
            reversed_values = list(answer_texts)[::-1]
            answer_options = dict(zip(answer_codes, reversed_values))
        if even_order:
            assert n % 2 != 0, "There must be a odd number of options!"
            middle_index = n // 2
            # Get the key of the item to be removed
            key_to_remove = list(answer_codes)[middle_index]
            # Create a new dictionary, excluding the item with key_to_remove
            # This uses a dictionary comprehension.
            answer_options = {
                key: value
                for key, value in answer_options.items()
                if key != key_to_remove
            }
            if all(isinstance(element, int) for element in list(answer_codes)):
                first_part = list(answer_codes)[:middle_index]
                last_part = list(answer_codes)[middle_index + 1 :]
                last_part = [x - 1 for x in last_part]
                answer_options = dict(
                    zip(first_part + last_part, answer_options.values())
                )
            elif set(list(answer_codes)).issubset(list(ascii_lowercase)):
                first_part = list(answer_codes)[:middle_index]
                # print("First part:", first_part)
                last_part = list(answer_codes)[middle_index + 1 :]
                # print("Last part:", last_part)
                last_parts = []
                for i in range(middle_index + 1, len(list(answer_codes))):
                    part = list(ascii_lowercase)[i - 1]
                    last_parts.append(part)
                    # print("Last parts:", last_parts)
                answer_options = dict(
                    zip(first_part + last_parts, answer_options.values())
                )
            elif set(list(answer_codes)).issubset(list(ascii_uppercase)):
                first_part = list(answer_codes)[:middle_index]
                # print("First part:", first_part)
                last_part = list(answer_codes)[middle_index + 1 :]
                # print("Last part:", last_part)
                last_parts = []
                for i in range(middle_index + 1, len(list(answer_codes))):
                    part = list(ascii_uppercase)[i - 1]
                    last_parts.append(part)
                    # print("Last parts:", last_parts)
                answer_options = dict(
                    zip(first_part + last_parts, answer_options.values())
                )

        answer_options = [f"{key}: {val}" for key, val in answer_options.items()]
        print(answer_options)

        interview_option = AnswerOptions(
            answer_options,
            from_to_scale=only_from_to_scale,
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            options_seperator=options_separator,
        )

        return interview_option


def conduct_survey_question_by_question(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    json_structured_output: bool = False,
    json_structure: Optional[List[str]] = constants.DEFAULT_JSON_STRUCTURE,
    constraints: Optional[Dict[str, List[str]]] = constants.DEFAULT_CONSTRAINTS,
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
    Conducts the survey with each question in a new context.

    :param model: LLM instance of vllm.
    :param system_prompt: The system prompt of the model.
    :param task_instruction: The task instructio the model will be prompted with.
    :param json_structured_output: If json_structured output should be used.
    :param json_structure: The structure the final ouput should have.
    :param batch_size: How many inferences should run in parallel.
    :param print_conversation: If True, the whole conversation will be printed.
    :param generation_kwargs: All keywords needed for SamplingParams.
    :return: Generated text by the LLM in double list format
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    if isinstance(interviews, LLMInterview):
        interviews = [interviews]

    inference_options: List[InferenceOptions] = []

    max_survey_length: int = 0

    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        inference_option = interviews[i]._generate_inference_options()
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_llm_response_pairs.append({})

    survey_results: List[InterviewResult] = []

    #TODO allow for different answer option constraints between surveys
    if constraints:
        for json_element in constraints.keys():
            if constraints[json_element] == constants.OPTIONS_ADJUST:
                constraints[json_element] = inference_options[0].answer_options[0].answer_text

    for i in (
        tqdm.tqdm(range(max_survey_length))
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            inference_option
            for inference_option in inference_options
            if len(inference_option.order) > i
        ]

        if json_structured_output:
            system_messages = [
                inference.json_system_prompt(json_options=json_structure)
                for inference in current_batch
            ]

        else:
            system_messages = [inference.system_prompt for inference in current_batch]
        prompts = [
            inference.create_single_question(inference.order[i], task_instruction=True)
            for inference in current_batch
        ]
        questions = [
            inference.create_single_question(inference.order[i], task_instruction=False)
            for inference in current_batch
        ]

        output = batch_generation(
            model=model,
            system_messages=system_messages,
            prompts=prompts,
            guided_decoding_options= "json" if json_structured_output else None,
            json_fields=json_structure,
            constraints=constraints,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )

        for survey_id, question, answer, item in zip(
            range(len(current_batch)), questions, output, current_batch
        ):
            question_llm_response_pairs[survey_id].update(
                {item.order[i]: QuestionLLMResponseTuple(question, answer)}
            )

        _intermediate_saves(interviews, n_save_step, intermediate_save_file, question_llm_response_pairs, i)

    for i, survey in enumerate(interviews):
        survey_results.append(InterviewResult(survey, question_llm_response_pairs[i]))

    return survey_results

def _intermediate_saves(interviews: List[LLMInterview], n_save_step: int, intermediate_save_file: str, question_llm_response_pairs: QuestionLLMResponseTuple, i: int):
    if n_save_step:
        if i % n_save_step == 0:
            intermediate_survey_results: List[InterviewResult] = []
            for j, interview in enumerate(interviews):
                intermediate_survey_results.append(InterviewResult(interview, question_llm_response_pairs[j]))
            parsed_results = raw_responses(intermediate_survey_results)
            utils.create_one_dataframe(parsed_results).to_csv(intermediate_save_file)

def _intermediate_save_path_check(n_save_step:int, intermediate_save_path:str):
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
                raise ValueError(f"Invalid intermediate save path: {intermediate_save_path}. Error: {e}")

        # Optional: Check it's writable
        if not os.access(parent_dir, os.W_OK):
            raise ValueError(f"Save path '{intermediate_save_path}' is not writable.")


def conduct_whole_survey_one_prompt(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    json_structured_output: bool = False,
    json_structure: Optional[List[str]] = constants.DEFAULT_JSON_STRUCTURE,
    constraints: Optional[Dict[str, List[str]]] = constants.DEFAULT_CONSTRAINTS,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    n_save_step: Optional[int] = None,
    intermediate_save_file: Optional[str] = None,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    **generation_kwargs: Any,
) -> List[InterviewResult]:
    """
    Conducts the entire survey in one single LLM prompt.

    :param model: LLM instance of vllm.
    :param system_prompt: The system prompt of the model.
    :param task_instruction: The task instructio the model will be prompted with.
    :param json_structured_output: If json_structured output should be used.
    :param json_structure: The structure the final ouput should have.
    :param batch_size: How many inferences should run in parallel.
    :param print_conversation: If True, the whole conversation will be printed.
    :param generation_kwargs: All keywords needed for SamplingParams.
    :return: Generated text by the LLM in double list format
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)

    if isinstance(interviews, LLMInterview):
        interviews = [interviews]
    inference_options: List[InferenceOptions] = []

    # We always conduct the survey in one prompt
    max_survey_length: int = 1

    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        inference_option = interviews[i]._generate_inference_options()
        inference_options.append(inference_option)

        question_llm_response_pairs.append({})

    survey_results: List[InterviewResult] = []

    for i in (
        tqdm.tqdm(range(max_survey_length))
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            inference_option
            for inference_option in inference_options
            if len(inference_option.order) > i
        ]

        if json_structured_output:
            all_json_structures = []
            all_constraints = []
            for inference_option in current_batch:
                full_json_structure = []
                full_constraints = {}
                for i in range(len(inference_option.answer_options)):
                    for json_element in json_structure:
                        new_element = f"{json_element}{i}"
                        if constraints:
                            constraints_element = constraints.get(json_element)
                            if constraints_element == constants.OPTIONS_ADJUST:
                                full_constraints[new_element] = inference_option.answer_options[i].answer_text
                            elif constraints_element != None:
                                full_constraints[new_element] = constraints_element
                        full_json_structure.append(new_element)
                all_constraints.append(full_constraints)
                all_json_structures.append(full_json_structure)
                    
            system_messages = [
                inference.json_system_prompt(all_json_structures[num])
                for num, inference in enumerate(current_batch)
            ]

        else:
            system_messages = [inference.system_prompt for inference in current_batch]
        prompts = [inference.create_all_questions() for inference in current_batch]

        output = batch_generation(
            model=model,
            system_messages=system_messages,
            prompts=prompts,
            guided_decoding_options= "json" if json_structured_output else None,
            json_fields=all_json_structures,
            constraints=all_constraints if constraints else None,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )

        for survey_id, prompt, answer in zip(
            range(len(current_batch)), prompts, output
        ):
            question_llm_response_pairs[survey_id].update(
                {-1: QuestionLLMResponseTuple(prompt, answer)}
            )
        
        _intermediate_saves(interviews, n_save_step, intermediate_save_file, question_llm_response_pairs, i)

    for i, survey in enumerate(interviews):
        survey_results.append(InterviewResult(survey, question_llm_response_pairs[i]))

    return survey_results


def conduct_survey_in_context(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    json_structured_output: bool = False,
    json_structure: Optional[List[str]] = constants.DEFAULT_JSON_STRUCTURE,
    constraints: Optional[Dict[str, List[str]]] = constants.DEFAULT_CONSTRAINTS,
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
    Conducts the entire survey multiple prompts but within the same context window.

    :param model: LLM instance of vllm.
    :param system_prompt: The system prompt of the model.
    :param task_instruction: The task instructio the model will be prompted with.
    :param json_structured_output: If json_structured output should be used.
    :param json_structure: The structure the final ouput should have.
    :param batch_size: How many inferences should run in parallel.
    :param print_conversation: If True, the whole conversation will be printed.
    :param generation_kwargs: All keywords needed for SamplingParams.
    :return: Generated text by the LLM in double list format
    """
    _intermediate_save_path_check(n_save_step, intermediate_save_file)
    if isinstance(interviews, LLMInterview):
        interviews = [interviews]

    inference_options: List[InferenceOptions] = []

    max_survey_length: int = 0

    question_llm_response: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        inference_option = interviews[i]._generate_inference_options()
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_llm_response.append({})

    survey_results: List[InterviewResult] = []

    all_prompts: List[List[str]] = []
    assistant_messages: List[List[str]] = []

    if constraints:
        for json_element in constraints.keys():
            if constraints[json_element] == constants.OPTIONS_ADJUST:
                constraints[json_element] = inference_options[0].answer_options[0].answer_text

    for i in range(len(interviews)):
        assistant_messages.append([])
        all_prompts.append([])

    for i in (
        tqdm.tqdm(range(max_survey_length))
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            inference_option
            for inference_option in inference_options
            if len(inference_option.order) > i
        ]
        current_surveys = [surv for surv in interviews if len(surv._questions) > i]

        first_question: bool = i == 0

        prompts = [
            inference.create_single_question(
                inference.order[i], task_instruction=first_question
            )
            for inference in current_batch
        ]
        questions = [
            inference.create_single_question(inference.order[i], task_instruction=False)
            for inference in current_batch
        ]
        for c in range(len(current_surveys)):
            all_prompts[c].append(prompts[c])

        current_assistant_messages: List[str] = []

        missing_indeces = []

        for index, surv in enumerate(current_surveys):
            prefilled_answer = surv._questions[i].prefilled_response
            if prefilled_answer:
                current_assistant_messages.append(prefilled_answer)
                missing_indeces.append(index)

        current_batch = [
            item for a, item in enumerate(current_batch) if a not in missing_indeces
        ]

        if len(current_batch) == 0:
            for c in range(len(current_surveys)):
                assistant_messages[c].append(current_assistant_messages[c])
            for survey_id, question, llm_response, item in zip(
                range(len(current_surveys)),
                questions,
                current_assistant_messages,
                current_surveys,
            ):
                question_llm_response[survey_id].update(
                    {
                        item._questions[i].item_id: QuestionLLMResponseTuple(
                            question, llm_response
                        )
                    }
                )
            continue
        if json_structured_output:
            system_messages = [
                inference.json_system_prompt(json_options=json_structure)
                for inference in current_batch
            ]
        else:
            system_messages = [inference.system_prompt for inference in current_batch]

        output = batch_turn_by_turn_generation(
            model=model,
            system_messages=system_messages,
            prompts=all_prompts,
            assistant_messages=assistant_messages,
            guided_decoding_options= "json" if json_structured_output else None,
            json_fields=json_structure,
            constraints=constraints,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )

        for num, index in enumerate(missing_indeces):
            output.insert(index, current_assistant_messages[num])
        for survey_id, question, llm_response, item in zip(
            range(len(current_surveys)), questions, output, current_surveys
        ):
            question_llm_response[survey_id].update(
                {
                    item._questions[i].item_id: QuestionLLMResponseTuple(
                        question, llm_response
                    )
                }
            )

        for o in range(len(output)):
            assistant_messages[o].append(output[o])
        # assistant_messages.append(output)

        _intermediate_saves(interviews, n_save_step, intermediate_save_file, question_llm_response, i)

    for i, survey in enumerate(interviews):
        survey_results.append(InterviewResult(survey, question_llm_response[i]))

    return survey_results


if __name__ == "__main__":
    pass
