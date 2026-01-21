from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.outputs import RequestOutput

from typing import List, Optional, Union, Dict, Any, Tuple

from .response_generation import ResponseGenerationMethod
from .reasoning_parser import parse_reasoning

from ..utilities.utils import generate_seeds, _make_cache_key

from .dynamic_pydantic import _generate_pydantic_model
from .response_generation import (
    ResponseGenerationMethod,
    JSONResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)

import random
import torch

import json
import re

def run_vllm_batch(
    model: LLM,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    seed: int = 42,
    # number_of_printed_conversation: int = 2,
    print_progress: bool = True,
    # <think>...</think> tokens are used by Qwen3 to separate reasoning
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    space_char: str = "Ġ",
    **generation_kwargs: Any,
) -> Tuple[List[str], List[str], List[str]]:

    # Prepare batch of messages
    batch_messages: List[List[Dict[str, str]]] = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        for system_message, prompt in zip(system_messages, prompts)
    ]

    batch_size: int = len(system_messages)

    seeds = generate_seeds(seed, batch_size=batch_size)

    logprob_result = None
    logprob_config = _update_logprob_kwargs(
        response_generation_method, generation_kwargs
    )

    # If users specify use_tqdm themselves, we use that flag instead
    print_progress = generation_kwargs.pop("use_tqdm", print_progress)

    if "sampling_params" in generation_kwargs.keys():
        import warnings

        warnings.warn(
            "Do not specify sampling_params for vllm inference. If you want to use hyperparameters, "
            "add them directly to the generation kwargs. Given argument sampling_params will be ignored."
        )
        generation_kwargs.pop("sampling_params")

    gen_kwargs, chat_kwargs = _split_kwargs(generation_kwargs)

    sampling_params_list = _create_sampling_params(
        batch_size=batch_size,
        seeds=seeds,
        response_generation_method=response_generation_method,
        **gen_kwargs,
    )

    outputs: List[RequestOutput] = model.chat(
        batch_messages,
        sampling_params=sampling_params_list,
        use_tqdm=print_progress,
        **chat_kwargs,
    )

    raw_reasonings, reasoning_outputs, plain_results = _extract_reasoning_and_answer(
        reasoning_start_token, reasoning_end_token, outputs
    )

    if logprob_config:
        logprob_result = _get_logprobs(
            model,
            response_generation_method,
            reasoning_start_token,
            reasoning_end_token,
            space_char,
            outputs,
            raw_reasonings,
        )
    else:
        logprob_result = [None] * len(plain_results)

    return (plain_results, logprob_result, reasoning_outputs)


def run_vllm_batch_conversation(
    model: LLM,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    assistant_messages: List[List[str]] = None,
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    seed: int = 42,
    # number_of_printed_conversation: int = 2,
    print_progress: bool = True,
    # <think>...</think> tokens are used by Qwen3 to separate reasoning
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    space_char: str = "Ġ",
    **generation_kwargs: Any,
) -> Tuple[List[str], List[str], List[str]]:

        
    batch_messages = []
    batch_size = len(system_messages)
    for i in range(batch_size):
        messages = []

        # Add system message
        if system_messages[i]:
            messages.append({"role": "system", "content": system_messages[i]})

        num_user_msgs = len(prompts[i])
        num_assistant_msgs = len(assistant_messages[i])

        for j in range(num_user_msgs):
            messages.append({"role": "user", "content": prompts[i][j]})
            if j < num_assistant_msgs:
                messages.append(
                    {"role": "assistant", "content": assistant_messages[i][j]}
                )

        batch_messages.append(messages)

    seeds = generate_seeds(seed, batch_size=batch_size)

    logprob_result = None
    logprob_config = _update_logprob_kwargs(
        response_generation_method, generation_kwargs
    )

    # If users specify use_tqdm themselves, we use that flag instead
    print_progress = generation_kwargs.pop("use_tqdm", print_progress)

    if "sampling_params" in generation_kwargs.keys():
        import warnings

        warnings.warn(
            "Do not specify sampling_params for vllm inference. If you want to use hyperparameters, "
            "add them directly to the generation kwargs. Given argument sampling_params will be ignored."
        )
        generation_kwargs.pop("sampling_params")

    gen_kwargs, chat_kwargs = _split_kwargs(generation_kwargs)

    sampling_params_list = _create_sampling_params(
        batch_size=batch_size,
        seeds=seeds,
        response_generation_method=response_generation_method,
        **gen_kwargs,
    )

    outputs: List[RequestOutput] = model.chat(
        batch_messages,
        sampling_params=sampling_params_list,
        use_tqdm=print_progress,
        **chat_kwargs,
    )

    raw_reasonings, reasoning_outputs, plain_results = _extract_reasoning_and_answer(
        reasoning_start_token, reasoning_end_token, outputs
    )

    if logprob_config:
        logprob_result = _get_logprobs(
            model,
            response_generation_method,
            reasoning_start_token,
            reasoning_end_token,
            space_char,
            outputs,
            raw_reasonings,
        )

    return (plain_results, logprob_result, reasoning_outputs)


def default_model_init(model_id: str, seed: int = 42, **model_keywords) -> LLM:
    """
    Initialize a vLLM model with default settings.

    Args:
        model_id: HuggingFace model identifier
        seed: Random seed for reproducibility
        **model_keywords: Additional keywords passed to LLM constructor

    Returns:
        LLM: Initialized vLLM model instance
    """
    random.seed(seed)
    torch.manual_seed(seed)
    print("Device_count: " + str(torch.cuda.device_count()))
    print(model_keywords)

    return LLM(
        model=model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=seed,
        **model_keywords,
    )


def _get_sampling_field_names() -> set[str]:
    """
    Dynamically fetch valid arguments for SamplingParams.
    """
    import inspect

    # inspect.signature is the most robust way to get constructor arguments
    sig = inspect.signature(SamplingParams)
    return set(sig.parameters.keys())


def _split_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Splits kwargs into (generation_args, chat_args).
    """
    sampling_keys = _get_sampling_field_names()

    generation_args = {}
    chat_args = {}

    for key, value in kwargs.items():
        if key in sampling_keys:
            generation_args[key] = value
        else:
            chat_args[key] = value

    return generation_args, chat_args


def _structured_sampling_params(
    batch_size: int,
    seeds: List[int],
    response_generation_method: Union[
        ResponseGenerationMethod, List[ResponseGenerationMethod]
    ],
    **generation_kwargs: Any,
) -> List[SamplingParams]:

    structured_output = []

    # Same for all calls
    if isinstance(response_generation_method, ResponseGenerationMethod):
        if isinstance(response_generation_method, JSONResponseGenerationMethod):
            pydantic_model = _generate_pydantic_model(
                fields=response_generation_method.json_fields,
                constraints=response_generation_method.constraints,
            )
            json_schema = pydantic_model.model_json_schema()
            global_structured_output = StructuredOutputsParams(json=json_schema)
            structured_output = [global_structured_output] * batch_size
            # remote inference
            # else:
            #     structured_output = [json_schema] * batch_size
        elif (
            isinstance(
                response_generation_method,
                (ChoiceResponseGenerationMethod, LogprobResponseGenerationMethod),
            )
            and response_generation_method.allowed_choices is not None
        ):
            _allowed_choices = [
                str(c) for c in response_generation_method.allowed_choices
            ]
            global_structured_output = StructuredOutputsParams(choice=_allowed_choices)
            structured_output = [global_structured_output] * batch_size
            # Remote Inference
            # else:
            # structured_output = [_allowed_choices] * batch_size

    # Different response generation methods for each question
    else:
        structured_output = []
        cache: Dict[str, StructuredOutputsParams] = {}
        for i in range(batch_size):
            current_method = response_generation_method[i]
            if isinstance(current_method, JSONResponseGenerationMethod):
                fields = current_method.json_fields
                cons = current_method.constraints

                key = _make_cache_key(fields, cons)

                if key not in cache:
                    pydantic_model = _generate_pydantic_model(
                        fields=fields, constraints=cons
                    )
                    json_schema = pydantic_model.model_json_schema()
                    cache[key] = StructuredOutputsParams(json=json_schema)

                    # Remote Inference
                    # else:
                    #     cache[key] = json_schema

                structured_output.append(cache[key])
            elif (
                isinstance(
                    current_method,
                    (ChoiceResponseGenerationMethod, LogprobResponseGenerationMethod),
                )
                and current_method.allowed_choices is not None
            ):
                _allowed_choices = [str(c) for c in current_method.allowed_choices]

                key = _make_cache_key(_allowed_choices, None)
                if key not in cache:
                    cache[key] = StructuredOutputsParams(choice=_allowed_choices)
                    # Remote Inference
                    # else:
                    #     cache[key] = _allowed_choices
                structured_output.append(cache[key])
            else:
                structured_output.append(None)

    if len(structured_output) == batch_size:
        sampling_params_list = [
            SamplingParams(
                seed=seeds[i],
                structured_outputs=structured_output[i],
                **generation_kwargs,
            )
            for i in range(batch_size)
        ]
    else:
        sampling_params_list = [
            SamplingParams(seed=seeds[i], **generation_kwargs)
            for i in range(batch_size)
        ]
    # Remote Inference
    # else:
    #     return structured_output

    return sampling_params_list


def _create_sampling_params(
    batch_size: int,
    seeds: List[int],
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ],
    **generation_kwargs: Any,
) -> List[SamplingParams]:
    """
    Create sampling parameters for generation.

    Args:
        batch_size: Number of prompts in batch
        seeds: Random seeds for generation
        answer_production_method: Output structure configuration
        use_vllm: If True, creates vLLM parameters
        **generation_kwargs: Additional sampling parameters

    Returns:
        Sampling parameters for vLLM or API configuration
    """

    use_structured: bool = response_generation_method and isinstance(
        response_generation_method, (list, ResponseGenerationMethod)
    )

    if use_structured:
        return _structured_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            response_generation_method=response_generation_method,
            **generation_kwargs,
        )

    return [
        SamplingParams(seed=seeds[i], **generation_kwargs) for i in range(batch_size)
    ]

def _get_logprobs(
    model,
    response_generation_method,
    reasoning_start_token,
    reasoning_end_token,
    space_char,
    outputs,
    raw_reasonings,
):
    logprob_result = []
    # ignore the first k tokens that belong to the reasoning
    rgms: List[LogprobResponseGenerationMethod] = []
    if isinstance(response_generation_method, LogprobResponseGenerationMethod):
        rgms.append(response_generation_method)
    elif isinstance(response_generation_method, list):
        rgms: List[LogprobResponseGenerationMethod] = [
            rgm for rgm in response_generation_method
        ]
    for rgm in rgms:
        if rgm.ignore_reasoning:
            tokenizer = model.get_tokenizer()
            logprob_positions = [
                (
                    len(
                        tokenizer.tokenize(
                            f"{reasoning_start_token}{_reasoning}{reasoning_end_token}"
                        )
                    )
                    + 1
                    + rgm.token_position
                    if _reasoning is not None
                    else rgm.token_position
                )
                for _reasoning in raw_reasonings
            ]
        else:
            logprob_positions = [rgm.token_position] * len(outputs)

        for req_output, logprob_position in zip(outputs, logprob_positions):
            try:
                answer_dict = {
                    x.decoded_token.lstrip(
                        space_char
                    ).lstrip(): x.logprob  # strip the space character and whitespace from tokenization
                    for x in req_output.outputs[0].logprobs[logprob_position].values()
                }
            except IndexError:  # less than [logprob_position] tokens in the output!
                answer_dict = {}
            logprob_result.append(answer_dict)
    return logprob_result


def _update_logprob_kwargs(response_generation_method, generation_kwargs):
    logprob_config = None

    if isinstance(response_generation_method, LogprobResponseGenerationMethod):
        logprob_config = response_generation_method
    elif isinstance(response_generation_method, list):
        logprob_config = next(
            (
                item
                for item in response_generation_method
                if isinstance(item, LogprobResponseGenerationMethod)
            ),
            None,
        )
    if logprob_config:
        generation_kwargs["logprobs"] = logprob_config.top_logprobs
        if logprob_config.token_limit is not None:
            generation_kwargs["max_tokens"] = logprob_config.token_limit

    return logprob_config

    # if response_generation_method:
    #     for rgm in response_generation_method:
    #         # TODO This is not implemented correcty yet
    #         if isinstance(rgm, LogprobResponseGenerationMethod):
    #             logprob_result = []
    #             # ignore the first k tokens that belong to the reasoning
    #             if rgm.ignore_reasoning:
    #                 tokenizer = model.get_tokenizer()
    #                 logprob_positions = [
    #                     (
    #                         len(
    #                             tokenizer.tokenize(
    #                                 f"{reasoning_start_token}{_reasoning}{reasoning_end_token}"
    #                             )
    #                         )
    #                         + 1
    #                         + rgm.token_position
    #                         if _reasoning is not None
    #                         else rgm.token_position
    #                     )
    #                     for _reasoning in raw_reasonings
    #                 ]
    #             else:
    #                 logprob_positions = [rgm.token_position] * len(outputs)

    #             for req_output, logprob_position in zip(outputs, logprob_positions):
    #                 try:
    #                     answer_dict = {
    #                         x.decoded_token.lstrip(
    #                             space_char
    #                         ).lstrip(): x.logprob  # strip the space character and whitespace from tokenization
    #                         for x in req_output.outputs[0]
    #                         .logprobs[logprob_position]
    #                         .values()
    #                     }
    #                 except (
    #                     IndexError
    #                 ):  # less than [logprob_position] tokens in the output!
    #                     answer_dict = {}
    #                 logprob_result.append(answer_dict)


# REGEX Method.
# We could use the thinking ids directly and decode the message again.
def _extract_reasoning_and_answer(
    reasoning_start_token: str, reasoning_end_token: str, outputs: List[RequestOutput]
):
    plain_results = []
    reasoning_output = []
    raw_reasonings = []  # keep the whitespace for length calculations

    patterns = [
        (reasoning_start_token, reasoning_end_token),
    ]

    for request_output in outputs:
        full_text = request_output.outputs[0].text

        # If we have no reasoning, directly output everything
        extracted_reasoning = None
        final_answer = full_text

        final_answer, extracted_reasoning = parse_reasoning(full_text, patterns=patterns)
       
        raw_reasonings.append(extracted_reasoning)
        if extracted_reasoning != None:
            reasoning_output.append(extracted_reasoning.strip())
        else:
            reasoning_output.append(extracted_reasoning)
        plain_results.append(final_answer)

    return raw_reasonings, reasoning_output, plain_results
