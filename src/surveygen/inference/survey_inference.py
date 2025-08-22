from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.outputs import RequestOutput

import torch

import numpy as np

import asyncio
import threading

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from typing import Any, List, Optional, Union, Dict

from .dynamic_pydantic import generate_pydantic_model
from .answer_production import (
    AnswerProductionMethod,
    JSON_AnswerProductionMethod,
    Choice_AnswerProductionMethod,
    Logprob_AnswerProductionMethod
)

import json
import random

from tqdm.auto import tqdm


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

    return LLM(
        model=model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=seed,
        **model_keywords,
    )


def _generate_seeds(seed: int, batch_size: int) -> List[int]:
    """
    Generate a list of random seeds.

    Args:
        seed: Base random seed
        batch_size: Number of seeds to generate

    Returns:
        List[int]: Generated random seeds
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=2**32, size=batch_size).tolist()


# TODO Structured output for API calls
def batch_generation(
    model: Union[LLM, AsyncOpenAI],
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    answer_production_method: Optional[
        Union[AnswerProductionMethod, List[AnswerProductionMethod]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    **generation_kwargs: Any,
):
    """
    Generate responses for a batch of prompts.

    Handles both vLLM and OpenAI API generation with support for:
    - Structured output (JSON or choice format)
    - Conversation printing
    - Progress tracking
    - Concurrent API requests

    Args:
        model: vLLM model or AsyncOpenAI client
        system_messages: System prompts for each conversation
        prompts: User prompts to generate responses for
        answer_production_method: Configuration for structured output
        seed: Random seed for reproducibility
        client_model_name: Model name when using OpenAI API
        api_concurrency: Max concurrent API requests
        print_conversation: If True, prints conversations
        print_progress: If True, shows progress bar
        **generation_kwargs: Additional generation parameters

    Returns:
        List[str]: Generated responses
    """
    random.seed(seed)

    # Prepare batch of messages
    batch_messages: List[List[Dict[str, str]]] = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        for system_message, prompt in zip(system_messages, prompts)
    ]

    batch_size: int = len(system_messages)

    seeds = _generate_seeds(seed, batch_size=batch_size)

    logprob_result = None

    if isinstance(model, LLM):
        # TODO: add support for List[AnswerProductionMethod]
        if isinstance(answer_production_method, Logprob_AnswerProductionMethod):
            generation_kwargs["logprobs"] = answer_production_method.top_logprobs
            generation_kwargs["max_tokens"] = answer_production_method.token_position + 1

        sampling_params_list = _create_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            answer_production_method=answer_production_method,
            **generation_kwargs,
        )
        outputs: List[RequestOutput] = model.chat(
            batch_messages,
            sampling_params=sampling_params_list,
            use_tqdm=print_progress,
        )
        result = [output.outputs[0].text for output in outputs]
        if isinstance(answer_production_method, Logprob_AnswerProductionMethod):
            logprob_result = []
            for req_output in outputs:
                logprob_position = answer_production_method.token_position
                try:
                    answer_dict = {
                        x.decoded_token: x.logprob
                        for x in req_output.outputs[0].logprobs[logprob_position].values()
                    }
                except IndexError: # less than [logprob_position] tokens in the output!
                    answer_dict = {}
                logprob_result.append(answer_dict)

    else:
        # TODO: add support for List[AnswerProductionMethod]
        if isinstance(answer_production_method, Logprob_AnswerProductionMethod):
            raise NotImplementedError("The Logprob_AnswerProductionMethod is not yet implemented " + \
                                      "for use with the OpenAI API. Use vllm offline inference instead.")
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = answer_production_method.top_logprobs

        result = _run_async_in_thread(
            client=model,
            client_model_name=client_model_name,
            batch_messages=batch_messages,
            seeds=seeds,
            concurrency_limit=api_concurrency,
            answer_production_method=answer_production_method,
            **generation_kwargs,
        )

    # TODO add argument to specify how many conversations should be printed (base argument should be reasonable)
    if print_conversation:
        conversation_print = "--- Conversation ---"
        if logprob_result is None:
            logprob_result = [None]*len(result)
        for system_message, prompt, answer, logprob_answer in zip(system_messages, prompts, result, logprob_result):
            round_print = f"{conversation_print}\n-- System Message --\n{system_message}\n-- User Message ---\n{prompt}\n-- Generated Message --\n{answer}"
            if isinstance(answer_production_method, Logprob_AnswerProductionMethod):
                round_print += '\n-- Logprobs --\n' + str(logprob_answer)
            tqdm.write(round_print)
            break

    return (result, logprob_result)


def _make_cache_key(fields: Any, constraints: Any) -> str:
    return json.dumps({"fields": fields, "constraints": constraints}, sort_keys=False)


def _create_sampling_params(
    batch_size: int,
    seeds: List[int],
    answer_production_method: Optional[
        Union[AnswerProductionMethod, List[AnswerProductionMethod]]
    ],
    use_vllm: bool = True,
    **generation_kwargs: Any,
) -> Union[List[SamplingParams], Dict[str, Any]]:
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

    if answer_production_method:
        sampling_params_list = _structured_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            answer_production_method=answer_production_method,
            use_vllm=use_vllm,
            **generation_kwargs,
        )
    else:
        if use_vllm:
            sampling_params_list = [
                SamplingParams(seed=seeds[i], **generation_kwargs)
                for i in range(batch_size)
            ]
        else:
            return []
    return sampling_params_list


def _structured_sampling_params(
    batch_size: int,
    seeds: List[int],
    answer_production_method: Union[
        AnswerProductionMethod, List[AnswerProductionMethod]
    ],
    use_vllm: bool = True,
    **generation_kwargs: Any,
) -> Union[List[SamplingParams], Dict[str, Any]]:
    
    guided_decodings = []

    if isinstance(answer_production_method, AnswerProductionMethod):
        if isinstance(answer_production_method, JSON_AnswerProductionMethod):
            pydantic_model = generate_pydantic_model(
                fields=answer_production_method.json_fields,
                constraints=answer_production_method.constraints,
            )
            json_schema = pydantic_model.model_json_schema()
            if use_vllm:
                global_guided_decoding = GuidedDecodingParams(json=json_schema)
                guided_decodings = [global_guided_decoding] * batch_size
            else:
                guided_decodings = [json_schema] * batch_size
        elif (isinstance(answer_production_method, (Choice_AnswerProductionMethod, Logprob_AnswerProductionMethod))
                  and answer_production_method.allowed_choices is not None):
            _allowed_choices = [str(c) for c in answer_production_method.allowed_choices]
            if use_vllm:
                global_guided_decoding = GuidedDecodingParams(choice=_allowed_choices)
                guided_decodings = [global_guided_decoding] * batch_size
            else:
                guided_decodings = [_allowed_choices] * batch_size

    else:
        guided_decodings = []
        cache: Dict[str, GuidedDecodingParams] = {}

        for i in range(batch_size):
            if isinstance(answer_production_method[i], JSON_AnswerProductionMethod):
                fields = answer_production_method[i].json_fields
                cons = answer_production_method[i].constraints

                key = _make_cache_key(fields, cons)

                if key not in cache:
                    pydantic_model = generate_pydantic_model(
                        fields=fields, constraints=cons
                    )
                    json_schema = pydantic_model.model_json_schema()
                    if use_vllm:
                        cache[key] = GuidedDecodingParams(json=json_schema)
                    else:
                        cache[key] = json_schema

                guided_decodings.append(cache[key])
            elif (isinstance(answer_production_method[i], (Choice_AnswerProductionMethod, Logprob_AnswerProductionMethod))
                  and answer_production_method[i].allowed_choices is not None):
                _allowed_choices = [str(c) for c in answer_production_method[i].allowed_choices]

                key = _make_cache_key(_allowed_choices, None)

                if key not in cache:
                    if use_vllm:
                        cache[key] = GuidedDecodingParams(choice=_allowed_choices)
                    else:
                        cache[key] = _allowed_choices
                guided_decodings.append(cache[key])

    if use_vllm and len(guided_decodings) == batch_size:
        sampling_params_list = [
            SamplingParams(
                seed=seeds[i],
                guided_decoding=guided_decodings[i],
                **generation_kwargs,
            )
            for i in range(batch_size)
        ]
    elif use_vllm:
        sampling_params_list = [
            SamplingParams(seed=seeds[i], **generation_kwargs)
            for i in range(batch_size)
        ]
    else:
        return guided_decodings

    return sampling_params_list


def batch_turn_by_turn_generation(
    model: LLM,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[List[str]] = [["Hi there! What is your name?", "Interesting"]],
    assistant_messages: List[List[str]] = None,
    answer_production_method: Optional[
        Union[AnswerProductionMethod, List[AnswerProductionMethod]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    **generation_kwargs,
) -> List[str]:
    """
    Generate responses for multi-turn conversations.

    Handles conversations with multiple back-and-forth exchanges between
    user and assistant. Supports:
    - Structured output formats
    - Pre-filled assistant messages
    - Conversation printing
    - Progress tracking

    Args:
        model: vLLM model or AsyncOpenAI client
        system_messages: System prompts for each conversation
        prompts: Lists of user messages for each conversation
        assistant_messages: Optional pre-filled assistant responses
        answer_production_method: Output structure configuration
        seed: Random seed for reproducibility
        client_model_name: Model name for OpenAI API
        api_concurrency: Max concurrent API requests
        print_conversation: If True, prints conversations
        print_progress: If True, shows progress bar
        **generation_kwargs: Additional generation parameters

    Returns:
        List[str]: Generated responses for each conversation
    """

    random.seed(seed)
    batch_messages = []
    batch_size = len(system_messages)
    for i in range(batch_size):
        messages = []

        # Add system message
        if system_messages[i]:
            messages.append({"role": "system", "content": system_messages[i]})

        num_user_msgs = len(prompts[i])
        num_assistant_msgs = len(assistant_messages[i])

        # TODO this implementation is wrong, because assistant messages supports a dict, so they can be anywhere and not just at the beginning
        for j in range(num_user_msgs):
            messages.append({"role": "user", "content": prompts[i][j]})
            if j < num_assistant_msgs:
                messages.append(
                    {"role": "assistant", "content": assistant_messages[i][j]}
                )

        batch_messages.append(messages)

    seeds = _generate_seeds(seed, batch_size=batch_size)

    if isinstance(model, LLM):
        sampling_params_list = _create_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            answer_production_method=answer_production_method,
            **generation_kwargs,
        )
        outputs: List[RequestOutput] = model.chat(
            batch_messages,
            sampling_params=sampling_params_list,
            use_tqdm=print_progress,
        )
        # TODO: add support for logprobs
        result = [output.outputs[0].text for output in outputs]

    else:
        result = _run_async_in_thread(
            client=model,
            client_model_name=client_model_name,
            batch_messages=batch_messages,
            seeds=seeds,
            concurrency_limit=api_concurrency,
            answer_production_method=answer_production_method,
            **generation_kwargs,
        )

    # TODO add argument to specify how many conversations should be printed
    if print_conversation:
        conversation_print = "Conversation:"
        for system_message, prompt_list, assistant_list, answer in zip(
            system_messages, prompts, assistant_messages, result
        ):
            round_print = f"{conversation_print}\nSystem Prompt:\n{system_message}"
            for j in range(len(prompt_list)):
                round_print = f"{round_print}\nUser Message:\n{prompt_list[j]}"
                if j < len(assistant_list):
                    prefill = assistant_list[j]
                    if prefill:
                        round_print = (
                            f"{round_print}\nAssistant Message:\n{assistant_list[j]}"
                        )
            round_print = f"{round_print}\nGenerated Answer:\n{answer}"
            tqdm.write(round_print)
            break

    return result


def _run_async_in_thread(
    client: AsyncOpenAI,
    client_model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    seeds: List[int],
    concurrency_limit: int = 10,
    answer_production_method: Optional[
        Union[AnswerProductionMethod, List[AnswerProductionMethod]]
    ] = None,
    **generation_kwargs,
):
    result_container = {}

    structured_output = _create_sampling_params(
        batch_size=len(batch_messages),
        seeds=seeds,
        answer_production_method=answer_production_method,
        use_vllm=False,
        **generation_kwargs,
    )

    def thread_target():
        try:
            res = asyncio.run(
                _run_api_batch_async(
                    client=client,
                    client_model_name=client_model_name,
                    batch_messages=batch_messages,
                    seeds=seeds,
                    concurrency_limit=concurrency_limit,
                    answer_production_method=answer_production_method,
                    structured_output=structured_output,
                    **generation_kwargs,
                )
            )
            result_container["result"] = res
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("result")


async def _run_api_batch_async(
    client: AsyncOpenAI,
    client_model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    seeds: List[int],
    concurrency_limit: int = 10,
    structured_output: List[Dict[str, Any]] = [],
    answer_production_method: Optional[
        Union[AnswerProductionMethod, List[AnswerProductionMethod]]
    ] = None,
    **generation_kwargs,
) -> List[str]:
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def get_completion(
        messages: list,
        seed: int,
        structured_output: Optional[Union[Dict[str, Any], List[str]]] = None,
        **generation_kwargs,
    ) -> ChatCompletion:
        async with semaphore:
            request_kwargs = {
                "model": client_model_name,
                "messages": messages,
                "seed": seed,
                **generation_kwargs,
            }

            if answer_production_method:
                if isinstance(answer_production_method, JSON_AnswerProductionMethod):
                    request_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "json_schema",
                            "schema": structured_output,
                        },
                    }
                elif isinstance(answer_production_method, Choice_AnswerProductionMethod):
                    #TODO: add warning if this is not running against vllm, i.e., guided_choice is not supported
                    request_kwargs["extra_body"] = {"guided_choice": structured_output}

            return await client.chat.completions.create(**request_kwargs)

    if len(structured_output) > 0:
        tasks = [
            get_completion(messages, seed, struct_output, **generation_kwargs)
            for messages, seed, struct_output in zip(
                batch_messages, seeds, structured_output
            )
        ]
    else:
        tasks = [
            get_completion(messages, seed, **generation_kwargs)
            for messages, seed in zip(batch_messages, seeds)
        ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for res in responses:
        if isinstance(res, Exception):
            print(f"A request failed permanently after all retries: {res}")
            final_results.append(f"Error: {res}")
        else:
            final_results.append(res.choices[0].message.content)

    return final_results
