from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.outputs import RequestOutput

import torch

import numpy as np

import asyncio
import threading

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from typing import Any, List, Optional, Union, Dict, Literal

from .dynamic_pydantic import generate_pydantic_model

import json

import random

from dataclasses import dataclass


@dataclass
class StructuredOutputOptions:
    category: Literal["choice", "json"]
    json_fields: Optional[List[str]] = None
    constraints: Optional[Dict[str, List[str]]] = None
    allowed_choices: Optional[List[str]] = None


def default_model_init(model_id: str, seed: int = 42, **model_keywords) -> LLM:
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
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=2**32, size=batch_size).tolist()


# TODO Structured output for API calls
def batch_generation(
    model: Union[LLM, AsyncOpenAI],
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    structured_output_options: Optional[
        Union[StructuredOutputOptions, List[StructuredOutputOptions]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    **generation_kwargs: Any,
):
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

    if isinstance(model, LLM):
        sampling_params_list = _create_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            structured_output_options=structured_output_options,
            **generation_kwargs,
        )
        outputs: List[RequestOutput] = model.chat(
            batch_messages,
            sampling_params=sampling_params_list,
            use_tqdm=print_progress,
        )
        result = [output.outputs[0].text for output in outputs]

    else:
        result = _run_async_in_thread(
            client=model,
            client_model_name=client_model_name,
            batch_messages=batch_messages,
            seeds=seeds,
            concurrency_limit=api_concurrency,
            structured_output_options=structured_output_options,
            **generation_kwargs,
        )

    # TODO add argurment to specify how many conversations should be printed (base argument should be reasonable)
    if print_conversation:
        conversation_print = "Conversation:"
        for system_message, prompt, answer in zip(system_messages, prompts, result):
            round_print = f"{conversation_print}\nSystem Message:\n{system_message}\nUser Message:\n{prompt}\nGenerated Message\n{answer}"
            print(round_print, flush=True)
            break

    return result


def _make_cache_key(fields: Any, constraints: Any) -> str:
    return json.dumps({"fields": fields, "constraints": constraints}, sort_keys=False)


def _create_sampling_params(
    batch_size: int,
    seeds: List[int],
    structured_output_options: Optional[
        Union[StructuredOutputOptions, List[StructuredOutputOptions]]
    ],
    use_vllm: bool = True,
    **generation_kwargs: Any,
) -> Union[List[SamplingParams], Dict[str, Any]]:
    if structured_output_options:
        sampling_params_list = _structured_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            structured_output_options=structured_output_options,
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
    structured_output_options: Union[
        StructuredOutputOptions, List[StructuredOutputOptions]
    ],
    use_vllm: bool = True,
    **generation_kwargs: Any,
) -> Union[List[SamplingParams], Dict[str, Any]]:
    if isinstance(structured_output_options, StructuredOutputOptions):
        if structured_output_options.category == "json":
            pydantic_model = generate_pydantic_model(
                fields=structured_output_options.json_fields,
                constraints=structured_output_options.constraints,
            )
            json_schema = pydantic_model.model_json_schema()
            if use_vllm:
                global_guided_decoding = GuidedDecodingParams(json=json_schema)
                guided_decodings = [global_guided_decoding] * batch_size
            else:
                guided_decodings = [json_schema] * batch_size
        elif structured_output_options.category == "choice":
            if use_vllm:
                global_guided_decoding = GuidedDecodingParams(
                    choice=structured_output_options.allowed_choices
                )
                guided_decodings = [global_guided_decoding] * batch_size
            else:
                guided_decodings = [
                    structured_output_options.allowed_choices
                ] * batch_size

    else:
        guided_decodings = []
        cache: Dict[str, GuidedDecodingParams] = {}

        for i in range(batch_size):
            if structured_output_options[i].category == "json":
                fields = structured_output_options[i].json_fields
                cons = structured_output_options[i].constraints

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
            elif structured_output_options[i].category == "choice":
                choice = structured_output_options[i].allowed_choices

                key = _make_cache_key(choice, None)

                if key not in cache:
                    if use_vllm:
                        cache[key] = GuidedDecodingParams(choice=choice)
                    else:
                        cache[key] = choice
                guided_decodings.append(cache[key])

    if use_vllm:
        sampling_params_list = [
            SamplingParams(
                seed=seeds[i],
                guided_decoding=guided_decodings[i],
                **generation_kwargs,
            )
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
    structured_output_options: Optional[
        Union[StructuredOutputOptions, List[StructuredOutputOptions]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    **generation_kwargs,
) -> List[str]:
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
            structured_output_options=structured_output_options,
            **generation_kwargs,
        )
        outputs: List[RequestOutput] = model.chat(
            batch_messages,
            sampling_params=sampling_params_list,
            use_tqdm=print_progress,
        )
        result = [output.outputs[0].text for output in outputs]

    else:
        result = _run_async_in_thread(
            client=model,
            client_model_name=client_model_name,
            batch_messages=batch_messages,
            seeds=seeds,
            concurrency_limit=api_concurrency,
            structured_output_options=structured_output_options,
            **generation_kwargs,
        )

    # TODO add argurment to specify how many conversations should be printed
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
            print(round_print, flush=True)
            break

    return result


def _run_async_in_thread(
    client: AsyncOpenAI,
    client_model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    seeds: List[int],
    concurrency_limit: int = 10,
    structured_output_options: Optional[
        Union[StructuredOutputOptions, List[StructuredOutputOptions]]
    ] = None,
    **generation_kwargs,
):
    result_container = {}

    structured_output = _create_sampling_params(
        batch_size=len(batch_messages),
        seeds=seeds,
        structured_output_options=structured_output_options,
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
                    structured_output_options=structured_output_options,
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
    structured_output_options: Optional[
        Union[StructuredOutputOptions, List[StructuredOutputOptions]]
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

            if structured_output_options:
                if structured_output_options.category == "json":
                    request_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "json_schema",
                            "schema": structured_output,
                        },
                    }
                elif structured_output_options.category == "choice":
                    request_kwargs["extra_body"] = {
                        "guided_choice": structured_output
                        }

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
