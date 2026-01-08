from openai import AsyncOpenAI

from typing import Tuple, List, Optional, Union, Any, Dict

from .response_generation import (
    ResponseGenerationMethod,
    JSONResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)

from ..utilities.utils import generate_seeds, _make_cache_key

from .dynamic_pydantic import _generate_pydantic_model

from .reasoning_parser import parse_reasoning

import asyncio
import threading

from tqdm.asyncio import tqdm_asyncio


def run_openai_batch(
    model: AsyncOpenAI,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    print_progress: bool = True,
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

    plain_results, reasoning_output, logprob_result = _run_async_in_thread(
        client=model,
        client_model_name=client_model_name,
        batch_messages=batch_messages,
        seeds=seeds,
        concurrency_limit=api_concurrency,
        response_generation_method=response_generation_method,
        print_progress=print_progress,
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
        **generation_kwargs,
    )

    return (plain_results, reasoning_output, logprob_result)


def run_openai_batch_conversation(
    model: AsyncOpenAI,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    assistant_messages: List[List[str]] = None,
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    print_progress: bool = True,
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

    plain_results, reasoning_output, logprob_result = _run_async_in_thread(
        client=model,
        client_model_name=client_model_name,
        batch_messages=batch_messages,
        seeds=seeds,
        concurrency_limit=api_concurrency,
        response_generation_method=response_generation_method,
        print_progress=print_progress,
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
        **generation_kwargs,
    )

    return (plain_results, reasoning_output, logprob_result)


def _run_async_in_thread(
    client: AsyncOpenAI,
    client_model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    seeds: List[int],
    concurrency_limit: int = 10,
    print_progress: bool = True,
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    **generation_kwargs,
):
    result_container = {}

    logprob_config = _update_logprob_kwargs(
        response_generation_method, generation_kwargs
    )

    sampling_params = _create_structured_output(
        batch_size=len(batch_messages),
        response_generation_method=response_generation_method,
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
                    print_progress=print_progress,
                    response_generation_method=response_generation_method,
                    sampling_params=sampling_params,
                    logprob_config=logprob_config,
                    reasoning_start_token=reasoning_start_token,
                    reasoning_end_token=reasoning_end_token,
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
    print_progress: bool = True,
    sampling_params: List[Dict[str, Any]] = [],
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    logprob_config: Optional[LogprobResponseGenerationMethod] = None,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    **generation_kwargs,
) -> List[str]:
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def get_completion(
        messages: list,
        seed: int,
        sampling_params: Optional[Union[Dict[str, Any], List[str]]] = None,
        response_generation_method: Optional[ResponseGenerationMethod] = None,
        **generation_kwargs,
    ):
        async with semaphore:
            request_kwargs = {
                "model": client_model_name,
                "messages": messages,
                "seed": seed,
                **generation_kwargs,
            }

            if response_generation_method:
                if isinstance(response_generation_method, JSONResponseGenerationMethod):
                    request_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "json_schema",
                            "schema": sampling_params,
                        },
                    }
                elif isinstance(
                    response_generation_method, ChoiceResponseGenerationMethod
                ):
                    if (
                        False
                    ):  # We could use this if we can ensure that the api is vllm.
                        request_kwargs["extra_body"] = {
                            "structured_outputs": {"choice": sampling_params}
                        }
                    else:
                        request_kwargs["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "Choice",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "selection": {
                                            "type": "string",
                                            "enum": sampling_params,
                                        }
                                    },
                                    "required": ["selection"],
                                    "additionalProperties": False,
                                },
                            },
                        }

            return await client.chat.completions.create(**request_kwargs)

    # pbar = tqdm.tqdm if print_progress else lambda x: x

    if sampling_params:
        tasks = [
            get_completion(messages, seed, struct_output, rgm, **generation_kwargs)
            for messages, seed, struct_output, rgm in zip(
                batch_messages,
                seeds,
                sampling_params,
                response_generation_method,
            )
        ]
    else:
        tasks = [
            get_completion(messages, seed, **generation_kwargs)
            for messages, seed in zip(batch_messages, seeds)
        ]
    if print_progress:
        responses = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Processing Prompts")
    else:
        responses = await asyncio.gather(
            *tasks, return_exceptions=True
        )

    final_results = []
    reasoning_output = []
    logprob_result = []

    patterns = [
        (reasoning_start_token, reasoning_end_token),
    ]

    for response in responses:
        if isinstance(response, Exception):
            print(f"A request failed permanently after all retries: {response}")
            final_results.append(f"Error: {response}")
        else:
            msg = response.choices[0].message

            # Automatic reasoning parsing
            reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "reasoning_content", None
            )

            if reasoning == None:
                # Fallback to parsing manually
                final_answer, extracted_reasoning = parse_reasoning(
                    msg.content, patterns=patterns
                )
                final_results.append(final_answer)
                if extracted_reasoning:
                    reasoning_output.append(extracted_reasoning.strip())
                else:
                    reasoning_output.append(extracted_reasoning)
            else:
                final_results.append(msg.content)
                reasoning_output.append(reasoning)
            if logprob_config and response.choices[0].logprobs:
                logprob_result.append(
                    [
                        [
                            {"token": top.token, "logprob": top.logprob}
                            for top in lp.top_logprobs
                        ]
                        for lp in response.choices[0].logprobs.content
                    ]
                )
            else:
                logprob_result.append(None)

    return final_results, logprob_result, reasoning_output


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
        generation_kwargs["logprobs"] = True
        generation_kwargs["top_logprobs"] = logprob_config.top_logprobs
        if logprob_config.token_limit is not None:
            generation_kwargs["max_tokens"] = logprob_config.token_limit

    return logprob_config


def _create_structured_output(
    batch_size: int,
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ],
) -> Dict[str, Any]:
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
        return _create_structured_params(
            batch_size=batch_size,
            response_generation_method=response_generation_method,
        )

    return None


def _create_structured_params(
    batch_size: int,
    response_generation_method: Union[
        ResponseGenerationMethod, List[ResponseGenerationMethod]
    ],
) -> List[Dict[str, Any]]:

    structured_output = []

    # Same for all calls
    if isinstance(response_generation_method, ResponseGenerationMethod):
        if isinstance(response_generation_method, JSONResponseGenerationMethod):
            pydantic_model = _generate_pydantic_model(
                fields=response_generation_method.json_fields,
                constraints=response_generation_method.constraints,
            )
            json_schema = pydantic_model.model_json_schema()
            structured_output = [json_schema] * batch_size
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
            structured_output = [_allowed_choices] * batch_size

    # Different response generation methods for each question
    else:
        structured_output = []
        cache: Dict[str, Any] = {}
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
                    cache[key] = json_schema

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
                    cache[key] = _allowed_choices
                structured_output.append(cache[key])
            else:
                structured_output.append(None)

    return structured_output
