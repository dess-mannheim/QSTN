from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.outputs import RequestOutput

import torch

import asyncio
import threading

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from typing import Any, List, Optional, Union, Dict, Literal

from .dynamic_pydantic import generate_pydantic_model

import random


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

#TODO Structured output for API calls
def batch_generation(
    model: Union[LLM, AsyncOpenAI],
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    #guided_decoding_params: Optional[List[GuidedDecodingParams]] = None,
    guided_decoding_options: Optional[Literal["json"]] = None,
    json_fields: Optional[Union[List[str], List[List[str]]]] = None,
    constraints: Optional[Union[List[Dict[str, List[str]]], Dict[str, List[str]]]] = None,
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

    seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
    if isinstance(model, LLM):
        if guided_decoding_options == "json":
            sampling_params_list = _create_sampling_params(batch_size=batch_size, seeds=seeds, json_fields=json_fields, constraints=constraints, **generation_kwargs)
        else:
            sampling_params_list = [
                SamplingParams(seed=seeds[i], **generation_kwargs)
                for i in range(batch_size)
            ]

        outputs: List[RequestOutput] = model.chat(
            batch_messages, sampling_params=sampling_params_list, use_tqdm=print_progress
        )
        result = [output.outputs[0].text for output in outputs]
    
    else:
        result = _run_async_in_thread(client=model, client_model_name=client_model_name, batch_messages=batch_messages, seeds=seeds, concurrency_limit=api_concurrency, **generation_kwargs)

    # TODO add argurment to specify how many conversations should be printed (base argument should be reasonable)
    if print_conversation:
        conversation_print = "Conversation:"
        print("Conversation:")
        for system_message, prompt, answer in zip(system_messages, prompts, result):
            round_print = f"{conversation_print}\nSystem Message:\n{system_message}\nUser Message:\n{prompt}\nGenerated Message\n{answer}"
            print(round_print, flush=True)
            break

    return result

def _create_sampling_params(batch_size: int, seeds: List[int],
        json_fields: Optional[Union[List[str], List[List[str]]]] = None,  
                            constraints: Optional[List[Dict[str, List[str]]]] = None, 
                            **generation_kwargs: Any) -> List[SamplingParams]:
    if isinstance(json_fields[0], str):
        pydantic_model = generate_pydantic_model(fields=json_fields, constraints=constraints)
        json_schema = pydantic_model.model_json_schema()
        global_guided_decoding = GuidedDecodingParams(json=json_schema)
        guided_decodings = [global_guided_decoding] * batch_size
    elif isinstance(json_fields[0], list):
        guided_decodings = []
        for i in range(batch_size):
            pydantic_model = generate_pydantic_model(
                        fields=json_fields[i], constraints=constraints[i]
                    )
            json_schema = pydantic_model.model_json_schema()
            guided_decodings.append(GuidedDecodingParams(json=json_schema))

    sampling_params_list = [
                SamplingParams(
                    seed=seeds[i],
                    guided_decoding=global_guided_decoding,
                    **generation_kwargs,
                )
                for i in range(batch_size)
            ]
    return sampling_params_list


def batch_turn_by_turn_generation(
    model: LLM,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[List[str]] = [["Hi there! What is your name?", "Interesting"]],
    assistant_messages: List[List[str]] = None,
    guided_decoding_options: Optional[Literal["json"]] = None,
    json_fields: Optional[Union[List[str], List[List[str]]]] = None,
    constraints: Optional[List[Dict[str, List[str]]]] = None,
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

    seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
    
    if isinstance(model, LLM):
        if guided_decoding_options == "json":
            sampling_params_list = _create_sampling_params(batch_size=batch_size, seeds=seeds, json_fields=json_fields, constraints=constraints, **generation_kwargs)
        else:
            sampling_params_list = [
                SamplingParams(seed=seeds[i], **generation_kwargs)
                for i in range(batch_size)
            ]

        outputs: List[RequestOutput] = model.chat(
            batch_messages, sampling_params=sampling_params_list, use_tqdm=print_progress
        )
        result = [output.outputs[0].text for output in outputs]
    else:
        result = _run_async_in_thread(client=model, client_model_name=client_model_name, batch_messages=batch_messages, seeds=seeds, concurrency_limit=api_concurrency, **generation_kwargs)

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

def _run_async_in_thread(client: AsyncOpenAI, client_model_name: str, batch_messages:List[List[Dict[str, str]]], seeds:List[int], concurrency_limit:int = 10, **generation_kwargs):
    result_container = {}

    def thread_target():
        try:
            res = asyncio.run(_run_api_batch_async(client=client, client_model_name=client_model_name, batch_messages=batch_messages, seeds=seeds, concurrency_limit=concurrency_limit, **generation_kwargs))
            result_container['result'] = res
        except Exception as e:
            result_container['error'] = e

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()

    if 'error' in result_container:
        raise result_container['error']
    
    return result_container.get('result')

async def _run_api_batch_async(client: AsyncOpenAI, client_model_name: str, batch_messages:List[List[Dict[str, str]]], seeds:List[int], concurrency_limit:int = 10, **generation_kwargs) -> List[str]:
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def get_completion(messages, seed) -> ChatCompletion:
            async with semaphore:
                # The semaphore ensures we don't send too many requests at once.
                # If a rate limit error *still* occurs, the client's
                # max_retries will have to be specified.
                return await client.chat.completions.create(
                    model=client_model_name,
                    messages=messages,
                    seed=seed,
                    **generation_kwargs
                )

        tasks = [get_completion(messages, seed) for messages, seed in zip(batch_messages, seeds)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for res in responses:
            if isinstance(res, Exception):
                print(f"A request failed permanently after all retries: {res}")
                final_results.append(f"Error: {res}")
            else:
                final_results.append(res.choices[0].message.content)
        
        return final_results