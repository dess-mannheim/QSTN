from vllm import LLM, SamplingParams

from vllm.sampling_params import GuidedDecodingParams

from typing import Any, List, Optional

import random

from vllm.outputs import RequestOutput

import torch

import gc

def default_model_init(model_id: str, seed=42) -> LLM:
    random.seed(seed)
    torch.manual_seed(seed)
    print("Device_count: " + str(torch.cuda.device_count()))

    return LLM(model=model_id, tensor_parallel_size=torch.cuda.device_count(), seed=seed, max_model_len=2000, enable_prefix_caching=True)

def shutdown_model(model: LLM) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None

def batch_generation(model = LLM, system_messages:list[str]=["You are a helpful assistant."], prompts:list[str]=["Hi there! What is your name?"], guided_decoding_params: Optional[List[GuidedDecodingParams]] = None, seed: int = 42, print_conversation:bool=False, print_progress:bool=True, **generation_kwargs: Any):
    random.seed(seed)
    
    # Prepare batch of messages
    batch_messages = [
    [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for system_message, prompt in zip(system_messages, prompts)
    ]

    batch_size = len(system_messages)

    seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]

    if guided_decoding_params:
        sampling_params_list = [
            SamplingParams(seed=seeds[i], guided_decoding=guided_decoding_params[i], **generation_kwargs)
            for i in range(batch_size)
        ]
    else:
        sampling_params_list = [
            SamplingParams(seed=seeds[i], **generation_kwargs)
            for i in range(batch_size)
        ]

    outputs:List[RequestOutput] = model.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=print_progress)
    result = [output.outputs[0].text for output in outputs]

    if print_conversation:
        print("Conversation:")
        for system_message, prompt, answer in zip(system_messages, prompts, result):
            print("System Message")
            print(system_message)
            print("User Message")
            print(prompt)
            print("Generated Message")
            print(answer)

    return result

    
def batch_turn_by_turn_generation(model:LLM, system_messages:List[str]=["You are a helpful assistant."], prompts:List[List[str]]=[["Hi there! What is your name?", "Interesting"]], assistant_messages:List[List[str]]=None, guided_decoding_params: Optional[List[GuidedDecodingParams]] = None, seed: int = 42, print_conversation:bool=False, print_progress:bool=True, **generation_kwargs) -> List[str]:
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

        #TODO this implementation is wrong, because assistant messages supports a dict, so they can be anywhere and not just at the beginning
        for j in range(num_user_msgs):
            messages.append({"role": "user", "content": prompts[i][j]})
            if j < num_assistant_msgs:
                messages.append({"role": "assistant", "content": assistant_messages[i][j]})

        batch_messages.append(messages)

    seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
    
    if guided_decoding_params:
        sampling_params_list = [
            SamplingParams(seed=seeds[i], guided_decoding=guided_decoding_params[i], **generation_kwargs)
            for i in range(batch_size)
        ]
    else:
        sampling_params_list = [
            SamplingParams(seed=seeds[i], **generation_kwargs)
            for i in range(batch_size)
        ]

    #print(batch_messages, flush=True)

    outputs: List[RequestOutput] = model.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=print_progress)
    result = [output.outputs[0].text for output in outputs]
    
    if print_conversation:
        print("Conversation:")
        for system_message, prompt_list, assistant_list, answer in zip(system_messages, prompts, assistant_messages, result):
            print("System Prompt:")
            print(system_message)
            for j in range(len(prompt_list)):
                print("User Message:")
                print(prompt_list[j])
                if j < len(assistant_list):
                    prefill = assistant_list[j]
                    if prefill:
                        print("Assistant Message")
                        print(assistant_list[j])
            print("Generated Answer")
            print(answer)

    return result