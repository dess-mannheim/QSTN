from typing import TYPE_CHECKING, Any, List, Optional, Union, Tuple

import random

if TYPE_CHECKING:
    from vllm import LLM
    from openai import AsyncOpenAI

has_vllm = False
has_openai = False

try:
    from vllm import LLM
    from .local_inference import run_vllm_batch, run_vllm_batch_conversation
    from vllm.outputs import RequestOutput

    has_vllm = True
except ImportError:
    LLM = Any

try:
    from openai import AsyncOpenAI
    from .remote_inference import run_openai_batch, run_openai_batch_conversation

    has_openai = True
except ImportError:
    AsyncOpenAI = Any

from .response_generation import (
    ResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)

import re

from tqdm.auto import tqdm


def _print_conversation(
    system_messages: List[str],
    prompts: List[str],
    assistant_messages: List[str],
    plain_results: List[str],
    reasoning_output: List[str],
    logprob_result: List[str],
    response_generation_method: List[ResponseGenerationMethod],
    number_of_printed_conversations: int = 2,
):
    if reasoning_output == None:
        reasonings = [None] * len(system_messages)
    else:
        reasonings = reasoning_output

    if logprob_result == None:
        logprobs = [None] * len(system_messages)
    else:
        logprobs = logprob_result

    if assistant_messages:

        conversation_print = "--- Conversation ---"
        for i, (
            system_message,
            prompt_list,
            answer,
            reasoning,
            logprob_answer,
            assistant_list,
        ) in enumerate(
            zip(
                system_messages,
                prompts,
                plain_results,
                reasonings,
                logprobs,
                assistant_messages,
            )
        ):
            if i >= number_of_printed_conversations:
                break

            round_print = (
                f"{conversation_print}\n-- System Message --\n{system_message}"
            )
            for j in range(len(prompt_list)):
                round_print = f"{round_print}\n-- User Message --\n{prompt_list[j]}"
                if j < len(assistant_list):
                    prefill = assistant_list[j]
                    if prefill:
                        round_print = f"{round_print}\n-- Assistant Message --\n{assistant_list[j]}"
            round_print = f"{round_print}\n-- Generated Answer --\n{answer}"
            if reasoning:
                round_print += "\n-- Reasoning --\n" + str(reasoning)

            if i < len(response_generation_method):
                current_method = response_generation_method[i]
                if isinstance(current_method, LogprobResponseGenerationMethod):
                    round_print += "\n-- Logprobs --\n" + str(logprob_answer)
            tqdm.write(round_print)
    else:
        conversation_print = "--- Conversation ---"
        for i, (system_message, prompt, answer, reasoning, logprob_answer) in enumerate(
            zip(system_messages, prompts, plain_results, reasonings, logprobs)
        ):
            if i >= number_of_printed_conversations:
                break
            round_print = f"{conversation_print}\n-- System Message --\n{system_message}\n-- User Message ---\n{prompt}\n-- Generated Message --\n{answer}"
            if reasoning:
                round_print += "\n-- Reasoning --\n" + str(reasoning)

            if i < len(response_generation_method):
                current_method = response_generation_method[i]
                if isinstance(current_method, LogprobResponseGenerationMethod):
                    round_print += "\n-- Logprobs --\n" + str(logprob_answer)
            tqdm.write(round_print)


# TODO Structured output for API calls
def batch_generation(
    model: Union[LLM, AsyncOpenAI],
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[str] = ["Hi there! What is your name?"],
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    number_of_printed_conversations: int = 2,
    print_progress: bool = True,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    space_char: str = "Ġ",
    **generation_kwargs: Any,
) -> Tuple:
    """
    Generate responses for a batch of prompts.

    Handles both vLLM and OpenAI API generation with support for:
    - Structured output (JSON or choice format)
    - Conversation printing
    - Progress tracking
    - Concurrent API requests

    Args:
        model (LLM or AsyncOpenAI): vLLM model or AsyncOpenAI client.
        system_messages (List(str)): System prompts for each conversation.
        prompts (List(str)): User prompts to generate responses for.
        response_generation_method (ResponseGenerationMethod or List(ResponseGenerationMethod), optional): Configuration for structured output.
        seed (int): Random seed for reproducibility, defaults to 42.
        client_model_name (str, optional): Model name when using OpenAI API.
        api_concurrency (int): Max concurrent API requests when using OpenAI API.
        number_if_printed_conversations (int): How many conversations should be printed. Defaults to 2.
        print_conversation (bool): If True, prints conversations. Defaults to False.
        print_progress (bool): If True, shows progress bar. Defaults to True.
        reasoning_start_token (str): Special token at the beginning of reasoning models' output. Will be used for manual parsing, if automatic parsing is not possible.
        reasoning_end_token (str): Special token to separate reasoning from regular model output. Will be used for manual parsing, if automatic parsing is not possible.
        space_token (str): Special char to encode spaces in tokens ("Ġ" for most byte-pair tokenizers).
        generation_kwargs: Additional generation parameters

    Returns:
        Tuple[List[str], List[str], List[str]]: Generated Response, Logprobs, Reasoning
    """

    model_type = type(model).__name__
    if model_type == "LLM" and not has_vllm:
        raise ImportError(
            "You are trying to use a vLLM model, but 'vllm' is not installed."
        )
    elif model_type == "AsyncOpenAI" and not has_openai:
        raise ImportError(
            "You are trying to use OpenAI, but 'openai' is not installed."
        )
    elif model_type != "LLM" and model_type != "AsyncOpenAI":
        raise ValueError(f"Unsupported model type: {type(model)}")
    random.seed(seed)

    # Inference
    if has_vllm and isinstance(model, LLM):
        plain_results, logprob_result, reasoning_outputs = run_vllm_batch(
            model,
            system_messages=system_messages,
            prompts=prompts,
            response_generation_method=response_generation_method,
            seed=seed,
            print_progress=print_progress,
            reasoning_start_token=reasoning_start_token,
            reasoning_end_token=reasoning_end_token,
            space_char=space_char,
            **generation_kwargs,
        )
    elif has_openai and isinstance(model, AsyncOpenAI):
        plain_results, logprob_result, reasoning_outputs = run_openai_batch(
            model,
            system_messages=system_messages,
            prompts=prompts,
            response_generation_method=response_generation_method,
            seed=seed,
            print_progress=print_progress,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            reasoning_start_token=reasoning_start_token,
            reasoning_end_token=reasoning_end_token,
            **generation_kwargs,
        )

    if print_conversation:
        _print_conversation(
            system_messages=system_messages,
            prompts=prompts,
            assistant_messages=None,
            plain_results=plain_results,
            reasoning_output=reasoning_outputs,
            logprob_result=logprob_result,
            response_generation_method=response_generation_method,
            number_of_printed_conversations=number_of_printed_conversations,
        )

    return (plain_results, logprob_result, reasoning_outputs)


def batch_turn_by_turn_generation(
    model: LLM,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[List[str]] = [["Hi there! What is your name?", "Interesting"]],
    assistant_messages: Optional[List[List[str]]] = None,
    response_generation_method: Optional[
        Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
    ] = None,
    seed: int = 42,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    number_of_printed_conversations: int = 2,
    print_progress: bool = True,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    space_char: str = "Ġ",
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
        model (LLM or AsyncOpenAI): vLLM model or AsyncOpenAI client.
        system_messages (List(str)): System prompts for each conversation.
        prompts (List(List(str))): User prompts to generate responses for. Can be multiple request per system prompt.
        assistant_messages (List(List(str)), optional): These are prefilled responses for the assistant. E.g. if the first list contains one entry, the first LLM assistant will be prefilled in the chat format and not be inferenced.
        response_generation_method (ResponseGenerationMethod or List(ResponseGenerationMethod), optional): Configuration for structured output.
        seed (int): Random seed for reproducibility, defaults to 42.
        client_model_name (str, optional): Model name when using OpenAI API.
        api_concurrency (int): Max concurrent API requests when using OpenAI API.
        print_conversation (bool): If True, prints conversations. Defaults to False.
        number_of_printed_conversations (int): How many conversations should be printed. Defaults to 2.
        print_progress (bool): If True, shows progress bar. Defaults to True.
        reasoning_start_token (str): Special token at the beginning of reasoning models' output. Will be used for manual parsing, if automatic parsing is not possible.
        reasoning_end_token (str): Special token to separate reasoning from regular model output. Will be used for manual parsing, if automatic parsing is not possible.
        space_token (str): Special char to encode spaces in tokens ("Ġ" for most byte-pair tokenizers).
        generation_kwargs: Additional generation parameters.

    Returns:
        Tuple[List[str], List[str], List[str]]: Generated Response, Logprobs, Reasoning
    """

    model_type = type(model).__name__
    if model_type == "LLM" and not has_vllm:
        raise ImportError(
            "You are trying to use a vLLM model, but 'vllm' is not installed."
        )
    elif model_type == "AsyncOpenAI" and not has_openai:
        raise ImportError(
            "You are trying to use OpenAI, but 'openai' is not installed."
        )
    elif model_type != "LLM" and model_type != "AsyncOpenAI":
        raise ValueError(f"Unsupported model type: {type(model)}")
    random.seed(seed)

    # Inference
    if has_vllm and isinstance(model, LLM):
        plain_results, logprob_result, reasoning_outputs = run_vllm_batch_conversation(
            model,
            system_messages=system_messages,
            prompts=prompts,
            assistant_messages=assistant_messages,
            response_generation_method=response_generation_method,
            seed=seed,
            print_progress=print_progress,
            reasoning_start_token=reasoning_start_token,
            reasoning_end_token=reasoning_end_token,
            space_char=space_char,
            **generation_kwargs,
        )
    elif has_openai and isinstance(model, AsyncOpenAI):
        plain_results, logprob_result, reasoning_outputs = (
            run_openai_batch_conversation(
                model,
                system_messages=system_messages,
                prompts=prompts,
                assistant_messages=assistant_messages,
                response_generation_method=response_generation_method,
                client_model_name=client_model_name,
                seed=seed,
                print_progress=print_progress,
                api_concurrency=api_concurrency,
                **generation_kwargs,
            )
        )

    if print_conversation:
        _print_conversation(
            system_messages=system_messages,
            prompts=prompts,
            assistant_messages=assistant_messages,
            plain_results=plain_results,
            reasoning_output=reasoning_outputs,
            logprob_result=logprob_result,
            response_generation_method=response_generation_method,
            number_of_printed_conversations=number_of_printed_conversations,
        )

    return (plain_results, logprob_result, reasoning_outputs)


# def batch_decoding(
#     model: Union[LLM, AsyncOpenAI],
#     prompts: List[str] = ["Hi there! What is your name?"],
#     stop_tokens: List[str] = ["\nA:"],
#     structured_output_options: Optional[
#         Union[ResponseGenerationMethod, List[ResponseGenerationMethod]]
#     ] = None,
#     seed: int = 42,
#     client_model_name: Optional[str] = None,
#     api_concurrency: int = 10,
#     print_conversation: bool = False,
#     print_progress: bool = True,
#     **generation_kwargs: Any,
# ):
#     """
#     Generate responses for a batch of prompts.

#     Handles both vLLM and OpenAI API generation with support for:
#     - Structured output (JSON or choice format)
#     - Conversation printing
#     - Progress tracking
#     - Concurrent API requests

#     Args:
#         model: vLLM model or AsyncOpenAI client
#         system_messages: System prompts for each conversation
#         prompts: User prompts to generate responses for
#         structured_output_options: Configuration for structured output
#         seed: Random seed for reproducibility
#         client_model_name: Model name when using OpenAI API
#         api_concurrency: Max concurrent API requests
#         print_conversation: If True, prints conversations
#         print_progress: If True, shows progress bar
#         **generation_kwargs: Additional generation parameters

#     Returns:
#         List[str]: Generated responses
#     """
#     random.seed(seed)

#     batch_size: int = len(prompts)

#     seeds = _generate_seeds(seed, batch_size=batch_size)

#     if isinstance(model, LLM):
#         sampling_params_list = _create_sampling_params(
#             batch_size=batch_size,
#             seeds=seeds,
#             structured_output_options=structured_output_options,
#             stop_tokens=stop_tokens,
#             **generation_kwargs,
#         )
#         outputs: List[RequestOutput] = model.generate(
#             prompts,
#             sampling_params=sampling_params_list,
#             use_tqdm=print_progress,
#         )
#         result = [output.outputs[0].text for output in outputs]

#     else:
#         result = _run_async_in_thread(
#             client=model,
#             client_model_name=client_model_name,
#             batch_messages=prompts,
#             seeds=seeds,
#             concurrency_limit=api_concurrency,
#             structured_output_options=structured_output_options,
#             **generation_kwargs,
#         )

#     # TODO add argurment to specify how many conversations should be printed (base argument should be reasonable)
#     if print_conversation:
#         conversation_print = "Conversation:"
#         for prompt, answer in zip(prompts, result):
#             round_print = f"{conversation_print}\nUser Message:\n{prompt}\nGenerated Message\n{answer}"
#             print(round_print, flush=True)
#             break

#     return result
