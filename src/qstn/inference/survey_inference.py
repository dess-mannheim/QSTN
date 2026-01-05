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
    
    if assistant_messages:
        conversation_print = "--- Conversation ---"
        for i, (system_message, prompt_list, answer, reasoning, logprob_answer, assistant_list) in enumerate(
            zip(system_messages, prompts, plain_results, reasoning_output, logprob_result, assistant_messages)
        ):
            if i >= number_of_printed_conversations:
                break

            round_print = f"{conversation_print}\n-- System Message --\n{system_message}"
            for j in range(len(prompt_list)):
                round_print = f"{round_print}\n-- User Message --\n{prompt_list[j]}"
                if j < len(assistant_list):
                    prefill = assistant_list[j]
                    if prefill:
                        round_print = (
                            f"{round_print}\n-- Assistant Message --\n{assistant_list[j]}"
                        )
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
            zip(system_messages, prompts, plain_results, reasoning_output, logprob_result)
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
        model: vLLM model or AsyncOpenAI client
        system_messages: System prompts for each conversation
        prompts: User prompts to generate responses for
        answer_production_method: Configuration for structured output
        seed: Random seed for reproducibility
        client_model_name: Model name when using OpenAI API
        api_concurrency: Max concurrent API requests
        print_conversation: If True, prints conversations
        print_progress: If True, shows progress bar
        reasoning_start_token: Special token at the beginning of reasoning models' output
        reasoning_end_token: Special token to separate reasoning from regular model output
        space_token: Special char to encode spaces in tokens ("Ġ" for most byte-pair tokenizers)
        **generation_kwargs: Additional generation parameters

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
            **generation_kwargs,
        )

    if print_conversation:
        _print_conversation(
            system_messages=system_messages,
            prompts=prompts,
            plain_results=plain_results,
            reasoning_output=reasoning_outputs,
            logprob_result=logprob_result,
            response_generation_method=response_generation_method,
            number_of_printed_conversations=number_of_printed_conversations,
        )

    return (plain_results, logprob_result, reasoning_outputs)

    batch_size: int = len(system_messages)

    seeds = _generate_seeds(seed, batch_size=batch_size)

    logprob_result = None

    if isinstance(model, LLM):
        # TODO: add support for List[AnswerProductionMethod]
        if isinstance(response_generation_method, LogprobResponseGenerationMethod):
            generation_kwargs["logprobs"] = response_generation_method.top_logprobs
            # set token limit separately to support reasoning models
            if response_generation_method.token_limit is not None:
                generation_kwargs["max_tokens"] = response_generation_method.token_limit

        sampling_params_list = _create_sampling_params(
            batch_size=batch_size,
            seeds=seeds,
            response_generation_method=response_generation_method,
            **generation_kwargs,
        )
        outputs: List[RequestOutput] = model.chat(
            batch_messages,
            sampling_params=sampling_params_list,
            use_tqdm=print_progress,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        # result = [output.outputs[0].text for output in outputs]

        # separate out reasoning
        # we parse this here because the OpenAI API separates it automatically
        plain_results = []
        reasoning_output = []
        raw_reasonings = []  # keep the whitespace for length calculations

        print(outputs)
        for output in outputs:
            out_obj = output.outputs[0]

            # 1. ATTEMPT NATIVE EXTRACTION (vLLM 2025 Standard)
            # vLLM now often populates 'reasoning_text' or 'reasoning'
            native_reasoning = getattr(out_obj, "reasoning_text", None) or getattr(
                out_obj, "reasoning", None
            )
            final_answer = out_obj.text
            print(out_obj)
            print(native_reasoning)
            if not native_reasoning:
                # We could support more patterns here
                patterns = [
                    (reasoning_start_token, reasoning_end_token),
                ]

                for start, end in patterns:
                    match = re.search(f"{start}(.*?){end}", final_answer, re.DOTALL)
                    if match:
                        native_reasoning = match.group(1)
                        # clean final answer
                        final_answer = final_answer.split(end)[-1].strip()
                        break

            raw_reasonings.append(native_reasoning if native_reasoning else None)

            reasoning_output.append(
                native_reasoning.strip() if native_reasoning else None
            )

            plain_results.append(final_answer.strip())

        if response_generation_method:
            for rgm in response_generation_method:
                # TODO This is not implemented correcty yet
                if isinstance(rgm, LogprobResponseGenerationMethod):
                    logprob_result = []
                    # ignore the first k tokens that belong to the reasoning
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
                                for x in req_output.outputs[0]
                                .logprobs[logprob_position]
                                .values()
                            }
                        except (
                            IndexError
                        ):  # less than [logprob_position] tokens in the output!
                            answer_dict = {}
                        logprob_result.append(answer_dict)

        # print the first result returned from vllm
        # if print_conversation:
        #    for req_output in outputs:
        #        print(req_output.outputs[0])
        #        break

    else:
        # TODO: add support for List[AnswerProductionMethod]
        if isinstance(response_generation_method[0], LogprobResponseGenerationMethod):
            raise NotImplementedError(
                "The Logprob_AnswerProductionMethod is not yet implemented "
                + "for use with the OpenAI API. Use vllm offline inference instead."
            )
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = response_generation_method.top_logprobs

        plain_results = _run_async_in_thread(
            client=model,
            client_model_name=client_model_name,
            batch_messages=batch_messages,
            seeds=seeds,
            concurrency_limit=api_concurrency,
            response_generation_method=response_generation_method,
            **generation_kwargs,
        )

        # TODO: handle reasoning
        reasoning_output = [None] * len(plain_results)

    if logprob_result is None:
        logprob_result = [None] * len(plain_results)

    # TODO add argument to specify how many conversations should be printed (base argument should be reasonable)
    if print_conversation:
        conversation_print = "--- Conversation ---"
        for system_message, prompt, answer, reasoning, logprob_answer in zip(
            system_messages, prompts, plain_results, reasoning_output, logprob_result
        ):
            round_print = f"{conversation_print}\n-- System Message --\n{system_message}\n-- User Message ---\n{prompt}\n-- Generated Message --\n{answer}"
            if reasoning is not None:
                round_print += "\n-- Reasoning --\n" + str(reasoning)
            if isinstance(response_generation_method, LogprobResponseGenerationMethod):
                round_print += "\n-- Logprobs --\n" + str(logprob_answer)
            tqdm.write(round_print)

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

    return (plain_results, logprob_result, reasoning_output)


def batch_turn_by_turn_generation(
    model: LLM,
    system_messages: List[str] = ["You are a helpful assistant."],
    prompts: List[List[str]] = [["Hi there! What is your name?", "Interesting"]],
    assistant_messages: List[List[str]] = None,
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
        Tuple[List[str], List[str], List[str]]: Generated Responses, Logprobs, Reasonings
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
        plain_results, logprob_result, reasoning_outputs = run_openai_batch_conversation(
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
            response_generation_method=response_generation_method,
            **generation_kwargs,
        )
        outputs: List[RequestOutput] = model.chat(
            batch_messages,
            sampling_params=sampling_params_list,
            use_tqdm=print_progress,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        # TODO: add support for logprobs
        result = [output.outputs[0].text for output in outputs]

        # separate out reasoning
        # we parse this here because the OpenAI API separates it automatically
        plain_results = []
        reasoning_output = []
        raw_reasonings = []  # keep the whitespace for length calculations
        logprob_result = []
        for output_text in result:
            reasoning_match = re.search(
                reasoning_start_token + r"(.*?)" + reasoning_end_token,
                output_text,
                re.DOTALL,
            )
            raw_reasonings.append(reasoning_match.group(1) if reasoning_match else None)
            reasoning_output.append(
                reasoning_match.group(1).strip() if reasoning_match else None
            )
            plain_results.append(output_text.split(reasoning_end_token)[-1].strip())

        if isinstance(response_generation_method, LogprobResponseGenerationMethod):
            # ignore the first k tokens that belong to the reasoning
            if response_generation_method.ignore_reasoning:
                tokenizer = model.get_tokenizer()
                logprob_positions = [
                    (
                        len(
                            tokenizer.tokenize(
                                f"{reasoning_start_token}{_reasoning}{reasoning_end_token}"
                            )
                        )
                        + 1
                        + response_generation_method.token_position
                        if _reasoning is not None
                        else response_generation_method.token_position
                    )
                    for _reasoning in raw_reasonings
                ]
            else:
                logprob_positions = [response_generation_method.token_position] * len(
                    outputs
                )

            for req_output, logprob_position in zip(outputs, logprob_positions):
                try:
                    answer_dict = {
                        x.decoded_token.lstrip(
                            space_char
                        ).lstrip(): x.logprob  # strip the space character and whitespace from tokenization
                        for x in req_output.outputs[0]
                        .logprobs[logprob_position]
                        .values()
                    }
                except IndexError:  # less than [logprob_position] tokens in the output!
                    answer_dict = {}
                logprob_result.append(answer_dict)

    else:
        plain_results = _run_async_in_thread(
            client=model,
            client_model_name=client_model_name,
            batch_messages=batch_messages,
            seeds=seeds,
            concurrency_limit=api_concurrency,
            response_generation_method=response_generation_method,
            print_progress=print_progress,
            **generation_kwargs,
        )

        # TODO: handle reasoning and logprobs
        reasoning_output = [None] * len(plain_results)
        logprob_result = [None] * len(plain_results)

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

    return (plain_results, logprob_result, reasoning_output)


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
