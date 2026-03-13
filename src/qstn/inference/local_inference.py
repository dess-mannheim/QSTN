import random
from collections.abc import Sequence
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]
from vllm.outputs import RequestOutput  # pyright: ignore[reportMissingImports]
from vllm.sampling_params import StructuredOutputsParams  # pyright: ignore[reportMissingImports]

from ..utilities.utils import _make_cache_key, generate_seeds
from .dynamic_pydantic import build_pydantic_model_from_json_object
from .reasoning_parser import parse_reasoning
from .response_generation import (
    ChoiceResponseGenerationMethod,
    JSONResponseGenerationMethod,
    LogprobResponseGenerationMethod,
    ResponseGenerationMethod,
)
from .utils import normalize_system_messages


def _run_vllm_chat_pipeline(
    model: LLM,
    batch_messages: list[list[dict[str, str]]],
    response_generation_method: ResponseGenerationMethod | list[ResponseGenerationMethod] | None,
    seed: int,
    print_progress: bool,
    reasoning_start_token: str,
    reasoning_end_token: str,
    space_char: str,
    **generation_kwargs: Any,
) -> tuple[list[str], list[str], list[str]]:
    """Run the shared vLLM chat pipeline for single and conversation batching."""
    batch_size = len(batch_messages)
    seeds = generate_seeds(seed, batch_size=batch_size)

    logprob_result = None
    logprob_config = _update_logprob_kwargs(response_generation_method, generation_kwargs)

    # If users specify use_tqdm themselves, we use that flag instead.
    print_progress = generation_kwargs.pop("use_tqdm", print_progress)

    if "sampling_params" in generation_kwargs.keys():
        import warnings

        warnings.warn(
            "Do not specify sampling_params for vllm inference. "
            "If you want to use hyperparameters, add them directly to the "
            "generation kwargs. Given argument sampling_params will be ignored.",
            stacklevel=2,
        )
        generation_kwargs.pop("sampling_params")

    gen_kwargs, chat_kwargs = _split_kwargs(generation_kwargs)

    sampling_params_list = _create_sampling_params(
        batch_size=batch_size,
        seeds=seeds,
        response_generation_method=response_generation_method,
        **gen_kwargs,
    )

    outputs: list[RequestOutput] = model.chat(
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


def run_vllm_batch(
    model: LLM,
    system_messages: Sequence[str | None] | None = ("You are a helpful assistant.",),
    prompts: Sequence[str] = ("Hi there! What is your name?",),
    response_generation_method: (
        ResponseGenerationMethod | list[ResponseGenerationMethod] | None
    ) = None,
    seed: int = 42,
    # number_of_printed_conversation: int = 2,
    print_progress: bool = True,
    # <think>...</think> tokens are used by Qwen3 to separate reasoning
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    space_char: str = "Ġ",
    **generation_kwargs: Any,
) -> tuple[list[str], list[str], list[str]]:
    normalized_system_messages = normalize_system_messages(
        system_messages=system_messages,
        batch_size=len(prompts),
    )

    # Prepare batch of messages
    batch_messages: list[list[dict[str, str]]] = []
    for system_message, prompt in zip(normalized_system_messages, prompts):
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if system_message is not None:
            messages.insert(0, {"role": "system", "content": system_message})
        batch_messages.append(messages)

    return _run_vllm_chat_pipeline(
        model=model,
        batch_messages=batch_messages,
        response_generation_method=response_generation_method,
        seed=seed,
        print_progress=print_progress,
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
        space_char=space_char,
        **generation_kwargs,
    )


def run_vllm_batch_conversation(
    model: LLM,
    system_messages: Sequence[str | None] | None = ("You are a helpful assistant.",),
    prompts: Sequence[Sequence[str]] = (("Hi there! What is your name?",),),
    assistant_messages: Sequence[Sequence[str]] = (),
    response_generation_method: (
        ResponseGenerationMethod | list[ResponseGenerationMethod] | None
    ) = None,
    seed: int = 42,
    # number_of_printed_conversation: int = 2,
    print_progress: bool = True,
    # <think>...</think> tokens are used by Qwen3 to separate reasoning
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    space_char: str = "Ġ",
    **generation_kwargs: Any,
) -> tuple[list[str], list[str], list[str]]:
    normalized_system_messages = normalize_system_messages(
        system_messages=system_messages,
        batch_size=len(prompts),
    )

    batch_messages = []
    batch_size = len(normalized_system_messages)
    if not assistant_messages:
        assistant_messages = tuple(() for _ in range(batch_size))
    for i in range(batch_size):
        messages = []

        # Add system message
        if normalized_system_messages[i] is not None:
            messages.append({"role": "system", "content": normalized_system_messages[i]})

        num_user_msgs = len(prompts[i])
        num_assistant_msgs = len(assistant_messages[i])

        for j in range(num_user_msgs):
            messages.append({"role": "user", "content": prompts[i][j]})
            if j < num_assistant_msgs:
                messages.append({"role": "assistant", "content": assistant_messages[i][j]})

        batch_messages.append(messages)

    return _run_vllm_chat_pipeline(
        model=model,
        batch_messages=batch_messages,
        response_generation_method=response_generation_method,
        seed=seed,
        print_progress=print_progress,
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
        space_char=space_char,
        **generation_kwargs,
    )


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


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
    seeds: list[int],
    response_generation_method: ResponseGenerationMethod | list[ResponseGenerationMethod],
    **generation_kwargs: Any,
) -> list[SamplingParams]:

    structured_output = []

    # Same for all calls
    if isinstance(response_generation_method, ResponseGenerationMethod):
        if isinstance(response_generation_method, JSONResponseGenerationMethod):
            pydantic_model = build_pydantic_model_from_json_object(
                json_object=response_generation_method.json_object,
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
            _allowed_choices = [str(c) for c in response_generation_method.allowed_choices]
            global_structured_output = StructuredOutputsParams(choice=_allowed_choices)
            structured_output = [global_structured_output] * batch_size
            # Remote Inference
            # else:
            # structured_output = [_allowed_choices] * batch_size

    # Different response generation methods for each question
    else:
        structured_output = []
        cache: dict[str, StructuredOutputsParams] = {}
        for i in range(batch_size):
            current_method = response_generation_method[i]
            if isinstance(current_method, JSONResponseGenerationMethod):
                key = _make_cache_key(current_method.get_json_prompt(), None)

                if key not in cache:
                    pydantic_model = build_pydantic_model_from_json_object(
                        json_object=current_method.json_object,
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
            SamplingParams(seed=seeds[i], **generation_kwargs) for i in range(batch_size)
        ]
    # Remote Inference
    # else:
    #     return structured_output

    return sampling_params_list


def _create_sampling_params(
    batch_size: int,
    seeds: list[int],
    response_generation_method: ResponseGenerationMethod | list[ResponseGenerationMethod] | None,
    **generation_kwargs: Any,
) -> list[SamplingParams]:
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

    return [SamplingParams(seed=seeds[i], **generation_kwargs) for i in range(batch_size)]


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
    rgms: list[LogprobResponseGenerationMethod] = []
    if isinstance(response_generation_method, LogprobResponseGenerationMethod):
        rgms.append(response_generation_method)
    elif isinstance(response_generation_method, list):
        rgms = [
            rgm
            for rgm in response_generation_method
            if isinstance(rgm, LogprobResponseGenerationMethod)
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
                # Strip space token and any leading whitespace from tokenization.
                answer_dict = {
                    x.decoded_token.lstrip(space_char).lstrip(): x.logprob
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
    #                         ).lstrip(): x.logprob
    #                         # strip the space character and whitespace from
    #                         # tokenization
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
    reasoning_start_token: str, reasoning_end_token: str, outputs: list[RequestOutput]
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
        if extracted_reasoning is not None:
            reasoning_output.append(extracted_reasoning.strip())
        else:
            reasoning_output.append(extracted_reasoning)
        plain_results.append(final_answer)

    return raw_reasonings, reasoning_output, plain_results
