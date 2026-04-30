"""Unit tests for remote inference helper functions and loop-runner behavior."""

import asyncio
from concurrent.futures import Future

import pytest

from qstn.inference import remote_inference
from qstn.inference.response_generation import (
    ChoiceResponseGenerationMethod,
    JSONItem,
    JSONObject,
    JSONResponseGenerationMethod,
    JSONSingleResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)
from qstn.utilities import survey_objects


@pytest.fixture(autouse=True)
def cleanup_loop_runners():
    """Ensure background loop runners do not leak across tests."""
    remote_inference._shutdown_all_client_loop_runners()
    yield
    remote_inference._shutdown_all_client_loop_runners()


def test_update_logprob_kwargs_remote():
    rgm = LogprobResponseGenerationMethod(token_limit=2)
    kw = {}
    out = remote_inference._update_logprob_kwargs(rgm, kw)
    assert out is rgm
    assert kw["logprobs"] is True
    assert kw["top_logprobs"] == rgm.top_logprobs

    # list scenario
    out2 = remote_inference._update_logprob_kwargs([rgm], {})
    assert out2 is rgm


def test_create_structured_output_and_params():
    # no structured input
    assert (
        remote_inference._create_structured_output(batch_size=1, response_generation_method=None)
        is None
    )

    # single JSON method
    rgm = JSONResponseGenerationMethod(json_object=JSONObject(children=[JSONItem("a")]))
    out = remote_inference._create_structured_output(batch_size=2, response_generation_method=rgm)
    assert isinstance(out, list) and len(out) == 2

    # list with JSON and choice
    rgm_list = [rgm, rgm]
    params = remote_inference._create_structured_params(
        batch_size=2, response_generation_method=rgm_list
    )
    assert params[0] == params[1]
    choice = ChoiceResponseGenerationMethod(allowed_choices=["x"])
    params2 = remote_inference._create_structured_params(
        batch_size=1, response_generation_method=choice
    )
    assert params2 == [["x"]]


def test_create_structured_params_includes_auto_answer_enum():
    """Remote JSON schemas should include automatic answer-option enums."""
    answer_texts = survey_objects.AnswerTexts(answer_texts=["a", "b"], indices=["1", "2"])
    options = survey_objects.AnswerOptions(
        answer_texts=answer_texts,
        response_generation_method=JSONSingleResponseGenerationMethod(),
    )

    params = remote_inference._create_structured_params(
        batch_size=1,
        response_generation_method=options.response_generation_method,
    )

    definitions = params[0].get("$defs", {})
    enum_values = [
        definition["enum"]
        for definition in definitions.values()
        if definition.get("enum") == ["1: a", "2: b"]
    ]
    assert enum_values == [["1: a", "2: b"]]


def test_client_loop_runner_executes_coroutine():
    """Loop runner should execute submitted coroutines on its background loop."""
    runner = remote_inference._ClientLoopRunner()

    async def _sample():
        return "ok"

    try:
        future = runner.submit(_sample())
        assert future.result(timeout=2) == "ok"
    finally:
        runner.shutdown()


def test_get_or_create_runner_reuses_runner_for_same_client():
    class DummyClient:
        pass

    client = DummyClient()
    runner_a = remote_inference._get_or_create_runner(client)
    runner_b = remote_inference._get_or_create_runner(client)

    assert runner_a is runner_b


def test_get_or_create_runner_creates_distinct_runners_for_distinct_clients():
    class DummyClient:
        pass

    client_a = DummyClient()
    client_b = DummyClient()
    runner_a = remote_inference._get_or_create_runner(client_a)
    runner_b = remote_inference._get_or_create_runner(client_b)

    assert runner_a is not runner_b


def test_run_async_in_thread_success(monkeypatch):
    """Wrapper should return the submitted coroutine result."""
    expected = (["r"], None, None)

    class DummyRunner:
        def submit(self, coro):
            future = Future()
            coro.close()
            future.set_result(expected)
            return future

    monkeypatch.setattr(remote_inference, "_get_or_create_runner", lambda _: DummyRunner())
    result = remote_inference._run_async_in_thread(
        client=None,
        client_model_name="m",
        batch_messages=[],
        seeds=[],
    )
    assert result == expected


def test_run_async_in_thread_error(monkeypatch):
    """Wrapper should surface errors raised by the submitted future."""

    class DummyError(Exception):
        pass

    class DummyRunner:
        def submit(self, coro):
            future = Future()
            coro.close()
            future.set_exception(DummyError("fail"))
            return future

    monkeypatch.setattr(remote_inference, "_get_or_create_runner", lambda _: DummyRunner())
    with pytest.raises(DummyError):
        remote_inference._run_async_in_thread(
            client=None,
            client_model_name="m",
            batch_messages=[],
            seeds=[],
        )


def test_api_batch_progress_label(monkeypatch):
    """Remote progress should use the clearer tqdm description."""
    captured: dict[str, str] = {}

    async def fake_gather(*tasks, total, desc):
        captured["desc"] = desc
        captured["total"] = total
        for task in tasks:
            task.close()
        return []

    monkeypatch.setattr(remote_inference.tqdm_asyncio, "gather", fake_gather)

    output, logprobs, reasoning = asyncio.run(
        remote_inference._run_api_batch_async(
            client=None,
            client_model_name="m",
            batch_messages=[[{"role": "user", "content": "p"}]],
            seeds=[1],
            print_progress=True,
        )
    )

    assert captured == {"desc": "Generating responses", "total": 1}
    assert output == []
    assert logprobs == []
    assert reasoning == []


def test_run_openai_batch_conversation_reuses_same_loop_for_same_client():
    """Two calls with the same client should not hit cross-loop reuse errors."""

    class LoopBoundCompletions:
        def __init__(self):
            self.loop = None

        async def create(self, **kwargs):
            current_loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = current_loop
            elif self.loop is not current_loop:
                raise RuntimeError("cross-loop reuse detected")

            class Msg:
                content = "ok"
                reasoning = None
                reasoning_content = None

            class Choice:
                message = Msg()
                logprobs = None

            class Response:
                choices = [Choice()]

            return Response()

    class LoopBoundClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": LoopBoundCompletions()})()

    client = LoopBoundClient()
    first_output, _, _ = remote_inference.run_openai_batch_conversation(
        model=client,
        system_messages=["s"] * 5,
        prompts=[["u"]] * 5,
        assistant_messages=[[] for _ in range(5)],
        client_model_name="m",
        api_concurrency=10,
        print_progress=False,
    )
    second_output, _, _ = remote_inference.run_openai_batch_conversation(
        model=client,
        system_messages=["s"],
        prompts=[["u"]],
        assistant_messages=[[]],
        client_model_name="m",
        api_concurrency=10,
        print_progress=False,
    )

    assert first_output == ["ok"] * 5
    assert second_output == ["ok"]
