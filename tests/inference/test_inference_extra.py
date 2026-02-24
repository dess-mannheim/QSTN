"""Additional integration-style inference tests.

These tests exercise cross-module behavior that is easy to regress:
- logprob extraction logic,
- default local model initialization wiring,
- async API response parsing,
- high-level survey inference delegation.
"""

import asyncio
from types import SimpleNamespace

from qstn.inference import local_inference, remote_inference, survey_inference
from qstn.inference.response_generation import LogprobResponseGenerationMethod


class DummyLogprobModel:
    """Minimal model exposing tokenizer API needed by `_get_logprobs`."""

    def __init__(self):
        class Tokenizer:
            def tokenize(self, txt):
                return txt.split()
        self._tokenizer = Tokenizer()

    def get_tokenizer(self):
        return self._tokenizer


def make_output(text, logprobs=None):
    """Create a minimal output object matching vLLM request output shape."""
    class Out:
        def __init__(self, text, logprobs):
            self.text = text
            self.logprobs = logprobs
    class Req:
        def __init__(self, text, logprobs):
            self.outputs = [Out(text, logprobs)]
    return Req(text, logprobs)


def test_get_logprobs_various():
    """`_get_logprobs` should support both raw and reasoning-offset token positions."""
    model = DummyLogprobModel()

    # prepare fake outputs with two entries
    out1 = make_output("ignored", logprobs=[SimpleNamespace(values=lambda: [SimpleNamespace(decoded_token=' a', logprob=-1.0)])])
    out2 = make_output("ignored2", logprobs=[SimpleNamespace(values=lambda: [SimpleNamespace(decoded_token=' b', logprob=-2.0)])])
    outputs = [out1, out2]

    # Baseline behavior without reasoning-aware offset.
    rgm = LogprobResponseGenerationMethod(token_position=0, ignore_reasoning=False)
    result = local_inference._get_logprobs(model, rgm, "<t>","</t>"," ",outputs, [None, None])
    assert result == [{"a": -1.0}, {"b": -2.0}]

    # Reasoning-aware behavior can point past available token positions.
    # In that case, the helper should gracefully return empty dicts.
    rgm2 = LogprobResponseGenerationMethod(token_position=0, ignore_reasoning=True)
    # Provide reasoning text so tokenizer-based offset logic is exercised.
    result2 = local_inference._get_logprobs(model, rgm2, "<t>","</t>"," ",outputs, ["R","S"])
    assert result2 == [{}, {}]


def test_default_model_init_monkeypatched(monkeypatch):
    """`default_model_init` should pass expected constructor kwargs to LLM."""
    calls = {}
    class DummyLLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)
    monkeypatch.setattr(local_inference, "LLM", DummyLLM)
    # Patch torch internals to keep test fully local and deterministic.
    monkeypatch.setattr(local_inference.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(local_inference.torch, "cuda", SimpleNamespace(device_count=lambda: 2, _is_in_bad_fork=lambda : False))

    local_inference.default_model_init("foo-model", seed=123, extra=7)
    assert calls.get("model") == "foo-model"
    assert calls.get("seed") == 123
    assert calls.get("tensor_parallel_size") == 2
    assert calls.get("extra") == 7


def test_run_api_batch_async_processing(monkeypatch):
    """Async OpenAI helper should parse reasoning and capture top logprobs."""
    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        async def _create(self, **kwargs):
            # Return a minimal object with OpenAI-like response structure.
            msg = SimpleNamespace(content=kwargs.get("messages")[1]["content"], reasoning=None, reasoning_content=None)
            lp = SimpleNamespace(top_logprobs=[SimpleNamespace(token="tok", logprob=-0.5)])
            resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, logprobs=SimpleNamespace(content=[lp]))])
            return resp

    fake = FakeClient()
    res = asyncio.run(
        remote_inference._run_api_batch_async(
            client=fake,
            client_model_name="m",
            batch_messages=[[{"role":"system","content":"s"},{"role":"user","content":"<think>R</think>ans"}]],
            seeds=[1],
            concurrency_limit=1,
            sampling_params=None,
            response_generation_method=None,
            logprob_config=LogprobResponseGenerationMethod(),
        )
    )
    # Final result tuple order: answers, logprobs, reasoning.
    assert res[0][0] == "ans"
    assert res[2][0] == "R"
    assert res[1][0][0][0]["token"] == "tok"


def test_survey_inference_batch_and_conversation(monkeypatch, capsys):
    """Ensure high-level batch functions delegate correctly and optionally print."""
    # Stub local inference entry points to avoid external model dependencies.
    def fake_vllm(model, **kwargs):
        # Produce one result per prompt to mimic batch behavior.
        batch_size = len(kwargs.get("system_messages", []))
        return (["ok"] * batch_size, [None] * batch_size, [None] * batch_size)
    # Patch both local module and survey_inference bindings.
    monkeypatch.setattr(local_inference, "run_vllm_batch", fake_vllm)
    monkeypatch.setattr(local_inference, "run_vllm_batch_conversation", fake_vllm)
    monkeypatch.setattr(survey_inference, "run_vllm_batch", fake_vllm)
    monkeypatch.setattr(survey_inference, "run_vllm_batch_conversation", fake_vllm)

    # Create a dummy class whose type name matches survey_inference checks.
    FakeLLM = type("LLM", (), {})
    # Ensure code path takes the vLLM branch.
    monkeypatch.setattr(survey_inference, "HAS_VLLM", True)
    monkeypatch.setattr(survey_inference, "LLM", FakeLLM)
    # Batch generation with multiple prompts.
    out = survey_inference.batch_generation(
        model=FakeLLM(),
        system_messages=["s1","s2"],
        prompts=["p1","p2"],
        response_generation_method=[None, None],
        print_conversation=True,
        number_of_printed_conversations=2,
    )
    assert out[0] == ["ok","ok"]
    captured = capsys.readouterr().out
    assert "-- System Message --" in captured

    # Turn-by-turn generation with prefilled assistant messages.
    out2 = survey_inference.batch_turn_by_turn_generation(
        model=FakeLLM(),
        system_messages=["s"],
        prompts=[["p1","p2"]],
        assistant_messages=[["a"]],
        response_generation_method=[None],
        print_conversation=True,
    )
    assert len(out2) == 3
    assert out2[0] == ["ok"]
    captured2 = capsys.readouterr().out
    assert "-- Assistant Message --" in captured2
