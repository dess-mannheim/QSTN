"""Unit tests for remote inference helper functions and threading wrapper behavior."""

import asyncio
import pytest

from qstn.inference import remote_inference
from qstn.inference.response_generation import (
    JSONResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)


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
    assert remote_inference._create_structured_output(batch_size=1, response_generation_method=None) is None

    # single JSON method
    rgm = JSONResponseGenerationMethod(json_fields=["a"], constraints=None)
    out = remote_inference._create_structured_output(batch_size=2, response_generation_method=rgm)
    assert isinstance(out, list) and len(out) == 2

    # list with JSON and choice
    rgm_list = [rgm, rgm]
    params = remote_inference._create_structured_params(batch_size=2, response_generation_method=rgm_list)
    assert params[0] == params[1]
    choice = ChoiceResponseGenerationMethod(allowed_choices=["x"])
    params2 = remote_inference._create_structured_params(batch_size=1, response_generation_method=choice)
    assert params2 == [["x"]]


def test_run_async_in_thread_success(monkeypatch):
    """Thread wrapper should return whatever asyncio.run returns."""
    expected = (["r"], None, None)

    def fake_run(coro):
        coro.close()
        return expected

    monkeypatch.setattr(asyncio, "run", fake_run)
    result = remote_inference._run_async_in_thread(
        client=None,
        client_model_name="m",
        batch_messages=[],
        seeds=[],
    )
    assert result == expected


def test_run_async_in_thread_error(monkeypatch):
    """Thread wrapper should surface errors raised by asyncio.run."""
    class DummyError(Exception):
        pass

    def raise_err(coro):
        coro.close()
        raise DummyError("fail")
    monkeypatch.setattr(asyncio, "run", raise_err)
    with pytest.raises(DummyError):
        remote_inference._run_async_in_thread(
            client=None,
            client_model_name="m",
            batch_messages=[],
            seeds=[],
        )
