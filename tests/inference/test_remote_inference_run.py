"""Behavioral tests for OpenAI remote batch runner wrappers."""

from qstn.inference import remote_inference


def test_run_openai_batch_pass_through(monkeypatch):
    """Ensure run_openai_batch forwards parameters to helper and returns result."""
    called = {}

    def fake_helper(**kwargs):
        called.update(kwargs)
        return (["ans"], [{"token": "A"}], ["reason"])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_helper)

    model = object()
    out = remote_inference.run_openai_batch(
        model=model,
        system_messages=["s"],
        prompts=["p"],
        client_model_name="cm",
        api_concurrency=5,
        foo=1,
    )
    assert out[0] == ["ans"]
    assert out[1] == [{"token": "A"}]
    assert out[2] == ["reason"]
    # helper was called with expected keys
    assert called.get("client") is model
    assert called.get("client_model_name") == "cm"
    # api_concurrency gets transformed to concurrency_limit when forwarded
    assert called.get("concurrency_limit") == 5
    assert called.get("batch_messages") == [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]
    ]


def test_run_openai_batch_supports_none_system_messages(monkeypatch):
    called = {}

    def fake_helper(**kwargs):
        called.update(kwargs)
        return (["ans"], [None], [None])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_helper)

    remote_inference.run_openai_batch(
        model=object(),
        system_messages=None,
        prompts=["p"],
        client_model_name="cm",
    )

    assert called.get("batch_messages") == [[{"role": "user", "content": "p"}]]


def test_run_openai_batch_keeps_empty_system_message(monkeypatch):
    called = {}

    def fake_helper(**kwargs):
        called.update(kwargs)
        return (["ans"], [None], [None])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_helper)

    remote_inference.run_openai_batch(
        model=object(),
        system_messages=[""],
        prompts=["p"],
        client_model_name="cm",
    )

    assert called.get("batch_messages") == [
        [{"role": "system", "content": ""}, {"role": "user", "content": "p"}]
    ]


def test_run_openai_batch_conversation(monkeypatch):
    called = {}

    def fake_helper(**kwargs):
        called.update(kwargs)
        return (["cans"], [{"token": "C"}], ["conv-reason"])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_helper)

    model = object()
    out = remote_inference.run_openai_batch_conversation(
        model=model,
        system_messages=["s"],
        prompts=[["p1"]],
        assistant_messages=[["a1"]],
        client_model_name="cm2",
    )
    assert out[0] == ["cans"]
    assert out[1] == [{"token": "C"}]
    assert out[2] == ["conv-reason"]
    assert called.get("batch_messages")[0][-1]["content"] == "a1"


def test_run_openai_batch_conversation_supports_none_system_messages(monkeypatch):
    called = {}

    def fake_helper(**kwargs):
        called.update(kwargs)
        return (["cans"], [None], [None])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_helper)

    remote_inference.run_openai_batch_conversation(
        model=object(),
        system_messages=None,
        prompts=[["p1"]],
        assistant_messages=[["a1"]],
        client_model_name="cm2",
    )

    assert called.get("batch_messages") == [
        [{"role": "user", "content": "p1"}, {"role": "assistant", "content": "a1"}]
    ]


def test_run_openai_batch_conversation_keeps_empty_system_message(monkeypatch):
    called = {}

    def fake_helper(**kwargs):
        called.update(kwargs)
        return (["cans"], [None], [None])

    monkeypatch.setattr(remote_inference, "_run_async_in_thread", fake_helper)

    remote_inference.run_openai_batch_conversation(
        model=object(),
        system_messages=[""],
        prompts=[["p1"]],
        assistant_messages=[[]],
        client_model_name="cm2",
    )

    assert called.get("batch_messages") == [
        [{"role": "system", "content": ""}, {"role": "user", "content": "p1"}]
    ]
