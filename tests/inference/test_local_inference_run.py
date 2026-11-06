"""Behavioral tests for vLLM local inference run helpers."""

import pytest

from qstn.inference import local_inference, survey_inference
from qstn.inference.local_inference import run_vllm_batch, run_vllm_batch_conversation
from qstn.inference.response_generation import LogprobResponseGenerationMethod


class DummyModel:
    def __init__(self, return_texts, generation_config=None):
        self.return_texts = return_texts
        self.generation_config = generation_config or {}
        self.default_sampling_params = None
        self.last_chat = None
        self.last_generate = None

        # mimic a tokenizer with tokenize method for logprob test
        class Tokenizer:
            def tokenize(self, txt):
                return txt.split()

        self._tokenizer = Tokenizer()

    def chat(self, batch_messages, sampling_params=None, use_tqdm=None, **kwargs):
        self.last_chat = {
            "batch_messages": batch_messages,
            "sampling_params": sampling_params,
            "use_tqdm": use_tqdm,
            "kwargs": kwargs,
        }
        # return list of fake RequestOutput objects with text from return_texts
        outputs = []
        for txt in self.return_texts:

            class Inner:
                def __init__(self, text):
                    self.outputs = [type("O", (), {"text": text, "logprobs": []})]

            outputs.append(Inner(txt))
        return outputs

    def generate(self, prompts, sampling_params=None, use_tqdm=None, **kwargs):
        self.last_generate = {
            "prompts": prompts,
            "sampling_params": sampling_params,
            "use_tqdm": use_tqdm,
            "kwargs": kwargs,
        }
        outputs = []
        for txt in self.return_texts:

            class Inner:
                def __init__(self, text):
                    self.outputs = [type("O", (), {"text": text, "logprobs": []})]

            outputs.append(Inner(txt))
        return outputs

    def get_tokenizer(self):
        return self._tokenizer

    def get_default_sampling_params(self):
        # vLLM lazily caches the model's generation config on this attribute.
        if self.default_sampling_params is None:
            self.default_sampling_params = self.generation_config
        return local_inference.SamplingParams(**self.default_sampling_params)


def test_run_vllm_batch_basic(monkeypatch):
    """`run_vllm_batch` should return parsed plain answers and empty aux outputs."""
    model = DummyModel(["hello"])
    res = run_vllm_batch(model, system_messages=["s"], prompts=["p"], seed=123)
    assert len(res) == 3
    assert res[0] == ["hello"]
    assert res[1] == [None]
    assert res[2] == [None]


def test_run_vllm_batch_supports_none_system_messages():
    model = DummyModel(["hello"])
    run_vllm_batch(model, system_messages=None, prompts=["p"], seed=123)

    assert model.last_chat["batch_messages"] == [[{"role": "user", "content": "p"}]]


def test_run_vllm_batch_keeps_empty_system_message():
    model = DummyModel(["hello"])
    run_vllm_batch(model, system_messages=[""], prompts=["p"], seed=123)

    assert model.last_chat["batch_messages"] == [
        [{"role": "system", "content": ""}, {"role": "user", "content": "p"}]
    ]


def test_run_vllm_batch_conversation_and_errors(monkeypatch):
    """Conversation helper should preserve model outputs and validate unsupported model types."""
    model = DummyModel(["a", "b"])

    # Multi-turn response roundtrip.
    out = run_vllm_batch_conversation(
        model,
        system_messages=["s"],
        prompts=[["p1", "p2"]],
        assistant_messages=[["x"]],
    )
    assert len(out) == 3
    assert out[0] == ["a", "b"]
    assert out[1] == [None, None]
    assert out[2] == [None, None]

    # Unsupported type in survey_inference.batch_generation should raise.
    with pytest.raises(ValueError):
        survey_inference.batch_generation(model=123)

    # Unsupported type still raises ValueError even if HAS_VLLM toggled.
    monkeypatch.setattr(survey_inference, "HAS_VLLM", False)
    with pytest.raises(ValueError):
        survey_inference.batch_generation(model=DummyModel(["z"]))


def test_run_vllm_batch_conversation_supports_none_system_messages():
    model = DummyModel(["a", "b"])
    out = run_vllm_batch_conversation(
        model,
        system_messages=None,
        prompts=[["p1", "p2"]],
        assistant_messages=[["x"]],
    )

    assert out[0] == ["a", "b"]
    assert model.last_chat["batch_messages"] == [
        [
            {"role": "user", "content": "p1"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "p2"},
        ]
    ]


def test_run_vllm_batch_conversation_keeps_empty_system_message():
    model = DummyModel(["a"])
    run_vllm_batch_conversation(
        model,
        system_messages=[""],
        prompts=[["p1"]],
        assistant_messages=[[]],
    )

    assert model.last_chat["batch_messages"] == [
        [{"role": "system", "content": ""}, {"role": "user", "content": "p1"}]
    ]


def test_run_vllm_batch_warns_for_sampling_params_and_respects_use_tqdm(monkeypatch):
    """`sampling_params` kwarg should warn, and `use_tqdm` should override print_progress."""
    model = DummyModel(["answer"])
    monkeypatch.setattr(local_inference, "_get_sampling_field_names", lambda: {"temperature"})

    with pytest.warns(UserWarning, match="sampling_params"):
        run_vllm_batch(
            model,
            system_messages=["s"],
            prompts=["p"],
            print_progress=True,
            use_tqdm=False,
            sampling_params={"ignored": True},
            temperature=0.2,
            custom_flag=True,
        )

    assert model.last_chat["use_tqdm"] is False
    assert model.last_chat["kwargs"] == {"custom_flag": True}
    assert model.last_chat["sampling_params"][0].temperature == 0.2


def test_run_vllm_batch_completion_mode_uses_generate_with_exact_prompt():
    model = DummyModel(["answer"])

    out = run_vllm_batch(
        model,
        system_messages=["SYS"],
        prompts=["Prompt"],
        inference_mode="completion",
        seed=123,
    )

    assert out[0] == ["answer"]
    assert model.last_chat is None
    assert model.last_generate["prompts"] == ["Prompt"]


def test_run_vllm_batch_conversation_completion_mode_uses_final_prompt_text():
    model = DummyModel(["next"])

    run_vllm_batch_conversation(
        model,
        system_messages=[None],
        prompts=[["p1", "p2"]],
        assistant_messages=[["a1"]],
        inference_mode="completion",
    )

    assert model.last_generate["prompts"] == ["p2"]


@pytest.mark.parametrize(
    ("inference_mode", "last_call"), [("chat", "last_chat"), ("completion", "last_generate")]
)
def test_run_vllm_batch_prefers_user_args_over_model_defaults(
    monkeypatch, inference_mode, last_call
):
    """Sampling priority is user args, then the model's generation config."""
    monkeypatch.setattr(
        local_inference, "_get_sampling_field_names", lambda: {"temperature", "max_tokens"}
    )
    model = DummyModel(["hello"], generation_config={"temperature": 0.6, "top_p": 0.9})

    run_vllm_batch(
        model,
        system_messages=[None],
        prompts=["p"],
        inference_mode=inference_mode,
        temperature=0.1,
        max_tokens=7,
    )

    params = getattr(model, last_call)["sampling_params"][0]
    assert params.temperature == 0.1
    assert params.top_p == 0.9
    assert params.max_tokens == 7


def test_run_vllm_batch_conversation_uses_model_defaults():
    model = DummyModel(["a"], generation_config={"temperature": 0.6})

    run_vllm_batch_conversation(
        model,
        system_messages=["s"],
        prompts=[["p"]],
        assistant_messages=[[]],
    )

    assert model.last_chat["sampling_params"][0].temperature == 0.6


def test_run_vllm_batch_without_generation_config_keeps_vllm_defaults():
    model = DummyModel(["hello"])

    run_vllm_batch(model, system_messages=["s"], prompts=["p"], seed=123)

    assert not hasattr(model.last_chat["sampling_params"][0], "temperature")


def test_run_vllm_batch_logprob_token_limit_overrides_model_default(monkeypatch):
    monkeypatch.setattr(local_inference, "_get_sampling_field_names", lambda: {"max_tokens"})
    model = DummyModel(["hello"], generation_config={"max_tokens": 512})

    run_vllm_batch(
        model,
        system_messages=["s"],
        prompts=["p"],
        response_generation_method=LogprobResponseGenerationMethod(
            token_limit=1, constrain_answer_options=False
        ),
    )

    assert model.last_chat["sampling_params"][0].max_tokens == 1
