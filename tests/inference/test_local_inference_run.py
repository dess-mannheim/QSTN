"""Behavioral tests for vLLM local inference run helpers."""

import pytest

from qstn.inference import survey_inference, local_inference
from qstn.inference.local_inference import run_vllm_batch, run_vllm_batch_conversation


class DummyModel:
    def __init__(self, return_texts):
        self.return_texts = return_texts
        self.last_chat = None
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

    def get_tokenizer(self):
        return self._tokenizer


def test_run_vllm_batch_basic(monkeypatch):
    """`run_vllm_batch` should return parsed plain answers and empty aux outputs."""
    model = DummyModel(["hello"])
    res = run_vllm_batch(model, system_messages=["s"], prompts=["p"], seed=123)
    assert len(res) == 3
    assert res[0] == ["hello"]
    assert res[1] == [None]
    assert res[2] == [None]


def test_run_vllm_batch_conversation_and_errors(monkeypatch):
    """Conversation helper should preserve model outputs and validate unsupported model types."""
    model = DummyModel(["a","b"])

    # Multi-turn response roundtrip.
    out = run_vllm_batch_conversation(model,
        system_messages=["s"],
        prompts=[["p1","p2"]],
        assistant_messages=[["x"]],
    )
    assert len(out) == 3
    assert out[0] == ["a", "b"]
    assert out[1] is None
    assert out[2] == [None, None]

    # Unsupported type in survey_inference.batch_generation should raise.
    with pytest.raises(ValueError):
        survey_inference.batch_generation(model=123)

    # Unsupported type still raises ValueError even if HAS_VLLM toggled.
    monkeypatch.setattr(survey_inference, "HAS_VLLM", False)
    with pytest.raises(ValueError):
        survey_inference.batch_generation(model=DummyModel(["z"]))


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
