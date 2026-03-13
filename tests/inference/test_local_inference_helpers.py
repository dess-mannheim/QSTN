"""Unit tests for helper utilities in `qstn.inference.local_inference` and printing paths."""

from types import SimpleNamespace

from qstn.inference import local_inference, survey_inference
from qstn.inference.response_generation import (
    ChoiceResponseGenerationMethod,
    JSONItem,
    JSONObject,
    JSONResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)


def test_get_sampling_field_names_and_split(monkeypatch):
    """Fields returned by signature are used to split kwargs."""
    # monkeypatch the helper to return a known set
    monkeypatch.setattr(local_inference, "_get_sampling_field_names", lambda: {"foo", "bar"})

    gen, chat = local_inference._split_kwargs({"foo": 1, "baz": 2, "bar": 3})
    assert gen == {"foo": 1, "bar": 3}
    assert chat == {"baz": 2}


def test_update_logprob_kwargs():
    """Configurations are returned and kwargs augmented correctly."""
    rgm = LogprobResponseGenerationMethod(token_position=5, token_limit=10, top_logprobs=3)
    kwargs = {}
    returned = local_inference._update_logprob_kwargs(rgm, kwargs)
    assert returned is rgm
    assert kwargs["logprobs"] == 3
    assert kwargs["max_tokens"] == 10

    # list containing one and none
    rgm2 = LogprobResponseGenerationMethod()
    kwargs2 = {}
    returned2 = local_inference._update_logprob_kwargs([rgm2], kwargs2)
    assert returned2 is rgm2
    assert "logprobs" in kwargs2

    kwargs3 = {}
    returned3 = local_inference._update_logprob_kwargs(
        [JSONResponseGenerationMethod(json_object=JSONObject(children=[JSONItem("a")]))], kwargs3
    )
    assert returned3 is None


def test_structured_sampling_params_and_cache(monkeypatch):
    """Different response generation methods lead to appropriate SamplingParams."""
    # simple case same method for all
    rgm = JSONResponseGenerationMethod(json_object=JSONObject(children=[JSONItem("a")]))
    outputs = local_inference._structured_sampling_params(
        batch_size=2,
        seeds=[1, 2],
        response_generation_method=rgm,
        extra=99,
    )
    assert len(outputs) == 2
    for sp in outputs:
        assert hasattr(sp, "structured_outputs")
        assert sp.extra == 99

    # list with mixed types and caching
    rgm_list = [rgm, rgm]
    outputs2 = local_inference._structured_sampling_params(
        batch_size=2, seeds=[3, 4], response_generation_method=rgm_list
    )
    assert outputs2[0].structured_outputs is outputs2[1].structured_outputs

    # choice method with allowed choices
    choice = ChoiceResponseGenerationMethod(allowed_choices=["x", "y"])
    outputs3 = local_inference._structured_sampling_params(
        batch_size=1, seeds=[5], response_generation_method=choice
    )
    assert outputs3[0].structured_outputs == {"choice": ["x", "y"]}


def test_create_sampling_params():
    """Sampling params reflect whether structured options are used."""
    # without response_generation_method
    simple = local_inference._create_sampling_params(
        batch_size=1, seeds=[1], response_generation_method=None
    )
    assert isinstance(simple[0], local_inference.SamplingParams)
    assert not hasattr(simple[0], "structured_outputs")

    # with structured but not list
    rgm = JSONResponseGenerationMethod(json_object=JSONObject(children=[JSONItem("a")]))
    struct = local_inference._create_sampling_params(
        batch_size=1, seeds=[2], response_generation_method=rgm
    )
    assert hasattr(struct[0], "structured_outputs")


def test_extract_reasoning_and_answer():
    """Outputs from vllm are parsed correctly into reasoning and answers."""

    class FakeOutput:
        def __init__(self, text):
            class Inner:
                def __init__(self, txt):
                    self.text = txt

            self.outputs = [Inner(text)]

    outputs = [FakeOutput("<think>r</think>hello"), FakeOutput("world")]
    raw, reasonings, answers = local_inference._extract_reasoning_and_answer(
        "<think>", "</think>", outputs
    )
    assert raw[0] == "r"
    assert answers[0] == "hello"
    assert reasonings[1] is None


def test_get_logprobs_ignores_non_logprob_methods():
    """Mixed response-method lists should ignore non-logprob entries safely."""

    class DummyModel:
        class Tokenizer:
            def tokenize(self, text):
                return text.split()

        def get_tokenizer(self):
            return self.Tokenizer()

    class FakeReq:
        def __init__(self):
            token = SimpleNamespace(decoded_token=" a", logprob=-0.7)
            lp_item = SimpleNamespace(values=lambda: [token])
            out = SimpleNamespace(text="ignored", logprobs=[lp_item])
            self.outputs = [out]

    result = local_inference._get_logprobs(
        model=DummyModel(),
        response_generation_method=[LogprobResponseGenerationMethod(), None],
        reasoning_start_token="<think>",
        reasoning_end_token="</think>",
        space_char=" ",
        outputs=[FakeReq()],
        raw_reasonings=[None],
    )
    assert result == [{"a": -0.7}]


def test_print_conversation_branches(monkeypatch):
    """Verify that printing handles both plain and multi-turn formats."""
    written = []
    monkeypatch.setattr(survey_inference.tqdm, "write", lambda msg: written.append(msg))
    # case without assistant_messages
    survey_inference._print_conversation(
        system_messages=["s1"],
        prompts=["p1"],
        assistant_messages=[],
        plain_results=["a1"],
        reasoning_output=[None],
        logprob_result=[None],
        response_generation_method=[None],
        number_of_printed_conversations=1,
    )
    assert "s1" in written[0]

    written.clear()
    # case with assistant_messages and logprob method
    rgm = LogprobResponseGenerationMethod()
    survey_inference._print_conversation(
        system_messages=["s2"],
        prompts=["p2"],
        assistant_messages=[["assist"]],
        plain_results=["a2"],
        reasoning_output=["r2"],
        logprob_result=["lp"],
        response_generation_method=[rgm],
        number_of_printed_conversations=1,
    )
    assert "assist" in written[0]
    assert "Logprobs" in written[0]


def test_print_conversation_with_none_methods(monkeypatch):
    """Conversation printing should tolerate `response_generation_method=None`."""
    written = []
    monkeypatch.setattr(survey_inference.tqdm, "write", lambda msg: written.append(msg))

    survey_inference._print_conversation(
        system_messages=["s"],
        prompts=["p"],
        assistant_messages=[],
        plain_results=["a"],
        reasoning_output=[None],
        logprob_result=[None],
        response_generation_method=None,
        number_of_printed_conversations=1,
    )
    assert "-- Generated Message --" in written[0]


def test_print_conversation_omits_none_system_message(monkeypatch):
    written = []
    monkeypatch.setattr(survey_inference.tqdm, "write", lambda msg: written.append(msg))

    survey_inference._print_conversation(
        system_messages=[None],
        prompts=["p"],
        assistant_messages=[],
        plain_results=["a"],
        reasoning_output=[None],
        logprob_result=[None],
        response_generation_method=None,
        number_of_printed_conversations=1,
    )
    assert "-- System Message --" not in written[0]
    assert "-- User Message ---" in written[0]


def test_print_conversation_keeps_empty_system_message(monkeypatch):
    written = []
    monkeypatch.setattr(survey_inference.tqdm, "write", lambda msg: written.append(msg))

    survey_inference._print_conversation(
        system_messages=[""],
        prompts=["p"],
        assistant_messages=[],
        plain_results=["a"],
        reasoning_output=[None],
        logprob_result=[None],
        response_generation_method=None,
        number_of_printed_conversations=1,
    )
    assert "-- System Message --" in written[0]
