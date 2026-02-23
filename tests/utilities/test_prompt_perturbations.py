"""Behavioral tests for safe prompt perturbation utilities.

These tests focus on the important contract:
- base perturbation methods are no-ops for empty/zero-probability input,
- placeholders like ``{VAR}`` are preserved,
- batch helpers rebuild prompt structure correctly.
"""

import random

from qstn.utilities import prompt_perturbations


def test_basic_character_perturbations_handle_empty_and_zero_probability():
    """Character-level perturbations should not mutate trivial inputs."""
    assert prompt_perturbations.key_typos("", probability=0.5) == ""
    assert prompt_perturbations.key_typos("abc", probability=0.0) == "abc"
    assert prompt_perturbations.keyboard_typos("", probability=0.5) == ""
    assert prompt_perturbations.keyboard_typos("abc", probability=0.0) == "abc"
    assert prompt_perturbations.letter_swaps("", probability=0.5) == ""
    assert prompt_perturbations.letter_swaps("abc", probability=0.0) == "abc"


def test_apply_safe_perturbation_preserves_placeholders():
    """`apply_safe_perturbation` must modify text outside placeholders only."""
    def uppercase(text, **_kwargs):
        return text.upper()

    out = prompt_perturbations.apply_safe_perturbation(
        ["hello {KEEP} world"],
        uppercase,
    )

    assert out == ["HELLO {KEEP} WORLD"]


def test_apply_safe_perturbation_with_key_typos_keeps_placeholder_content():
    """Even aggressive mutation must keep placeholder tokens unchanged."""
    random.seed(7)
    out = prompt_perturbations.apply_safe_perturbation(
        ["abc {X} def"],
        prompt_perturbations.key_typos,
        probability=1.0,
    )

    assert "{X}" in out[0]
    assert out[0] != "abc {X} def"


def test_batch_perturbation_helpers_round_trip_structure(monkeypatch):
    """Synonym/paraphrase helpers should perturb segments and preserve placeholders."""
    def fake_batch_generation(model, system_messages, prompts, **kwargs):
        return ([f"{prompt}_mod" for prompt in prompts], None, None)

    monkeypatch.setattr(prompt_perturbations, "batch_generation", fake_batch_generation)

    prompts = ["a{X}b"]
    synonyms = prompt_perturbations.make_synonyms(prompts, model="m", instruction="inst:")
    paraphrases = prompt_perturbations.make_paraphrase(prompts, model="m", instruction="inst:")

    assert synonyms[0] == "inst:a_mod{X}inst:b_mod"
    assert paraphrases[0] == "inst:a_mod{X}inst:b_mod"


def test_apply_safe_perturbation_dispatches_batch_helpers(monkeypatch):
    """Batch-safe wrapper should delegate to designated batch perturbation functions."""
    called = {"synonyms": False}

    def fake_synonyms(all_prompts, model, instruction):
        called["synonyms"] = True
        return ["done"]

    monkeypatch.setattr(prompt_perturbations, "make_synonyms", fake_synonyms)

    out = prompt_perturbations.apply_safe_perturbation(
        ["x"],
        prompt_perturbations.make_synonyms,
        model="m",
        instruction="i",
    )

    assert called["synonyms"] is True
    assert out == ["done"]
