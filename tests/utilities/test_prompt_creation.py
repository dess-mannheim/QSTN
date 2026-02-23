"""Unit tests for prompt construction utilities.

Focus:
- persona text rendering,
- output format generation behavior,
- prompt assembly validation in `PromptCreation`.
"""

import pytest

from qstn.utilities.prompt_creation import (
    COT_STRING,
    OutputForm,
    Persona,
    PersonaCall,
    PromptCreation,
)


def test_persona_prompt_includes_all_configured_fields():
    """Persona prompt should include call, name, attributes, and description."""
    persona = Persona(
        name="Bob",
        attributes={"age": 30, "city": "Berlin"},
        description="A concise responder.",
        persona_call=PersonaCall.ACT,
    )

    prompt = persona.get_persona_prompt()

    assert prompt.startswith("Act as Bob.")
    assert "age: 30" in prompt
    assert "city: Berlin" in prompt
    assert "A concise responder." in prompt


def test_persona_freetext_call_is_applied():
    """Free-text persona call should be carried through verbatim."""
    persona = Persona(
        persona_call=PersonaCall.FREETEXT,
        persona_call_freetext="Please behave like",
    )
    assert persona.persona_call_text == " Please behave like"


def test_output_form_single_answer_and_json_validation(monkeypatch):
    """OutputForm should randomize options and validate JSON explanation lengths."""
    output = OutputForm()

    shuffled = {"called": False}

    def fake_shuffle(values):
        shuffled["called"] = True
        values.reverse()

    monkeypatch.setattr("qstn.utilities.prompt_creation.random.shuffle", fake_shuffle)
    options = ["A", "B", "C"]
    output.single_answer(options, randomize=True)

    assert shuffled["called"] is True
    assert output.get_output_prompt().endswith("C|B|A.")

    with pytest.raises(AssertionError, match="Length of attributes"):
        output.json(["a", "b"], ["only_one"])


def test_prompt_creation_requires_task_and_builds_full_prompt():
    """PromptCreation should require a task and then build a full composite prompt."""
    creator = PromptCreation()

    with pytest.raises(ValueError, match="Task instruction is not set"):
        creator.generate_prompt()

    creator.create_persona(name="Assistant", persona_call=PersonaCall.YOU)
    creator.set_task_instruction("Decide yes or no")
    creator.set_output_format_closed_answer(["yes", "no"])

    combined = creator.generate_prompt()
    assert "You are Assistant." in combined
    assert "Decide yes or no." in combined
    assert "yes|no" in combined

    creator.set_output_format_cot()
    assert creator.get_output_prompt() == COT_STRING
