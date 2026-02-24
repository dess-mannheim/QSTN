"""Smoke tests for constants and prompt template primitives."""

from qstn.utilities import constants, prompt_templates, placeholder


def test_constants_values():
    """Core constant values should be defined and of expected types."""
    assert constants.SYSTEM_PROMPT_FIELD == "system_prompt"
    assert isinstance(constants.OPTIONS_ADJUST, list)


def test_templates_contain_placeholders():
    """Templates should format correctly and include placeholder markers."""
    # pick a couple of templates, ensure formatting works
    s = prompt_templates.LIST_OPTIONS_DEFAULT.format(options="x")
    assert "x" in s
    t = prompt_templates.SCALE_OPTIONS_DEFAULT.format(start="1", end="5")
    assert "1" in t and "5" in t
    # placeholder constants are braces strings
    assert placeholder.PROMPT_QUESTIONS.startswith("{{")
