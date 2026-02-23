"""Direct unit tests for `parse_reasoning` edge cases."""

import pytest

from qstn.inference.reasoning_parser import parse_reasoning


@pytest.mark.parametrize("full_text,patterns,expected", [
    # no tags at all, should return text as answer with no reasoning
    ("Hello world", [("<s>","</s>")], ("Hello world", None)),
    # start and end present
    ("<think>R</think>Answer", [("<think>","</think>")], ("Answer", "R")),
    # only end tag
    ("R</think>Answer", [("<think>","</think>")], ("Answer", "R")),
    # only start tag (cutoff)
    ("<think>R", [("<think>","</think>")], ("", "R")),
    # multiple patterns: first matched used
    ("<a>foo</a>bar", [("<a>","</a>"),("<b>","</b>")], ("bar","foo")),
    # pattern list empty
    ("<think>here</think>res", [], ("<think>here</think>res", None)),
])
def test_parse_reasoning_various(full_text, patterns, expected):
    """Ensure all reasoning parsing cases behave as documented."""
    answer, reasoning = parse_reasoning(full_text, patterns)
    assert (answer, reasoning) == expected


def test_parse_reasoning_edge_cases():
    """Check that when tags appear but no closing sequence occurs we still return correct answer."""
    # For multiple tagged sections, parser removes the first section only.
    answer, reasoning = parse_reasoning("<think>a</think><think>b</think>c", [("<think>","</think>")])
    assert answer == "<think>b</think>c"
    assert reasoning == "a"

    # tag content empty
    answer, reasoning = parse_reasoning("<think></think>", [("<think>","</think>")])
    assert answer == "" and reasoning == ""
