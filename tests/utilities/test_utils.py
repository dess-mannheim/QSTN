"""Unit tests for generic helper functions in `qstn.utilities.utils`.

These tests validate determinism, dataframe assembly behavior, and safe template
substitution semantics.
"""

import pytest
import pandas as pd

from qstn.utilities import utils


class DummyKey:
    """Minimal key object exposing `questionnaire_name` for dataframe merge tests."""

    def __init__(self, name):
        self.questionnaire_name = name


def test_make_cache_key_stable():
    """`_make_cache_key` should be deterministic for identical inputs."""
    key1 = utils._make_cache_key([1,2], {"a":3})
    key2 = utils._make_cache_key([1,2], {"a":3})
    assert key1 == key2
    assert isinstance(key1, str)


def test_generate_seeds_reproducible():
    """`generate_seeds` should be reproducible and sized by batch."""
    seeds1 = utils.generate_seeds(seed=5, batch_size=3)
    seeds2 = utils.generate_seeds(seed=5, batch_size=3)
    assert seeds1 == seeds2
    assert len(seeds1) == 3
    # Different base seed should produce a different sample stream.
    seeds3 = utils.generate_seeds(seed=6, batch_size=3)
    assert seeds3 != seeds1


def test_create_one_dataframe_empty():
    """create_one_dataframe should return empty DataFrame when given empty dict."""
    assert utils.create_one_dataframe({}).empty


def test_create_one_dataframe_nonempty():
    """Merged dataframe should include source questionnaire names and all rows."""
    df1 = pd.DataFrame({"a":[1]})
    df2 = pd.DataFrame({"a":[2]})
    data = {
        DummyKey("Q1"): df1,
        DummyKey("Q2"): df2,
    }
    result = utils.create_one_dataframe(data)
    assert len(result) == 2
    assert "questionnaire_name" in result.columns
    assert set(result.questionnaire_name) == {"Q1","Q2"}


def test_safe_format_with_regex():
    """Regex formatter should replace known placeholders and keep unknown placeholders."""
    template = "Hello {{name}}, {{missing}} stays."
    out = utils.safe_format_with_regex(template, {"{{name}}":"Alice"})
    assert "Alice" in out
    assert "{{missing}}" in out


def test_safe_format_empty_template():
    """Formatting an empty template should return an empty string."""
    assert utils.safe_format_with_regex("", {}) == ""
