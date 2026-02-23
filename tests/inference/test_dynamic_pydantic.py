"""Tests for dynamic enum/model generation used by structured inference outputs."""

import pytest
from pydantic import ValidationError

from qstn.inference.dynamic_pydantic import _generate_pydantic_model, _create_enum


def test_create_enum_and_model_with_list_constraints():
    """Enum factory should create correct members and pydantic model enforces values."""
    enum = _create_enum("Color", ["red", "blue"])
    assert enum.RED.value == "red"
    assert enum.BLUE.value == "blue"

    model = _generate_pydantic_model(fields=["color"], constraints={"color": ["red","blue"]})
    # correct values
    inst = model(color="red")
    # enum member is returned, verify its value
    assert inst.color.value == "red"
    # invalid should raise
    with pytest.raises(ValidationError):
        model(color="green")


def test_generate_model_with_float_constraint():
    """Float constraint should result in float typed field."""
    model = _generate_pydantic_model(fields=["score"], constraints={"score": "float"})
    inst = model(score=0.5)
    assert isinstance(inst.score, float)
    with pytest.raises(ValidationError):
        model(score="not a float")


def test_generate_model_with_dict_fields_and_warnings():
    """Providing constraints for nonexistent fields emits a RuntimeWarning."""
    with pytest.warns(RuntimeWarning):
        _generate_pydantic_model(fields={"a": "x"}, constraints={"b": ["c"]})

    # also ensure fields come through when dict used
    model = _generate_pydantic_model(fields={"a": "x", "b": "y"}, constraints=None)
    inst = model(a="hello", b="world")
    assert inst.a == "hello" and inst.b == "world"
