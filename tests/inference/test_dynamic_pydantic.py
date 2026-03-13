"""Tests for tree-based pydantic model generation used by structured outputs."""

import pytest
from pydantic import ValidationError

from qstn.inference.dynamic_pydantic import build_pydantic_model_from_json_object
from qstn.inference.response_generation import Constraints, JSONItem, JSONObject


def test_build_pydantic_model_enforces_enum_values():
    """Enum constraints should produce schema-backed runtime validation."""
    model = build_pydantic_model_from_json_object(
        JSONObject(
            children=[
                JSONItem(
                    json_field="color",
                    constraints=Constraints(enum=["red", "blue"]),
                )
            ]
        )
    )

    inst = model(color="red")

    assert inst.color.value == "red"
    with pytest.raises(ValidationError):
        model(color="green")


def test_build_pydantic_model_uses_declared_scalar_types():
    """Declared scalar types should be enforced by the generated model."""
    model = build_pydantic_model_from_json_object(
        JSONObject(children=[JSONItem(json_field="score", value_type="float")])
    )

    inst = model(score=0.5)

    assert isinstance(inst.score, float)
    with pytest.raises(ValidationError):
        model(score="not a float")


def test_build_pydantic_model_supports_nested_objects_and_aliases():
    """Nested JSONObjects should become nested pydantic models with aliases."""
    model = build_pydantic_model_from_json_object(
        JSONObject(
            children=[
                JSONObject(
                    json_field="Question 1",
                    children=[JSONItem(json_field="answer", value_type="int")],
                )
            ]
        )
    )

    inst = model(**{"Question 1": {"answer": 2}})

    assert inst.question_1.answer == 2
    with pytest.raises(ValidationError):
        model(**{"Question 1": {"answer": "wrong"}})


def test_build_pydantic_model_rejects_nested_objects_without_json_field():
    """Nested JSONObjects must expose a field name in the parent schema."""
    with pytest.raises(ValueError, match="Nested JSONObject entries must define `json_field`."):
        build_pydantic_model_from_json_object(
            JSONObject(children=[JSONObject(children=[JSONItem(json_field="answer")])])
        )
