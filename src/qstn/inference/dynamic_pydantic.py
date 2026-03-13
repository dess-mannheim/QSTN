import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from .response_generation import JSONItem, JSONObject


def _safe_name(text: str, prefix: str = "field") -> str:
    name = re.sub(r"\W+", "_", text).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"{prefix}_{name}"
    return name


def _model_name(text: str | None, fallback: str) -> str:
    if text is None:
        return fallback
    safe = _safe_name(text, fallback.lower())
    return "".join(part.capitalize() for part in safe.split("_")) or fallback


def _enum_member_name(value: Any, index: int) -> str:
    base_name = re.sub(r"\W+", "_", str(value)).strip("_").upper()
    if not base_name:
        base_name = "VALUE"
    if base_name[0].isdigit():
        base_name = f"VALUE_{base_name}"
    return f"{base_name}_{index}"


def _create_enum_type(name: str, values: list[Any]) -> type[Enum]:
    members = {
        _enum_member_name(value, index): value for index, value in enumerate(values, start=1)
    }
    return Enum(name, members)


def _base_type_for_item(item: JSONItem) -> type:
    if item.constraints.enum is not None:
        return _create_enum_type(
            _model_name(item.json_field, "EnumValue"),
            item.constraints.enum,
        )
    if item.value_type == "string":
        return str
    if item.value_type == "float":
        return float
    if item.value_type == "int":
        return int
    if item.value_type == "bool":
        return bool
    raise ValueError(f"Unsupported value_type: {item.value_type}")


def _type_for_item(item: JSONItem) -> type:
    base_type = _base_type_for_item(item)
    if item.constraints.nullable:
        return base_type | None
    return base_type


def _field_for_item(item: JSONItem) -> tuple[type, Any]:
    field_kwargs: dict[str, Any] = {
        "alias": item.json_field,
    }

    if item.value_type in {"float", "int"}:
        if item.constraints.ge is not None:
            field_kwargs["ge"] = item.constraints.ge
        if item.constraints.le is not None:
            field_kwargs["le"] = item.constraints.le

    if item.value_type == "string":
        if item.constraints.min_length is not None:
            field_kwargs["min_length"] = item.constraints.min_length
        if item.constraints.max_length is not None:
            field_kwargs["max_length"] = item.constraints.max_length
        if item.constraints.pattern is not None:
            field_kwargs["pattern"] = item.constraints.pattern

    return _type_for_item(item), Field(..., **field_kwargs)


def build_pydantic_model_from_json_object(
    json_object: JSONObject,
    model_name: str = "StructuredOutput",
) -> type[BaseModel]:
    model_fields: dict[str, tuple[type, Any]] = {}

    for child in json_object.children:
        if isinstance(child, JSONItem):
            internal_name = _safe_name(child.json_field, "field")
            model_fields[internal_name] = _field_for_item(child)
            continue

        if child.json_field is None:
            raise ValueError("Nested JSONObject entries must define `json_field`.")

        nested_model = build_pydantic_model_from_json_object(
            json_object=child,
            model_name=_model_name(child.json_field, "NestedObject"),
        )
        internal_name = _safe_name(child.json_field, "object")
        model_fields[internal_name] = (nested_model, Field(..., alias=child.json_field))

    return create_model(
        model_name,
        __config__=ConfigDict(
            populate_by_name=True,
            extra="forbid",
        ),
        **model_fields,
    )
