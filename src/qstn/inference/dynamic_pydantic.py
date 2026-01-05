from typing import List, Dict, Union
import warnings
from pydantic import BaseModel, create_model
from enum import Enum


def _create_enum(name: str, values: List[str | int]) -> Enum:
    """A helper to create an Enum class dynamically."""
    return Enum(name, {str(v).upper(): v for v in values})


def _generate_pydantic_model(
    fields: Union[List[str], Dict[str, str]], constraints: Dict[str, List[str]]
) -> BaseModel:
    """Dynamically creates a Pydantic model based on a list of fields and constraints.

    This helper function is used to generate a Pydantic `BaseModel` on the fly.
    It defines fields as strings by default, but can apply specific constraints,
    such as creating an `Enum` for a field with a predefined list of options, or
    typing a field as a `float`.

    Args:
        fields (Union[List[str], Dict[str, str]]): The fields to include in the
            generated model. If a `List[str]`, each string becomes a field
            name. If a `Dict[str, str]`, the keys are used as field names (the
            dictionary values are currently ignored).
        constraints (Dict[str, Union[List[str], str]]): A dictionary mapping field
            names to their constraints.
            - If the value is a `List[str]`, the corresponding field will be
              constrained to an `Enum` of those string options.
            - If the value is the literal string 'float', the field will be
              typed as a `float`, suitable for probabilities or scores.
            Fields not present in this dictionary will default to type `str`.

    Warns:
        RuntimeWarning: If the `constraints` dictionary contains keys for fields
            that are not defined in the `fields` parameter.

    Returns:
        pydantic.BaseModel: A dynamically generated Pydantic `BaseModel` class
            with the specified fields and types.
    """

    model_fields = {}
    if constraints:
        if isinstance(fields, Dict):
            difference = set(constraints.keys()) - set(fields.keys())
        else:
            difference = set(constraints.keys()) - set(fields)
        if len(difference) > 0:
            warnings.warn(
                f"Constraints specified for non-existing fields: {difference}. "
                + "Constraints should be provided in the format {'a JSON field': ['option 1',...]}.",
                RuntimeWarning,
                stacklevel=2,
            )

    if isinstance(fields, Dict):
        elements = fields.keys()
    else:
        elements = fields

    for field in elements:
        if constraints:
            if field in constraints and isinstance(constraints[field], list):
                enum_type = _create_enum(field.capitalize() + "Enum", constraints[field])
                model_fields[str(field)] = (enum_type, ...)
            # allow for probability distribution across answer options
            elif field in constraints and constraints[field] == "float":
                model_fields[str(field)] = (float, ...)
        else:
            model_fields[str(field)] = (str, ...)

    return create_model("DynamicModel", **model_fields)
