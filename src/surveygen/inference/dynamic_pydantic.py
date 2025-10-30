from typing import List, Dict, Any, Union
import warnings
from pydantic import BaseModel, create_model
from enum import Enum


def create_enum(name: str, values: List[str]):
    return Enum(name, {v.upper(): v for v in values})


def generate_pydantic_model(
    fields: Union[List[str], Dict[str, str]], constraints: Dict[str, List[str]]
) -> BaseModel:
    model_fields = {}

    if isinstance(fields, Dict):
        difference = set(constraints.keys())- set(fields.keys())
    else:
        difference = set(constraints.keys())- set(fields)
    if len(difference) > 0:
        warnings.warn(f"Constraints specified for non-existing fields: {difference}. " + 
                       "Constraints should be provided in the format {'a JSON field': ['option 1',...]}.",
                       RuntimeWarning, stacklevel=2)

    if isinstance(fields, Dict):
        elements = fields.keys()
    else:
        elements = fields

    for field in elements:
        if field in constraints and isinstance(constraints[field], list):
            enum_type = create_enum(field.capitalize() + "Enum", constraints[field])
            model_fields[str(field)] = (enum_type, ...)
        # allow for probability distribution across answer options
        elif field in constraints and constraints[field] == "float":
            model_fields[str(field)] = (float, ...)
        else:
            model_fields[str(field)] = (str, ...)

    return create_model("DynamicModel", **model_fields)
