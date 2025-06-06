from typing import List, Dict, Any
from pydantic import BaseModel, create_model
from enum import Enum

def create_enum(name: str, values: List[str]):
    return Enum(name, {v.upper(): v for v in values})

def generate_pydantic_model(fields: List[str], constraints: Dict[str, List[str]]) -> BaseModel:
    model_fields = {}
    for field in fields:
        if field in constraints:
            enum_type = create_enum(field.capitalize() + "Enum", constraints[field])
            model_fields[field] = (enum_type, ...)
        else:
            model_fields[field] = (str, ...)
    return create_model('DynamicModel', **model_fields)