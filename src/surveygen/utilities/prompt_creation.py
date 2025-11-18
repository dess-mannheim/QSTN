from typing import Optional, Dict, Any, Final, List
from enum import Enum
import random

from . import prompt_templates

JSON_START_STRING: Final[
    str
] = """```json
{
"""

JSON_END_STRING: Final[
    str
] = """
}
```"""

FORCED_OPTION_STRING: Final[str] = "Respond only with one of these options:"

COT_STRING: Final[str] = "Think step by step."


class PersonaCall(Enum):
    YOU = "You are"
    I = "I am"
    ACT = "Act as"
    FREETEXT = ""


class Persona:
    def __init__(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        persona_call: PersonaCall = PersonaCall.YOU,
        persona_call_freetext: Optional[str] = None,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.description = description or ""
        self.persona_call_text = self.set_persona_call(
            persona_call, persona_call_freetext
        )

    def set_persona_call(
        self,
        persona_call: PersonaCall = PersonaCall.YOU,
        persona_call_freetext: Optional[str] = None,
    ) -> None:
        persona_str = ""
        if persona_call:
            persona_str = persona_call.value
            if persona_call_freetext:
                persona_str += f" {persona_call_freetext}"
        return persona_str

    def get_persona_prompt(self) -> str:
        prompt = ""
        if self.name:
            prompt += f"{self.persona_call_text} {self.name}. "
        else:
            prompt += f"{self.persona_call_text} "
        if self.attributes:
            attr_str = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
            prompt += f"Attributes: {attr_str}. "
        if self.description:
            prompt += f"{self.description} "
        return prompt

    # def __str__(self) -> str:
    #     attr_str = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
    #     return f"Persona(Name: {self.name}, Attributes: {attr_str}, Description: {self.description})"


class OutputForm:

    def __init__(self):
        self.output_prompt = ""

    def get_output_prompt(self) -> str:
        return self.output_prompt

    def single_answer(
        self,
        forced_options: List[str],
        start_string: str = FORCED_OPTION_STRING,
        randomize: bool = False,
    ) -> None:
        if randomize:
            random.shuffle(forced_options)

        option_string = "|".join(forced_options)

        self.output_prompt = f"{start_string} {option_string}."

    def json(
        self,
        json_attributes: List[str],
        json_explanation: Optional[List[str]],
        json_instructions: Optional[str] = prompt_templates.SYSTEM_JSON_DEFAULT,
        start_string: str = JSON_START_STRING,
        end_string: str = JSON_END_STRING,
        randomize: bool = False,
    ) -> None:
        if json_explanation:
            assert len(json_attributes) == len(
                json_explanation
            ), "Length of attributes and explanation is not the same!"
        assert start_string, "The start string cannot be None"

        if randomize:
            if json_explanation is not None:
                combined = list(zip(json_attributes, json_explanation))
                random.shuffle(combined)

                json_attributes, json_explanation = zip(*combined)
                json_attributes, json_explanation = list(json_attributes), list(
                    json_explanation
                )
            else:
                random.shuffle(json_attributes)

        i = 0
        lines = []
        for i, attribute in enumerate(json_attributes):
            if json_explanation:
                line = f'  "{attribute}": <{json_explanation[i]}>'
            else:
                line = f'  "{attribute}": <{attribute}>'
            lines.append(line)

        if json_instructions is not None:
            self.output_prompt = json_instructions + "\n"
        else:
            self.output_prompt = ""
        self.output_prompt += start_string
        self.output_prompt += ",\n".join(lines)
        self.output_prompt += end_string

    def chain_of_thought(self, start_string: str = COT_STRING) -> None:
        self.output_prompt = start_string

    def no_output_instruction(self) -> None:
        self.output_prompt = ""


class PromptCreation:
    def __init__(self):
        self._persona: Optional[Persona] = None
        self._task_instruction: Optional[str] = None
        self._output_form: Optional[OutputForm] = None

    def create_persona(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        persona_call: PersonaCall = PersonaCall.YOU,
        persona_call_freetext: Optional[str] = None,
    ) -> Persona:
        """Creates a persona with a name, attributes, and an optional description."""
        self._persona = Persona(
            name, attributes, description, persona_call, persona_call_freetext
        )
        return self._persona

    def set_task_instruction(self, instruction: str) -> None:
        """Sets the task instruction."""
        self._task_instruction = instruction

    def set_output_format_closed_answer(
        self,
        forced_options: Optional[List[str]],
        start_string: str = FORCED_OPTION_STRING,
        randomize: bool = False,
    ) -> None:
        """Sets the desired output format for the LLM."""
        self._output_form = OutputForm()
        self._output_form.single_answer(forced_options, start_string, randomize)

    def set_output_format_json(
        self,
        json_attributes: List[str],
        json_explanation: Optional[List[str]] = None,
        json_instructions: Optional[str] = prompt_templates.SYSTEM_JSON_DEFAULT,
        start_string: str = JSON_START_STRING,
        end_string: str = JSON_END_STRING,
        randomize: bool = False,
    ) -> None:
        """Sets the desired output format for the LLM."""
        self._output_form = OutputForm()
        self._output_form.json(
            json_attributes=json_attributes,
            json_explanation=json_explanation,
            json_instructions=json_instructions,
            start_string=start_string,
            end_string=end_string,
            randomize=randomize,
        )

    def set_output_format_cot(self, start_string: str = COT_STRING) -> None:
        self._output_form.chain_of_thought(start_string)

    def get_persona_prompt(self) -> str:
        assert self._persona is not None, "No persona prompt set!"

        return self._persona.get_persona_prompt()

    def get_output_prompt(self) -> str:
        assert self._output_form is not None, "No output form set!"

        return self._output_form.get_output_prompt()

    def generate_prompt(self) -> str:
        """Generates the final prompt based on the persona, task instruction, and output format."""
        if not self._task_instruction:
            raise ValueError("Task instruction is not set.")

        prompt = ""

        if self._persona:
            prompt += self.get_persona_prompt()

        prompt += f"{self._task_instruction}. "

        if self._output_form:
            prompt += self._output_form.get_output_prompt()

        return prompt


# Example usage
if __name__ == "__main__":
    creator = PromptCreation()

    # Create a persona
    # persona = creator.create_persona(
    #     name="AI Assistant",
    #     #attributes={"Knowledge": "Extensive", "Tone": "Friendly"},
    #     description="A helpful AI with a vast knowledge base."
    # )

    # Set task instruction
    creator.set_task_instruction("Explain the concept of gravity")
    # Set output format
    creator.set_ouput_format_json(
        ["reasoning", "answer"],
        ["First think about your response here.", "Give your final answer here."],
    )

    # creator.set_single_answer_output_format(["Yes", "No"])
    # Generate and print the prompt
    prompt = creator.generate_prompt()
    print(prompt)
