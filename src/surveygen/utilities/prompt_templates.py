from typing import Final

from . import placeholder

# --- Empty Answer Option Template ---
# use this if answer options are already provided in the system prompt
NO_ANSWER_OPTIONS: Final[None] = None


# --- List Option Templates (for multiple choice) ---
# Placeholder: {options}

LIST_OPTIONS_DEFAULT: Final[str] = "Options are: {options}"
LIST_OPTIONS_CHOICES: Final[str] = (
    "Please select one of the following choices: {options}"
)
LIST_OPTIONS_BRACKETS: Final[str] = "The available options are [{options}]."
LIST_OPTIONS_SELECT: Final[str] = "Select an answer from the list: {options}"
LIST_OPTIONS_CONVERSATIONAL: Final[str] = (
    "Okay, here are your choices: {options}. Which one is it?"
)
LIST_OPTIONS_DIRECT: Final[str] = "Your choices: {options}"
LIST_OPTIONS_QUESTION: Final[str] = "Which of the following is correct? {options}"
LIST_OPTIONS_MINIMAL: Final[str] = "{options}"


# --- Scale Option Templates (for ranges) ---
# Placeholders: {start}, {end}

SCALE_OPTIONS_DEFAULT: Final[str] = "Options range from {start} to {end}"
SCALE_OPTIONS_RATING: Final[str] = (
    "Please provide a rating on a scale from {start} to {end}."
)
SCALE_OPTIONS_SPECTRUM: Final[str] = (
    "Indicate your response on a spectrum where {start} and {end}."
)
SCALE_OPTIONS_LIKELIHOOD: Final[str] = (
    "How likely do you rate this, from {start} to {end}?"
)
SCALE_OPTIONS_CONVERSATIONAL: Final[str] = (
    "On a scale from {start} to {end}, how would you rate it?"
)
SCALE_OPTIONS_DIRECT: Final[str] = "Rate from {start} to {end}."
SCALE_OPTIONS_ARROW: Final[str] = "Scale: {start} -> {end}"
SCALE_OPTIONS_MINIMAL: Final[str] = "{start} to {end}"


# --- System Prompt Templates ---
# optionally add formatting instructions to the system prompt
# Placeholder: {options}

SYSTEM_JSON_DEFAULT: Final[str] = (
    f"""You only respond in the following JSON format:\n{placeholder.JSON_TEMPLATE}"""
)

SYSTEM_JSON_SINGLE_ANSWER: Final[str] = (
    f"""You only respond with the most probable answer option in the following JSON format:\n{placeholder.JSON_TEMPLATE}"""
)

SYSTEM_JSON_REASONING: Final[
    str
] = f"""You always reason about the possible answer options first.
You respond with your reasoning and the most probable answer option in the following JSON format:\n{placeholder.JSON_TEMPLATE}"""

SYSTEM_JSON_ALL_OPTIONS: Final[str] = (
    f"""You only respond with a probability for each answer option in the following JSON format:\n{placeholder.JSON_TEMPLATE}"""
)

SYSTEM_SINGLE_ANSWER: Final[str] = (
    f"""You only respond with the most probable answer option."""
)

# German versions

DE_SYSTEM_JSON_DEFAULT: Final[str] = (
    f"""Du antwortest ausschließlich im folgenden JSON-Format:\n {placeholder.JSON_TEMPLATE}"""
)

DE_SYSTEM_JSON_SINGLE_ANSWER: Final[str] = (
    f"""Du antwortest ausschließlich mit der wahrscheinlichsten Antwortoption im folgenden JSON-Format:\n{placeholder.JSON_TEMPLATE}"""
)

DE_SYSTEM_JSON_REASONING: Final[
    str
] = f"""Du argumentierst immer zuerst über die möglichen Antwort-Optionen.
Du antwortest mit deiner Argumentation und der wahrscheinlichsten Antwort-Option im folgenden JSON-Format:\n{placeholder.JSON_TEMPLATE}"""

DE_SYSTEM_JSON_ALL_OPTIONS: Final[str] = (
    f"""Du antwortest ausschließlich mit einer Wahrscheinlichkeit für jede Antwort-Option im folgenden JSON-Format:\n{placeholder.JSON_TEMPLATE}"""
)

DE_SYSTEM_SINGLE_ANSWER: Final[str] = (
    f"""Du antwortest ausschließlich mit der wahrscheinlichsten Antwortoption."""
)
