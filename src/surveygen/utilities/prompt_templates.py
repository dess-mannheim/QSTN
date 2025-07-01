from typing import Final

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
