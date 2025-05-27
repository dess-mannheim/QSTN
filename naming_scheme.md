# SurveyGen: Naming Scheme

## Mikro-Level Naming Scheme (Q&A Level)

- Item Stem (Questions related to a specific category e.g. "What is your opinion of ...") -> item_stem
- Item Statement (e.g. _Is talkative_ in "Do you think this statement fits to you is talkative") -> item_statement
- Item = Item Statement + Item Stem (doesn't have to be composite, can be single question) -> item
- Item ID (im01, im02, (identifier for item)) -> item_id
- Answer Code (1: ..., 2: ..., can also be characters) -> answer_code
- Answer Text (e.g. "Agree", "Disagree", ...) -> answer_text
    - randomized
    - reversed
    - odd vs. even scaling
- Answer option = Answer Code + Answer Text -> answer_option
- Refusal (non-substantive responses like "Don't know", etc.)

**Scale Types**

- Nominal Scale:
    - Single-Choice (e.g. marital status)
    - Multiple-Choice (e.g. spoken languages)
- Ordinal Scale:
    - Likert Scales (e.g. agreement)
    - semantic differential scales (bipolar adjectives at scale ends, e.g. expensive <-> affordable)
- Interval Scales
    - numeric/linear scales (often from 1 to 10)
- Ratio Scales
    - interval scales with an absolute zero (working hours per week)


## Macro-Level Naming Scheme (Survey Level)

- Constructs (multiple items build one construct)
