# Tutorial 1: Multiple Choice Questionnaires

Often when inferencing LLMs, e.g., when annotating data into labels, we want to have answers to multiple choice questions or we want models to give answers on a scale. SurveyGen supports various checks to ensure that the output of the model is robust. In this tutorial we will take a look at how to use SurveyGen to produce robust results and how to use different response generation methods, such as restricting the model output, logprobs and open answer formats.


For now we keep our example from the [quickstart tutorial](https://surveygen.readthedocs.io/en/latest/quickstart.html) but we want our models to answer on a Likert Scale from "1: Strongly Dislike to "5: Strongly Like".


Let's setup our two interviews again with our "sample population".


```python
from surveygen.llm_interview import LLMInterview
from surveygen.utilities import placeholder

import pandas as pd

personas = {
    "New York": "a black middle aged man from New York",
    "Texas": "a white middle aged man from Texas",
}

system_prompt = "Act as if you were {persona}!"
prompt = f"Please tell us how you feel about the following parties:\n{placeholder.PROMPT_QUESTIONS}"

party_questionnaire = pd.read_csv("parties.csv")

interviews = []

for name, persona in personas.items():
    interviews.append(
        LLMInterview(
            interview_name=name,
            interview_source=party_questionnaire,
            system_prompt=system_prompt.format_map({"persona": persona}),
            prompt=prompt,
        )
    )
```

So far we have not specified how the LLM should answer so it will be a free text answer. The first thing we can now adjust is to simply prompt the model to only respond in a very specific way. SurveyGen allows us to do this in a controlled manner, so we can later easily adjust our prompts or our response generation methods.


```python
from surveygen import survey_manager

likert_scale = [
    "Strongly Dislike",
    "Dislike",
    "Neiter Dislike nor Like",
    "Like",
    "Strongly Like",
]

options = survey_manager.SurveyOptionGenerator.generate_likert_options(
    n=5,
    answer_texts=likert_scale,
    random_order=False,
    reversed_order=False,
    idx_type="integer",
    options_separator="|",
    only_from_to_scale=False,
    start_idx=1,
    list_prompt_template="Only respond with one of the following {options}.",
)
```

With SurveyGen we can easily create a likert scala. If we want to control for prompt pertubations, for example random or reverse order of options, we simply set a flag to change them. For now let's keep the options as they are. To include these options in the prompt we simply can use the ``prepare_interview`` function of ``LLMInterview`` to specify how our questions should be asked. We can specify placeholders to define at which point in the prompt the options or questions should be specified. Note that ``placeholder.PROMPT_OPTIONS`` can also be specified in the system prompt or prompt above, if we do want to place it independently from the questions. Again we can simply set a flag, if we want our questions to be in a random order.

```python
from surveygen.utilities import placeholder

interview.prepare_interview(
    question_stem=f"How do you feel towards {placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}",
    answer_options=options,
    randomized_item_order=False,
)
```

Let's take a look at our current prompts. Depending on how we conduct the interview, the prompts will contain all questions or just one of them. For now let's look at the prompt if we ask each question in a new context.

```python
system_prompt, prompt = interviews[0].get_prompt_for_interview_type()
print(system_prompt)
print(prompt)
```

```txt
Act as if you were a black middle aged man from New York! Answer in a single short sentence.
Please tell us how you feel about the following parties:
How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike|2: Dislike|3: Neiter Dislike nor Like|4: Like|5: Strongly Like.
```

Okay, now we prompt the model to only respond with one of the possible answers. Let's see how a small Llama model responds!


```python
from vllm import LLM

model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = LLM(model_id, max_model_len=5000, seed=42)

from surveygen import survey_manager


results = survey_manager.conduct_survey_question_by_question(
    model,
    interview,
    client_model_name=model_id,
    print_conversation=True,
    temperature=0.8,
    max_tokens=5000,
)
```

We get the following output for our "survey participants".

|interview_name|interview_item_id|question                                                                                                                                                                        |llm_response                                                                               |logprobs|reasoning|
|--------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------|---------|
|New York      |2                |How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|Ain't no party for me, I go with Neutral Dislike.                                          |        |         |
|New York      |1                |How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|Ain't no party I like more than da Democratic Party, we stand up for da people, ya hear me?|        |         |
|Texas         |2                |How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|I'd say I Like it, partner, 'cause we seem to be alignin' on a whole lotta issues.         |        |         |
|Texas         |1                |How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|I'd say 3: Neither Like nor Dislike.                                                       |        |         |


Well, this is not what we wanted. The small model took the "Only answer with exactly" quite liberally, sometimes not even including one of the labels, making it quite hard to interpret the output. We could now go three ways, if we do not want to involve manual labor or finetuning: 
1. *Try to find a different prompt.* Generally this is not preferable, because while it might be possible to find a prompt that works for this specific case, it might not work for a different model/different setting.
2. *Use LLM-as-a-judge.* We can give the output to another LLM and try to get our labels this way.
3. *Use Guided Decoding.* We can restrict the models output to ensure that our models only respond in a certain way.

SurveyGen supports all of these methods. For now, let's take a look at restricting the models output. For this we can simply give another option to the ``SurveyOptionGenerator`` we used before and prepare the interviews again.

```python
from surveygen.inference import ChoiceResponseGenerationMethod
from surveygen import utilities

choice_rgm = ChoiceResponseGenerationMethod(utilities.constants.OPTIONS_ADJUST, output_index_only=False)

options = survey_manager.SurveyOptionGenerator.generate_likert_options(
    n=5,
    answer_texts=likert_scale,
    response_generation_method=choice_rgm,
    random_order=False,
    reversed_order=False,
    idx_type="integer",
    options_separator="|",
    only_from_to_scale=False,
    start_idx=1,
    list_prompt_template="Only respond with exactly one of the following {options}.",
)

for interview in interviews:
    interview.prepare_interview(
        question_stem=f"How do you feel towards {placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}",
        answer_options=options,
        randomized_item_order=True,
    )
```

We get the following output:

|interview_name|interview_item_id|question                                                                                                                                                                        |llm_response                                                                               |logprobs|reasoning|
|--------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------|---------|
|New York      |1                |How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|2: Dislike                                                                                 |        |         |
|New York      |2                |How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|2: Dislike                                                                                 |        |         |
|Texas         |1                |How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|2: Dislike                                                                                 |        |         |
|Texas         |2                |How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|3: Neiter Dislike nor Like                                                                 |        |         |

While we now get exactly the output format that we specified, the answers themselves are not really reflecting the answers that we got from our free text responses from before. One way to get more meaningful output, especially with smaller models, is to first let the model generate a reasoning string and only then answer with one of the options. If we still want to easily parse the answers, we can instruct and restrict the model to only answer in JSON.


For this we have to slightly modify our prompt, but SurveyGen offers easy automatic adjustment depending on the output method via placeholders:

```python
# We now use a new placeholder to automatically adjust for our output format
for interview in interviews:
    interview.prompt = f"Please tell us how you feel about the following parties:\n{placeholder.PROMPT_QUESTIONS}\n{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"


# Define our Response Generation Method (If we want to, we can also adjust our automatic output instructions here)
reasoning_rgm = JSONReasoningResponseGenerationMethod()

# Adjust the options to now include the reasoning
options = survey_manager.SurveyOptionGenerator.generate_likert_options(
    n=5,
    answer_texts=likert_scale,
    response_generation_method=reasoning_rgm,
    random_order=False,
    reversed_order=False,
    idx_type="integer",
    options_separator="|",
    only_from_to_scale=False,
    start_idx=1,
    list_prompt_template="These are the {options}.",
)

for interview in interviews:
    interview.prepare_interview(
        question_stem=f"How do you feel towards {placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}",
        answer_options=options,
        randomized_item_order=True,
    )

# Finally we get the following prompts
system_prompt, prompt = interviews[0].get_prompt_for_interview_type()
print(system_prompt)
print(prompt)
```

```txt
Act as if you were a black middle aged man from New York!
Please tell us how you feel about the following parties:
How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike|2: Dislike|3: Neiter Dislike nor Like|4: Like|5: Strongly Like.
You always reason about the possible answer options first.
You respond with your reasoning and the most probable answer option in the following JSON format:
```json
{
  "reasoning": <your reasoning about the answer options>,
  "answer": <1: Strongly Dislike, 2: Dislike, 3: Neiter Dislike nor Like, 4: Like, 5: Strongly Like>
}
```

We run our survey again:

```python
results = survey_manager.conduct_survey_question_by_question(
    model,
    interviews,
    client_model_name=model_id,
    print_conversation=True,
    temperature=0.8,
    max_tokens=5000,
)
```

SurveyGen allows us to easily parse the json output of the model:

```python
from surveygen import parser
parsed_results = parser.json_parse_all(results)
df = create_one_dataframe(parsed_results)
```

And now we get answers that are more aligned with the free text answers from before but also easily parsable responses. If you are interested in which Response Generation Methods reflect public opinion most closely and also which are most efficient to use, we encourage you to check out this paper: [Survey Response Generation: Generating Closed-Ended Survey Responses In-Silico with Large Language Models](https://arxiv.org/abs/2510.11586).

|interview_name|interview_item_id|question                                                                                                                                                                        |reasoning                                                                                  |answer                    |
|--------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------------------------|
|New York      |2                |How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|The Republican Party's views on issues like income inequality and access to healthcare don't align with my experiences and values as a black New Yorker, making it unlikely I'd support them strongly, but I also don't feel strongly opposed to their existence.|3: Neiter Dislike nor Like|
|New York      |1                |How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|The Democratic Party has historically represented the values and priorities of my community, particularly on issues like social justice, education, and economic equality, making it more likely that I'd have a strong positive association with the party.|5: Strongly Like          |
|Texas         |2                |How do you feel towards The Republican Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|The Republican Party aligns more closely with my conservative values and the politics of Texas, where I reside, so I'd likely favor the party.|4: Like                   |
|Texas         |1                |How do you feel towards The Democratic Party? Only respond with exactly one of the following 1: Strongly Dislike&#124;2: Dislike&#124;3: Neiter Dislike nor Like&#124;4: Like&#124;5: Strongly Like.|The Democratic Party's views on issues like gun control, climate change, and taxation are often at odds with my own conservative values, so I'm leanin' towards dislikin' 'em, but not strongly dislikin' 'em, since I still believe we can have a civil discussion about our differences|2: Dislike                |
