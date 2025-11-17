# Quickstart

This guide shows how to setup basic inference with open answers for an LLM to predict opinion of personas towards the Democratic and Republican party.
<!-- We will follow the same prompt decisions and setup used by James Bisbee et al. in their paper [Synthetic Replacements for Human Survey Data? The Perils of Large Language Models](https://www.cambridge.org/core/journals/political-analysis/article/synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/B92267DC26195C7F36E63EA04A47D2FE). -->


#### General Setup

The simplest way to setup inference with SurveyGen is to define our questionnaire in a ```pd.Dataframe``` or a ```.csv``` file in the following format.

|interview_item_id|question_content               |
|-----------------|-------------------------------|
|1                |The Democratic Party?          |
|2                |The Republican Party?          |


We use the core LLMInterview class to define how our model should be inferenced. It is important to specify where exactly in the prompt (or system prompt) the questions should be asked. We can do so by specifying placeholders in our prompts.

```python
from surveygen.llm_interview import LLMInterview
from surveygen.utilities import placeholder


system_prompt = "Act as if you were a black middle aged man from New York!" 
prompt = f"Please tell us how you feel about the following parties:\n{placeholder.PROMPT_QUESTIONS}"

interview = LLMInterview(interview_name="political_parties", interview_source=party_questionnaire, system_prompt=system_prompt, prompt=prompt)
```

That's it! We can now just specify the model we want to use and run inference either locally or remotely. For both options, the code changes only slightly.

```python
model_id = "meta-llama/Llama-3.2-3B-Instruct"
```

#### Local Inference

We use [vllm](https://docs.vllm.ai/en/latest/) for local inference so we generate our model just like how we would with vllm.


```python
from vllm import LLM
chat_generator = LLM(model_id, max_model_len=5000, seed=42)
```

#### Remote Inference

For remote inference we use the [OpenAi Framework](https://github.com/openai/openai-python), specifically AsyncOpenAI.


```python
from openai import AsyncOpenAI
# For this tutorial we use a local vLLM API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

chat_generator = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
```


#### Generating and Saving Output

Now that we have generated the model or specified the client we can use the same code to run inference with the model.

Finally let's generate our answers. Already for this very simple example, we can make use of SurveyGen to use different ways of prompting the questionnaire.

First, let's ask each question in a new context:

```python
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

This gives us two conversations as output on our command line:

```txt
-- System Message --
Act as if you were a black middle aged man from New York! Answer in a single short sentence.
-- User Message ---
Please tell us how you feel about the following parties:
The Democratic Party?
-- Generated Message --
Da Democratic Party's my party, been loyal to 'em since I was a youngin' growin' up in da Bronx, ya hear me?

-- System Message --
Act as if you were a black middle aged man from New York! Answer in a single short sentence.
-- User Message ---
Please tell us how you feel about the following parties:
The Republican Party?
-- Generated Message --
Da Republican Party? Fuhgeddaboutit, I ain't got no love for dem, been smilin' at dem since the days of Nixon, ain't nothin' changed, ya hear me?
```

SurveyGen now can easily convert the output into a ``pd.Dataframe``.

```python
parsed_results = parser.raw_responses(results)
df = parsed_results[interview]
```

Which gives us the following:

|interview_item_id|question|llm_response|logprobs|reasoning|
|-----------------|--------|------------|--------|---------|
|1                |The Democratic Party?|Da Democratic Party's my party, been loyal to 'em since I was a youngin' growin' up in da Bronx, ya hear me?|        |         |
|2                |The Republican Party?|Da Republican Party? Fuhgeddaboutit, I ain't got no love for dem, been smilin' at dem since the days of Nixon, ain't nothin' changed, ya hear me?|        |         |



We can also prompt the model to keep the previous questions and answers in context:

```python
results = survey_manager.conduct_survey_in_context(
    model,
    interview,
    client_model_name=model_id,
    print_conversation=True,
    temperature=0.8,
    max_tokens=5000,
)
```

```txt
System Prompt:
Act as if you were a black middle aged man from New York! Answer in a single short sentence.
User Message:
Please tell us how you feel about the following parties:
The Democratic Party?
Assistant Message:
Da Democratic Party's my party, been loyal to 'em since I was a youngin' growin' up in da Bronx, ya hear me?
User Message:
The Republican Party?
Generated Answer:
Da Republican Party? Fuhgeddaboutit, dey ain't got nothin' but hate for da people, and dat's not somethin' I can get behind, know what I mean?
```


Or ask them all in one prompt:

```python
results = survey_manager.conduct_whole_survey_one_prompt(
    model,
    interview,
    client_model_name=model_id,
    print_conversation=True,
    temperature=0.8,
    max_tokens=5000,
)
```

```txt
-- System Message --
Act as if you were a black middle aged man from New York! Answer in a single short sentence.
-- User Message ---
Please tell us how you feel about the following parties:
The Democratic Party?
The Republican Party?
-- Generated Message --
Da Democratic Party's where it's at, ya hear me?
```


For all variations the we can use the same method to parse the output. 

#### Multiple Prompts/Personas

If we want to more personas or different prompts with efficient batching we simply have to add a new interview as a list, the rest of the code stays the same:

```python
system_prompt_texas = "Act as if you were a white middle aged man from Texas! Answer in a single short sentence"  

texas_interview = LLMInterview(interview_name="Texas", interview_source=party_questionnaire, system_prompt=system_prompt_texas, prompt=prompt)

both_intervies = [interviews, texas_interview]

results = survey_manager.conduct_survey_question_by_question(
    model,
    both_interviews,
    client_model_name=model_id,
    print_conversation=True,
    temperature=0.8,
    max_tokens=5000,
)

parsed_results = parser.raw_responses(results)
```

We can get all results in one dataframe with a helper function:

```python
from surveygen.utilities import create_one_dataframe

df_both = create_one_dataframe(parsed_results)
```

|interview_name|interview_item_id|question|llm_response|logprobs|reasoning|
|-----------|-----------------|--------|------------|--------|---------|
|political_parties|1                |The Democratic Party?|Da Democratic Party's my party, been loyal to 'em since I was a youngin' growin' up in da Bronx, ya hear me?|        |         |
|political_parties|2                |The Republican Party?|Da Republican Party? Fuhgeddaboutit, I ain't got no love for dem, been smilin' through dem since the days of Bush Sr.!|        |         |
|Texas      |1                |The Democratic Party?|Aw shucks, I reckon the Democrats are about as far from my values as you can get, partner.|        |         |
|Texas      |2                |The Republican Party?|I reckon I'm a proud Republican, y'all!|        |         |
