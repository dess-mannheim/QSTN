# QSTN: A Modular Framework for Robust Questionnaire Inference with Large Language Models
    
<div align="center">

![Overview](overview.svg)

</div>

  

QSTN is a Python framework designed to facilitate the creation of robust inference experiments with Large Language Models based around questionnaires. It provides a full pipeline from perturbation of prompts, to choosing Response Generation Methods, inferencing and finally parsing of the output. QSTN supports both local inference with vllm and remote inference via the OpenAI API.

Detailed information and guides are available in our [documentation](https://qstn.readthedocs.io/en/latest/). Tutorial notebooks can also be found in this [repository](https://github.com/dess-mannheim/QSTN/tree/main/docs/guides).

## Installation

To install the project and dependencies you can use `pip`.

```bash
pip install qstn
```

Or install this package from source:

```bash
pip install git+https://github.com/dess-mannheim/QSTN.git
```

## Getting Started

Below you can find a minimum working example of how to use QSTN. It can be easily integrated into existing projects, requiring just three function calls to operate. Users familiar with vllm or the OpenAI API can use the same Model/Client calls and arguments. In this example reasoning and the generated response are automatically parsed. For more elaborate examples, see the [tutorial notebooks](https://github.com/dess-mannheim/QSTN/tree/main/docs/guides).

```python
import qstn
import pandas as pd
from vllm import LLM

# 1. Prepare questionnaire and persona data
questionnaires = pd.read_csv("hf://datasets/qstn/ex/q.csv")
personas = pd.read_csv("hf://datasets/qstn/ex/p.csv")
prompt = (
    f"Please tell us how you feel about:\n"
    f"{qstn.utilities.placeholder.PROMPT_QUESTIONS}"
)
interviews = [
    qstn.prompt_builder.LLMPrompt(
        questionnaire_source=questionnaires,
        system_prompt=persona,
        prompt=prompt,
    ) for persona in personas.system_prompt]

# 2. Run Inference
model = LLM("Qwen/Qwen3-4B", max_model_len=5000)
results = qstn.survey_manager.conduct_survey_single_item(
    model, interviews, max_tokens=500
)

# 3. Parse Results
parsed_results = qstn.parser.raw_responses(results)
```

## Citation

Authors: Maximilian Kreutner, Jens Rupprecht, Georg Ahnert, Ahmed Salem, and Markus Strohmaier

This package will soon have a arxiv paper.
