from typing import List, Dict, Optional

from ..llm_interview import LLMInterview
from ..utilities.survey_objects import InterviewResult

from ..inference.survey_inference import batch_generation, AnswerProductionMethod

from ..utilities import constants

from vllm import LLM

import pandas as pd
import numpy as np

import json

import json_repair

import re

from collections import defaultdict
import warnings


DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_PROMPT: str = (
    "Your task is to parse the correct answer option from an open text answer a LLM has given to survey questions. You will be provided with the survey question, possible answer options and the LLM answer. Answer ONLY and EXACTLY with one of the possible answer options or 'INVALID', if the provided LLM answer does give one of the options."
)


def json_parser_str(answer: str) -> Dict[str, str] | None:
    try:
        result_json = json.loads(answer)
    except:
        try:
            result_json = json_repair.loads(answer, skip_json_loads=True)
        except:
            return None

    return result_json


def json_parse_all(survey_results: List[InterviewResult]) -> Dict[LLMInterview, pd.DataFrame]:
    final_result = {}

    for survey_result in survey_results:
        answers = []
        for key, value in survey_result.results.items():
            # value:QuestionAnswerTuple
            parsed_llm_response = json_parser_str(value.llm_response)
            if parsed_llm_response is not None:
                answer_format = parsed_llm_response.keys()
                answers.append((key, value.question, *parsed_llm_response.values()))
            else:
                answers.append(
                    (key, value.question, value.llm_response, "ERROR: Parsing")
                )
        try:
            df = pd.DataFrame(
                answers,
                columns=[constants.INTERVIEW_ITEM_ID, constants.QUESTION, *answer_format],
            )
        except:
            print(answers)
            df = pd.DataFrame(
                answers,
                columns=[
                    constants.INTERVIEW_ITEM_ID,
                    constants.QUESTION,
                    constants.LLM_RESPONSE,
                    "error_col",
                ],
            )
        final_result[survey_result.interview] = df

    return final_result


def json_parse_whole_survey_all(
    survey_results: List[InterviewResult],
) -> Dict[LLMInterview, pd.DataFrame]:
    parsed_results = json_parse_all(survey_results)

    all_results = {}

    for survey, df in parsed_results.items():

        if "error_col" in df.columns:
            all_results[survey] = df
            continue

        pattern = re.compile(r"^([a-zA-Z_]+)(\d+)$")
        matched = [pattern.match(col) for col in df.columns]

        stubnames = set()
        reshape_cols = []
        static_cols = []
        seen_stubs = set()
        stubnames = []

        for col, m in zip(df.columns, matched):
            if m:
                stub = m.group(1)
                reshape_cols.append(col)
                if stub not in seen_stubs:
                    stubnames.append(stub)
                    seen_stubs.add(stub)
            else:
                static_cols.append(col)

        long_df = pd.wide_to_long(
            df,
            stubnames=stubnames,
            i=[constants.INTERVIEW_ITEM_ID],
            j="new_survey_item_id",
            sep="",
            suffix="\\d+",
        ).reset_index()

        num_rows_to_update = len(long_df)

        long_df.loc[0:num_rows_to_update, constants.INTERVIEW_ITEM_ID] = [
            survey_question.item_id
            for survey_question in survey._questions[0:num_rows_to_update]
        ]
        long_df.loc[0:num_rows_to_update, constants.QUESTION] = [
            survey.generate_question_prompt(survey_question)
            for survey_question in survey._questions[0:num_rows_to_update]
        ]
        long_df = long_df.drop(columns=constants.INTERVIEW_ITEM_ID).rename(
            columns={"new_survey_item_id": constants.INTERVIEW_ITEM_ID}
        )
        all_results[survey] = long_df

    return all_results


def raw_responses(survey_results: List[InterviewResult]) -> Dict[LLMInterview, pd.DataFrame]:
    all_results = {}
    for survey_result in survey_results:
        all_results[survey_result.interview] = survey_result.to_dataframe()
    return all_results


# def llm_parse_all(model:LLM, survey_results:List[SurveyResult], system_prompt:str = DEFAULT_SYSTEM_PROMPT, prompt:str = DEFAULT_PROMPT, use_structured_ouput:bool = False, seed = 42, **generation_kwargs) -> Dict[LLMSurvey, pd.DataFrame]:
#     #TODO LLM Parser in batches, same output as json parser
#     all_results = {}
#     for survey_result in survey_results:
#         prompts = []
#         ids = []
#         questions = []
#         answers = []
#         for item_id, question_llm_response_tuple in survey_result.results.items():
#             ids.append(item_id)
#             questions.append(question_llm_response_tuple.question)
#             answers.append(question_llm_response_tuple.llm_response)
#             prompts.append(f"{prompt} \nQuestion: {question_llm_response_tuple.question} \nResponse by LLM: {question_llm_response_tuple.llm_response}")
#         llm_parsed_results = batch_generation(model, system_messages=[system_prompt] * len(prompts), prompts=prompts, seed=seed, **generation_kwargs)


#         all_results[survey_result.survey] = pd.DataFrame(zip(ids, questions, answers, llm_parsed_results), columns=[constants.SURVEY_ITEM_ID, constants.QUESTION, constants.LLM_RESPONSE, constants.PARSED_RESPONSE])

#     return all_results


def llm_parse_all(
    model: LLM,
    survey_results: List[InterviewResult],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt: str = DEFAULT_PROMPT,
    answer_production_method: Optional[AnswerProductionMethod] = None,
    seed=42,
    **generation_kwargs,
) -> Dict[LLMInterview, pd.DataFrame]:
    all_items_to_process = []
    for survey_result in survey_results:
        for item_id, question_llm_response_tuple in survey_result.results.items():
            all_items_to_process.append(
                {
                    constants.INTERVIEW_NAME: survey_result.interview,
                    constants.INTERVIEW_ITEM_ID: item_id,
                    constants.QUESTION: question_llm_response_tuple.question,
                    constants.LLM_RESPONSE: question_llm_response_tuple.llm_response,
                    "prompt": f"{prompt} \nQuestion: {question_llm_response_tuple.question} \nResponse by LLM: {question_llm_response_tuple.llm_response}",
                }
            )

    if not all_items_to_process:
        all_results = {}
    # or handle as you see fit, e.g., return {}
    else:
        # 2. BATCH: Prepare prompts for a single batch generation call.
        all_prompts = [item["prompt"] for item in all_items_to_process]
        system_messages = [system_prompt] * len(all_prompts)

        # Perform the single, efficient batch inference.
        llm_parsed_results, logprobs = batch_generation(
            model,
            system_messages = system_messages,
            prompts = all_prompts,
            answer_production_method = answer_production_method,
            seed = seed,
            **generation_kwargs,
        )

    for item, parsed_result in zip(all_items_to_process, llm_parsed_results):
        item[constants.PARSED_RESPONSE] = parsed_result

    # Group the results by survey_name to build the final DataFrames.
    # defaultdict is perfect for this task.
    grouped_data = defaultdict(list)
    for item in all_items_to_process:
        grouped_data[item[constants.INTERVIEW_NAME]].append(
            {
                constants.INTERVIEW_ITEM_ID: item[constants.INTERVIEW_ITEM_ID],
                constants.QUESTION: item[constants.QUESTION],
                constants.LLM_RESPONSE: item[constants.LLM_RESPONSE],
                constants.PARSED_RESPONSE: item[constants.PARSED_RESPONSE],
            }
        )
    all_results = {
        survey_name: pd.DataFrame(data_list)
        for survey_name, data_list in grouped_data.items()
    }

    return all_results


def _filter_logprobs_by_choices(
        logprob_df: pd.DataFrame,
        choices: pd.Series
    ) -> pd.DataFrame:

    matches_found = []
    
    # check for each output token whether any of the choices start with this token
    for token in logprob_df['token']:
        boolean_index = choices.str.startswith(token)
        if len(choices[boolean_index]) > 1:
            warnings.warn(
                f"Multiple allowed_choices ({list(choices[boolean_index])}) match the same output token: {token}",
                stacklevel=2
            )
        matches_found.append(boolean_index.any())
    
    return logprob_df[matches_found]


def _logprobs_filter(
        logprobs: Dict[str, float],
        allowed_choices: Dict[str, List[str]]
    ) -> Dict[str, float]:
    
    # normalize logprobs
    logprob_df = pd.DataFrame({'token': logprobs.keys(), 'prob': logprobs.values()})
    logprob_df['prob'] = logprob_df.prob.apply(np.exp)
    logprob_df = logprob_df[logprob_df.prob > 0]

    # flatten to check for collisions between answer options
    all_valid_outputs = [output for choices in allowed_choices.values() for output in choices]
    _ = _filter_logprobs_by_choices(logprob_df, pd.Series(all_valid_outputs))

    # filter the individual survey answers
    choice_results = {}
    for choice, valid_outputs in allowed_choices.items():
        valid_logprobs = _filter_logprobs_by_choices(logprob_df, pd.Series(valid_outputs))
        if len(valid_logprobs) == 0:
            warnings.warn(f"Could not find logprobs for answer option {choice} with possible outputs {valid_outputs}")
        choice_results[choice] = valid_logprobs['prob'].sum()
    
    # normalize so that probs sum up to 1
    overall_sum = sum(choice_results.values())
    choice_results = {choice: token_sum/overall_sum for choice, token_sum in choice_results.items()}

    return choice_results


def logprobs_parse_all(
        survey_results: List[InterviewResult],
        allowed_choices: List[str] | Dict[str, List[str]]
    ) -> Dict[LLMInterview, pd.DataFrame]:
    """
    Filter and aggregate the logprobs that are returned when using the Logprob_AnswerProductionMethod

    Args:
        survey_results: List of InterviewResult that is returned from running a survey
        allowed_choices: List of possible answer options OR dictionary that maps answer options to multiple tokens that encode each option
    """
    final_result = {}

    # if each choice only maps to one token
    if isinstance(allowed_choices, list):
        allowed_choices = {c: [c] for c in allowed_choices}

    for survey_result in survey_results:
        answers = []
        for item_id, qa_tuple in survey_result.results.items():
            if qa_tuple.logprobs is None:
                warnings.warn(
                    "No logprobs found in InterviewResult. " + \
                    "Make sure to use Logprob_AnswerProductionMethod to generate logprobs.",
                    stacklevel = 2
                )
                answer_format = ["error_col"]
                answers.append((item_id, qa_tuple.question, "ERROR: Parsing"))
            else:
                filtered_logprobs = _logprobs_filter(qa_tuple.logprobs, allowed_choices)
                answer_format = filtered_logprobs.keys()
                answers.append((item_id, qa_tuple.question, *filtered_logprobs.values()))
        
            df = pd.DataFrame(
                    answers,
                    columns=[constants.INTERVIEW_ITEM_ID, constants.QUESTION, *answer_format],
                )
            final_result[survey_result.interview] = df

    return final_result