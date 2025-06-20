
from typing import List, Dict

from ..survey_manager import SurveyResult, LLMSurvey

from ..inference.survey_inference import batch_generation

from ..utilities import constants

from vllm import LLM

import pandas as pd

import json

import re

from collections import defaultdict


DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_PROMPT: str = "Your task is to parse the correct answer option from an open text answer a LLM has given to survey questions. You will be provided with the survey question, possible answer options and the LLM answer. Answer ONLY and EXACTLY with one of the possible answer options or 'INVALID', if the provided LLM answer does give one of the options."

def json_parser_str(answer:str) -> Dict[str, str]:
    try:
        result_json = json.loads(answer)
    except:
        return None

    return result_json

def json_parse_all(survey_results: List[SurveyResult]) -> Dict[LLMSurvey, pd.DataFrame]:
    final_result = {}

    for survey_result in survey_results:
        answers = []
        for key, value in survey_result.results.items():
            #value:QuestionAnswerTuple
            parsed_llm_response = json_parser_str(value.llm_response)
            if parsed_llm_response:
                answer_format = parsed_llm_response.keys()
                answers.append((key, value.question, *parsed_llm_response.values()))
            else:
                answers.append((key, value.question, value.llm_response, "ERROR: Parsing"))
        try:
            df = pd.DataFrame(answers, columns=[constants.SURVEY_ITEM_ID, constants.QUESTION, *answer_format])
        except:
            print(answers)
            df = pd.DataFrame(answers, columns=[constants.SURVEY_ITEM_ID, constants.QUESTION, constants.LLM_RESPONSE, "error_col"])
        final_result[survey_result.survey] = df

    return final_result

def json_parse_whole_survey_all(survey_results:List[SurveyResult]) -> Dict[LLMSurvey, pd.DataFrame]:
    parsed_results =  json_parse_all(survey_results)

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

        long_df = pd.wide_to_long(df, 
                                stubnames=stubnames,
                                i=[constants.SURVEY_ITEM_ID],
                                j='new_survey_item_id', 
                                sep='', 
                                suffix='\\d+').reset_index()
        
        num_rows_to_update = len(long_df)
        
        long_df.loc[0:num_rows_to_update, constants.SURVEY_ITEM_ID] = [survey_question.item_id for survey_question in survey._questions[0:num_rows_to_update]]
        long_df.loc[0:num_rows_to_update, constants.QUESTION] = [survey.generate_question_prompt(survey_question) for survey_question in survey._questions[0:num_rows_to_update]]
        long_df = long_df.drop(columns=constants.SURVEY_ITEM_ID).rename(columns={'new_survey_item_id': constants.SURVEY_ITEM_ID})
        all_results[survey] = long_df
    
    return all_results

def raw_responses(survey_results:List[SurveyResult])-> Dict[LLMSurvey, pd.DataFrame]:
    all_results = {}
    for survey_result in survey_results:
        all_results[survey_result.survey] = survey_result.to_dataframe()
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


def llm_parse_all(model:LLM, survey_results:List[SurveyResult], system_prompt:str = DEFAULT_SYSTEM_PROMPT, prompt:str = DEFAULT_PROMPT, use_structured_ouput:bool = False, seed = 42, **generation_kwargs) -> Dict[LLMSurvey, pd.DataFrame]:
    all_items_to_process = []
    for survey_result in survey_results:
        for item_id, question_llm_response_tuple in survey_result.results.items():
            all_items_to_process.append({
                constants.SURVEY_NAME: survey_result.survey,
                constants.SURVEY_ITEM_ID: item_id,
                constants.QUESTION: question_llm_response_tuple.question,
                constants.LLM_RESPONSE: question_llm_response_tuple.llm_response,
                'prompt': f"{prompt} \nQuestion: {question_llm_response_tuple.question} \nResponse by LLM: {question_llm_response_tuple.llm_response}"
            })

    if not all_items_to_process:
        all_results = {}
    # or handle as you see fit, e.g., return {}
    else:
        # 2. BATCH: Prepare prompts for a single batch generation call.
        all_prompts = [item['prompt'] for item in all_items_to_process]
        system_messages = [system_prompt] * len(all_prompts)

        # Perform the single, efficient batch inference.
        llm_parsed_results = batch_generation(
            model,
            system_messages=system_messages,
            prompts=all_prompts,
            seed=seed,
            **generation_kwargs
        )

    for item, parsed_result in zip(all_items_to_process, llm_parsed_results):
        item[constants.PARSED_RESPONSE] = parsed_result

    # Group the results by survey_name to build the final DataFrames.
    # defaultdict is perfect for this task.
    grouped_data = defaultdict(list)
    for item in all_items_to_process:
        grouped_data[item[constants.SURVEY_NAME]].append({
            constants.SURVEY_ITEM_ID: item[constants.SURVEY_ITEM_ID],
            constants.QUESTION: item[constants.QUESTION],
            constants.LLM_RESPONSE: item[constants.LLM_RESPONSE],
            constants.PARSED_RESPONSE: item[constants.PARSED_RESPONSE]
        })
    all_results = {
        survey_name: pd.DataFrame(data_list)
        for survey_name, data_list in grouped_data.items()
    }

    return all_results