from typing import List, Dict, Optional

from ..prompt_builder import LLMPrompt
from ..utilities.survey_objects import InferenceResult

from ..inference.survey_inference import batch_generation, ResponseGenerationMethod

from ..utilities import constants

#from vllm import LLM

import pandas as pd
import numpy as np

import json

import json_repair


from collections import defaultdict
import warnings


DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_PROMPT: str = (
    "Your task is to parse the correct answer option from an open text "
    + "answer a LLM has given to survey questions. You will be provided with the survey question, "
    + "possible answer options and the LLM answer. Answer ONLY and EXACTLY with one of the possible "
    + "answer options or 'INVALID', if the provided LLM answer does give one of the options. \n"
    + "Question: {question} \nResponse by LLM: {llm_response}"
)


def parse_json_str(answer: str) -> Dict[str, str] | None:
    try:
        result_json = json.loads(answer)
    except:
        try:
            result_json = json_repair.loads(answer, skip_json_loads=True)
        except:
            return None

    return result_json


def parse_json(
    survey_results: List[InferenceResult],
) -> Dict[LLMPrompt, pd.DataFrame]:
    """Parses json output of a survey of LLMs.
    Args:
        survey_results List[InterviewResult]: All results for all interviews.

    Returns:
        Dict[LLMInterview]: A dictionary where the keys are the LLMInterviews and the values are a Dataframe with the questions and answers.
    """
    final_result = {}

    for survey_result in survey_results:
        answers: List[pd.DataFrame] = []
        for key, value in survey_result.results.items():
            # value:QuestionAnswerTuple
            parsed_llm_response = parse_json_str(value.llm_response)
            reasoning = value.reasoning
            logprobs = value.logprobs
            if isinstance(parsed_llm_response, dict):
                # remove reserved values from dictionary
                for reserved_key in [constants.QUESTIONNAIRE_ITEM_ID, constants.QUESTION]:
                    if reserved_key in parsed_llm_response:
                        parsed_llm_response.pop(reserved_key)
                answer_format = parsed_llm_response.keys()

                row_data = [key, value.question, *parsed_llm_response.values()]
                row_columns = [
                    constants.QUESTIONNAIRE_ITEM_ID,
                    constants.QUESTION,
                    *answer_format,
                ]

                if reasoning is not None:
                    row_data.append(reasoning)
                    row_columns.append("built_in_reasoning")

                if logprobs is not None:
                    row_data.append(logprobs)
                    row_columns.append("logprobs")

                answers.append(
                    pd.DataFrame(data=[row_data], columns=row_columns, index=[0])
                )
            else:
                answers.append(
                    pd.DataFrame(
                        data=[
                            (key, value.question, value.llm_response, "ERROR: Parsing")
                        ],
                        columns=[
                            constants.QUESTIONNAIRE_ITEM_ID,
                            constants.QUESTION,
                            constants.LLM_RESPONSE,
                            "error_col",
                        ],
                        index=[0],
                    )
                )
        final_result[survey_result.questionnaire] = pd.concat(
            answers, ignore_index=True
        )  # handles inconsistent columns

    return final_result


def parse_json_battery(
    survey_results: List[InferenceResult],
) -> Dict[LLMPrompt, pd.DataFrame]:
    """Parses json output of a survey of LLMs when prompted with one prompt.
    Args:
        survey_results List[InterviewResult]: All results for all interviews.

    Returns:
        Dict[LLMInterview]: A dictionary where the keys are the LLMInterviews and the values are a Dataframe with the questions and answers.
    """
    parsed_results: Dict[LLMPrompt, pd.DataFrame] = parse_json(survey_results)

    all_results = {}

    for survey, df in parsed_results.items():

        if "error_col" in df.columns:
            all_results[survey] = df
            continue

        source_row = df.iloc[0]

        grouped_items = {}

        for col_name, cell_value in source_row.items():
            for i in range(len(survey._questions)):
                current_question = survey._questions[i]
                current_id = current_question.item_id
                if col_name.endswith(f"_{current_question.question_content}"):
                    new_col_name = col_name.removesuffix(
                        f"_{current_question.question_content}"
                    )
                    if current_id not in grouped_items:
                        grouped_items[current_id] = {
                            constants.QUESTIONNAIRE_ITEM_ID: current_id
                        }
                    grouped_items[current_id][constants.QUESTION] = (
                        survey.generate_question_prompt(current_question)
                    )
                    grouped_items[current_id][new_col_name] = cell_value

            final_data_list = list(grouped_items.values())

        all_results[survey] = pd.DataFrame(final_data_list)

        # long_df.loc[0:minimum_rows, constants.INTERVIEW_ITEM_ID] = [
        #     survey_question.item_id
        #     for survey_question in survey._questions[0:minimum_rows]
        # ]
        # long_df.loc[0:minimum_rows, constants.QUESTION] = [
        #     survey.generate_question_prompt(survey_question)
        #     for survey_question in survey._questions[0:minimum_rows]
        # ]
        # long_df = long_df.drop(columns=constants.INTERVIEW_ITEM_ID).rename(
        #     columns={"new_survey_item_id": constants.INTERVIEW_ITEM_ID}
        # )
        # all_results[survey] = long_df

    return all_results


def raw_responses(
    survey_results: List[InferenceResult],
) -> Dict[LLMPrompt, pd.DataFrame]:
    """Organizes the questions and answers of a survey in a pandas Dataframe.
    Args:
        survey_results List[InterviewResult]: All results for all interviews.

    Returns:
        Dict[LLMInterview, pd.Dataframe]: A dictionary where the keys are the LLMInterviews and the values are a Dataframe with the questions and answers.
    """

    all_results = {}
    for survey_result in survey_results:
        all_results[survey_result.questionnaire] = survey_result.to_dataframe()
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


# def parse_with_llm(
#     model: LLM,
#     survey_results: List[InferenceResult],
#     system_prompt: str = DEFAULT_SYSTEM_PROMPT,
#     prompt: str = DEFAULT_PROMPT,
#     answer_production_method: Optional[ResponseGenerationMethod] = None,
#     print_conversation: bool = False,
#     print_progress: bool = True,
#     seed=42,
#     **generation_kwargs,
# ) -> Dict[LLMPrompt, pd.DataFrame]:
#     all_items_to_process = []
#     for survey_result in survey_results:
#         for item_id, question_llm_response_tuple in survey_result.results.items():
#             all_items_to_process.append(
#                 {
#                     constants.QUESTIONNAIRE_NAME: survey_result.questionnaire,
#                     constants.QUESTIONNAIRE_ITEM_ID: item_id,
#                     constants.QUESTION: question_llm_response_tuple.question,
#                     constants.LLM_RESPONSE: question_llm_response_tuple.llm_response,
#                     "prompt": prompt.format(
#                         question=question_llm_response_tuple.question,
#                         llm_response=question_llm_response_tuple.llm_response,
#                     ),
#                 }
#             )

#     if not all_items_to_process:
#         all_results = {}
#     # or handle as you see fit, e.g., return {}
#     else:
#         # 2. BATCH: Prepare prompts for a single batch generation call.
#         all_prompts = [item["prompt"] for item in all_items_to_process]
#         system_messages = [system_prompt] * len(all_prompts)

#         # Perform the single, efficient batch inference.
#         llm_parsed_results, logprobs, reasoning_output = batch_generation(
#             model,
#             system_messages=system_messages,
#             prompts=all_prompts,
#             response_generation_method=answer_production_method,  # TODO: fix automatic system prompt
#             seed=seed,
#             print_conversation=print_conversation,
#             print_progress=print_progress,
#             chat_template_kwargs={
#                 "enable_thinking": False
#             },  # disable reasoning to facilitate parsing
#             **generation_kwargs,
#         )

#     for item, parsed_result in zip(all_items_to_process, llm_parsed_results):
#         item[constants.PARSED_RESPONSE] = parsed_result

#     # Group the results by survey_name to build the final DataFrames.
#     # defaultdict is perfect for this task.
#     grouped_data = defaultdict(list)
#     for item in all_items_to_process:
#         grouped_data[item[constants.QUESTIONNAIRE_NAME]].append(
#             {
#                 constants.QUESTIONNAIRE_ITEM_ID: item[constants.QUESTIONNAIRE_ITEM_ID],
#                 constants.QUESTION: item[constants.QUESTION],
#                 constants.LLM_RESPONSE: item[constants.LLM_RESPONSE],
#                 constants.PARSED_RESPONSE: item[constants.PARSED_RESPONSE],
#             }
#         )
#     all_results = {
#         survey_name: pd.DataFrame(data_list)
#         for survey_name, data_list in grouped_data.items()
#     }

#     return all_results


def _filter_logprobs_by_choices(
    logprob_df: pd.DataFrame, choices: pd.Series
) -> pd.DataFrame:

    matches_found = []

    # check for each output token whether any of the choices start with this token
    for token in logprob_df["token"]:
        boolean_index = choices.str.startswith(token)
        # if len(choices[boolean_index]) > 1:
        #    warnings.warn(
        #        f"Multiple allowed_choices ({list(choices[boolean_index])}) match the same output token: {token}",
        #        stacklevel=2
        #    )
        matches_found.append(boolean_index.any())

    return logprob_df[matches_found]


def _logprobs_filter(
    logprobs: Dict[str, float], allowed_choices: Dict[str, List[str]]
) -> Dict[str, float]:

    # normalize logprobs
    logprob_df = pd.DataFrame({"token": logprobs.keys(), "prob": logprobs.values()})
    logprob_df["prob"] = logprob_df.prob.apply(np.exp)
    logprob_df = logprob_df[logprob_df.prob > 0]

    # flatten to check for collisions between answer options
    # TODO: implement this properly---only collisions between answer options matter, not, e.g., TRUMP vs. trump!
    # all_valid_outputs = [output for choices in allowed_choices.values() for output in choices]
    # _ = _filter_logprobs_by_choices(logprob_df, pd.Series(all_valid_outputs))

    # filter the individual survey answers
    choice_results = {}
    for choice, valid_outputs in allowed_choices.items():
        valid_logprobs = _filter_logprobs_by_choices(
            logprob_df, pd.Series(valid_outputs)
        )
        if len(valid_logprobs) == 0:
            warnings.warn(
                f"Could not find logprobs for answer option '{choice}' with possible outputs {valid_outputs}"
            )
            choice_results[choice] = np.nan
        else:
            choice_results[choice] = valid_logprobs["prob"].sum()

    # normalize so that probs sum up to 1
    overall_sum = sum(
        [_result for _result in choice_results.values() if not np.isnan(_result)]
    )  # only consider values != nan
    if not np.isnan(overall_sum) and overall_sum > 0:
        choice_results = {
            choice: token_sum / overall_sum
            for choice, token_sum in choice_results.items()
        }

    return choice_results


def parse_logprobs(
    survey_results: List[InferenceResult],
    allowed_choices: List[str] | Dict[str, List[str]],
) -> Dict[LLMPrompt, pd.DataFrame]:
    """
    Filter and aggregate the logprobs that are returned when using the Logprob_AnswerProductionMethod

    Args:
        survey_results: List of InterviewResult that is returned from running a survey
        allowed_choices: List of possible answer options OR dictionary that maps answer options to multiple tokens that encode each option

    Returns:
        Dict[LLMInterview, pd.Dataframe]: A dictionary where the keys are the LLMInterviews and the values are a Dataframe with the questions and answers.
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
                    "No logprobs found in InterviewResult. "
                    + "Make sure to use Logprob_AnswerProductionMethod to generate logprobs.",
                    stacklevel=2,
                )
                answer_format = ["error_col"]
                answers.append((item_id, qa_tuple.question, "ERROR: Parsing"))
            else:
                filtered_logprobs = _logprobs_filter(qa_tuple.logprobs, allowed_choices)
                answer_format = filtered_logprobs.keys()
                answers.append(
                    (item_id, qa_tuple.question, *filtered_logprobs.values())
                )

            df = pd.DataFrame(
                answers,
                columns=[
                    constants.QUESTIONNAIRE_ITEM_ID,
                    constants.QUESTION,
                    *answer_format,
                ],
            )
            final_result[survey_result.questionnaire] = df

    return final_result
