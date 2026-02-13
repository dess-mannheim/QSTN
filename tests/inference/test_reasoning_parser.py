import pytest

import pandas as pd

import qstn


TESTED_REASONING = "reasoning_text"
TESTED_FINAL_ANSWER = "Answer: 42"

@pytest.fixture
def reasoning_mock_builder(mock_openai_response_factory):
    """A higher-level factory for your specific business logic."""
    def _builder(start="<think>", end="</think>", text=TESTED_REASONING, answer=TESTED_FINAL_ANSWER):
        full_text = f"{start}{text}{end}{answer}"
        return mock_openai_response_factory(content=full_text)
    return _builder

def test_reasoning_parsing_with_default_values(
    mock_questionnaires, mock_personas, llm_prompt_factory, reasoning_mock_builder
):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=reasoning_mock_builder(
            "<think>", "</think>"
        ),
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3,
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    first_interview: pd.Dataframe = parsed_results[interviews[0]]
    
    # --- ASSERTIONS ---
    # Test 1: Reasoning parsed correctly
    assert first_interview.loc[0].reasoning == TESTED_REASONING

    # Test 2: Answer parsed correctly
    assert first_interview.loc[0].llm_response == TESTED_FINAL_ANSWER


def test_reasoning_parsing_with_no_start_tag_values(
    mock_questionnaires, mock_personas, llm_prompt_factory, reasoning_mock_builder
):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=reasoning_mock_builder(
            "", "</think>"
        ),
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3,
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    first_interview: pd.Dataframe = parsed_results[interviews[0]]
    
    # --- ASSERTIONS ---
    # Test 1: Reasoning parsed correctly
    assert first_interview.loc[0].reasoning == TESTED_REASONING

    # Test 2: Answer parsed correctly
    assert first_interview.loc[0].llm_response == TESTED_FINAL_ANSWER


def test_reasoning_parsing_with_no_start_tag_and_specified_none(
    mock_questionnaires, mock_personas, llm_prompt_factory, reasoning_mock_builder
):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=reasoning_mock_builder(
            "", "</think>"
        ),
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3,
        reasoning_start_token=None
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    first_interview: pd.Dataframe = parsed_results[interviews[0]]
    
    # --- ASSERTIONS ---
    # Test 1: Reasoning parsed correctly
    assert first_interview.loc[0].reasoning == TESTED_REASONING

    # Test 2: Answer parsed correctly
    assert first_interview.loc[0].llm_response == TESTED_FINAL_ANSWER

def test_reasoning_parsing_with_no_start_tag_and_specified_empty(
    mock_questionnaires, mock_personas, llm_prompt_factory, reasoning_mock_builder
):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=reasoning_mock_builder(
            "", "</think>"
        ),
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3,
        reasoning_start_token=""
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    first_interview: pd.Dataframe = parsed_results[interviews[0]]
    
    # --- ASSERTIONS ---
    # Test 1: Reasoning parsed correctly
    assert first_interview.loc[0].reasoning == TESTED_REASONING

    # Test 2: Answer parsed correctly
    assert first_interview.loc[0].llm_response == TESTED_FINAL_ANSWER


def test_reasoning_parsing_with_non_default_reasoning_tokens(
    mock_questionnaires, mock_personas, llm_prompt_factory, reasoning_mock_builder
):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=reasoning_mock_builder(
            "ME_START_THINKING", "ME_END_THINKING"
        ),
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3,
        reasoning_start_token="ME_START_THINKING",
        reasoning_end_token="ME_END_THINKING"
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    first_interview: pd.Dataframe = parsed_results[interviews[0]]
    
    # --- ASSERTIONS ---
    # Test 1: Reasoning parsed correctly
    assert first_interview.loc[0].reasoning == TESTED_REASONING

    # Test 2: Answer parsed correctly
    assert first_interview.loc[0].llm_response == TESTED_FINAL_ANSWER


def test_reasoning_parsing_with_no_reasoning_start_and_end_tokens(
    mock_questionnaires, mock_personas, llm_prompt_factory, reasoning_mock_builder
):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=reasoning_mock_builder(
            "", "", ""
        ),
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3,
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    first_interview: pd.Dataframe = parsed_results[interviews[0]]
    
    # --- ASSERTIONS ---
    # Test 1: Answer parsed correctly
    assert first_interview.loc[0].llm_response == TESTED_FINAL_ANSWER