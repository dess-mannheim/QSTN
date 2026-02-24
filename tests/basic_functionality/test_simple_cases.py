"""End-to-end smoke tests for core survey execution modes.

These tests validate that single-item, sequential, and battery survey flows
produce expected dataframe outputs with mocked API responses.
"""

import pandas as pd

from qstn.utilities.survey_objects import QuestionnaireItem
import qstn

def test_simple_single_item(mock_questionnaires, mock_personas, llm_prompt_factory, mock_openai_client):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    # Add additional question only to the first interview
    interview = interviews[0]
    new_item = QuestionnaireItem(item_id=3, question_content="How do you feel about Green?")
    interview.insert_questions(new_item, 1)

    results = qstn.survey_manager.conduct_survey_single_item(
        model=mock_openai_client,
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    # --- ASSERTIONS ---
    first_interview = parsed_results[interviews[0]]

    # 1. Check if we got a DataFrame back
    assert isinstance(first_interview, pd.DataFrame)
    assert not first_interview.empty

    # 2. Check logic for the specific interview we modified (First Persona)
    # We expect 3 answers because we started with 2 and inserted 1.
    assert len(first_interview) == 3

    # 3. Check if the INSERTED question is present and in the correct relative position
    # The IDs should be ordered 1 -> 3 -> 2 based on our insertion at index 1
    assert 3 in first_interview['questionnaire_item_id'].values
    assert "How do you feel about Green?" == first_interview['question'].iloc[1]
    
    # 4. Check if the mocked answer came through (just to test if it was actually called)
    assert first_interview.iloc[0]['llm_response'] == "I feel neutral about this."

    # 5. Check the second interview (which we did NOT touch)
    # It should still only have 2 questions and answers
    second_interview = parsed_results[interviews[1]]
    assert len(second_interview) == 2


def test_simple_sequential(mock_questionnaires, mock_personas, llm_prompt_factory, mock_openai_client):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    # Add additional question only to the first interview
    interview = interviews[0]
    new_item = QuestionnaireItem(item_id=3, question_content="How do you feel about Green?")
    interview.insert_questions(new_item, 1)

    results = qstn.survey_manager.conduct_survey_sequential(
        model=mock_openai_client,
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    # --- ASSERTIONS ---
    first_interview = parsed_results[interviews[0]]

    # 1. Check if we got a DataFrame back
    assert isinstance(first_interview, pd.DataFrame)
    assert not first_interview.empty

    # 2. Check logic for the specific interview we modified (First Persona)
    # We expect 3 answers because we started with 2 and inserted 1.
    assert len(first_interview) == 3

    # 3. Check if the INSERTED question is present and in the correct relative position
    # The IDs should be ordered 1 -> 3 -> 2 based on our insertion at index 1
    assert 3 in first_interview['questionnaire_item_id'].values
    assert "How do you feel about Green?" == first_interview['question'].iloc[1]
    
    # 4. Check if the mocked answer came through (just to test if it was actually called)
    assert first_interview.iloc[0]['llm_response'] == "I feel neutral about this."

    # 5. Check the second interview (which we did NOT touch)
    # It should still only have 2 questions and answers
    second_interview = parsed_results[interviews[1]]
    assert len(second_interview) == 2

def test_simple_battery(mock_questionnaires, mock_personas, llm_prompt_factory, mock_openai_client):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    # Add additional question only to the first interview
    interview = interviews[0]
    new_item = QuestionnaireItem(item_id=3, question_content="How do you feel about Green?")
    interview.insert_questions(new_item, 1)

    results = qstn.survey_manager.conduct_survey_battery(
        model=mock_openai_client,
        llm_prompts=interviews,
        max_tokens=100,
        client_model_name="mock-model",
        api_concurrency=3
    )

    # --- PARSING ---
    parsed_results = qstn.parser.raw_responses(results)

    # --- ASSERTIONS ---
    first_interview = parsed_results[interviews[0]]

    # 1. Check if we got a DataFrame back
    assert isinstance(first_interview, pd.DataFrame)
    assert not first_interview.empty

    # 2. Battery mode aggregates into a single questionnaire row.
    assert len(first_interview) == 1
    assert first_interview["questionnaire_item_id"].iloc[0] == -1
    assert "How do you feel about Green?" in first_interview["question"].iloc[0]

    # 3. Check if the mocked answer came through.
    assert first_interview.iloc[0]['llm_response'] == "I feel neutral about this."
