import pytest
import pandas as pd

from unittest.mock import MagicMock, AsyncMock

from qstn.utilities.survey_objects import QuestionnaireItem
import qstn

from openai import AsyncOpenAI

def test_full_survey_workflow(mock_questionnaires, mock_personas, llm_prompt_factory, mock_openai_client):
    # --- SETUP ---
    interviews = llm_prompt_factory(mock_personas, mock_questionnaires)

    print(interviews)

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