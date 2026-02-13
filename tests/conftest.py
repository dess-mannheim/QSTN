# tests/conftest.py
import pytest
from unittest.mock import MagicMock, AsyncMock

from openai import AsyncOpenAI

import pandas as pd

import qstn
from qstn.prompt_builder import LLMPrompt

@pytest.fixture(scope="function")
def mock_questionnaires():
    return pd.DataFrame([
        {"questionnaire_item_id": 1, "question_content": "How do you feel about Red?"},
        {"questionnaire_item_id": 2, "question_content": "How do you feel about Blue?"}
    ])

@pytest.fixture(scope="function")
def mock_personas():
    return pd.DataFrame([
        {"system_prompt": "You are a helpful assistant."},
        {"system_prompt": "You are a grumpy bot."}
    ])


@pytest.fixture(scope="function")
def llm_prompt_factory():
    """
    Returns a builder function to generate LLMPrompt objects dynamically.
    """
    def _builder(system_prompts: list[str], questionnaire: pd.DataFrame, custom_template: str = None):
        if custom_template:
            template = custom_template
        else:
            template = (
                f"Please tell us how you feel about:\n"
                f"{qstn.utilities.placeholder.PROMPT_QUESTIONS}"
            )

        interviews = [
            qstn.prompt_builder.LLMPrompt(
                questionnaire_source=questionnaire,
                system_prompt=persona_text,
                prompt=template,
            ) for persona_text in system_prompts.system_prompt
        ]

        return interviews

    return _builder

@pytest.fixture(scope="function")
def mock_openai_client():
    """
    Creates a mock AsyncOpenAI client that mimics the structure of a real response.
    """
    class MockOpenAIClient(AsyncMock):
        pass
    MockOpenAIClient.__name__ = "AsyncOpenAI"

    mock_client = MockOpenAIClient(spec=AsyncOpenAI)
    mock_client.__class__.__name__ = "AsyncOpenAI"
    mock_client.chat.completions.create = AsyncMock()

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "I feel neutral about this."
    mock_completion.choices[0].message.reasoning = None
    mock_completion.choices[0].message.reasoning_content = None
    
    mock_client.chat.completions.create.return_value = mock_completion
    
    return mock_client

@pytest.fixture
def mock_openai_response_factory(mock_openai_client):
    """
    Mock responses
    """
    def _create(content:str ="Hello world"):
        mock_client = mock_openai_client

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = content
        mock_completion.choices[0].message.reasoning = None
        mock_completion.choices[0].message.reasoning_content = None

        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client
    return _create