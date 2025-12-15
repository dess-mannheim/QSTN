import streamlit as st
import pandas as pd
from io import StringIO

from qstn.survey_manager import SurveyCreator

from qstn.prompt_builder import LLMPrompt

st.set_page_config(layout="wide")
st.title("QSTN")

# Define example dataframes once (used for both demo and defaults)
example_questionnaire = pd.DataFrame({
    'questionnaire_item_id': [1, 2, 3],
    'question_content': ['Coffee', 'Pizza', 'Ice cream'],
    'question_stem': [
        'How do you feel about?',
        'How do you feel about?',
        'How do you feel about?'
    ]
})

example_population = pd.DataFrame({
    'questionnaire_name': ['Student', 'Teacher'],
    'system_prompt': ['You are a student.', 'You are a teacher.'],
    'questionnaire_instruction': [
        'Please answer the following questions.',
        'Please answer the following questions.'
    ]
})

# Demo section showing expected CSV format
with st.expander("📋 View Example CSV Format", expanded=False):
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.subheader("Questionnaire CSV Format")
        st.write("**Required columns:** `questionnaire_item_id`, `question_content`, `question_stem`")
        st.write(example_questionnaire)
    
    with demo_col2:
        st.subheader("Population CSV Format")
        st.write("**Required columns:** `questionnaire_name`, `system_prompt`, `questionnaire_instruction`")
        st.write(example_population)

st.divider()

col1, col2 = st.columns(2)

# Initialize session state for dataframes if not present
if "df_questionnaire" not in st.session_state:
    st.session_state.df_questionnaire = None
if "df_population" not in st.session_state:
    st.session_state.df_population = None

df_population = None
df_questionnaire = None
using_defaults = False

with col1:
    uploaded_questionnaire = st.file_uploader("Select a questionnaire to start with")
    if uploaded_questionnaire is not None:
        # New file uploaded - read and store it
        df_questionnaire = pd.read_csv(uploaded_questionnaire)
        st.session_state.df_questionnaire = df_questionnaire
        st.write(df_questionnaire)
    elif st.session_state.df_questionnaire is not None:
        # Use previously uploaded file from session state
        df_questionnaire = st.session_state.df_questionnaire
        st.write(df_questionnaire)
        if st.button("Clear", key="clear_questionnaire", help="Reset to example questionnaire"):
            st.session_state.df_questionnaire = None
            st.rerun()
    else:
        # No file uploaded and no previous file - use example
        df_questionnaire = example_questionnaire
        using_defaults = True
        st.info("ℹ️ Using example questionnaire. Upload a file to use your own data.")
        st.write(df_questionnaire)

with col2:
    uploaded_population = st.file_uploader("Select a population to start with")
    if uploaded_population is not None:
        # New file uploaded - read and store it
        df_population = pd.read_csv(uploaded_population)
        st.session_state.df_population = df_population
        st.write(df_population)
    elif st.session_state.df_population is not None:
        # Use previously uploaded file from session state
        df_population = st.session_state.df_population
        st.write(df_population)
        if st.button("Clear", key="clear_population", help="Reset to example population"):
            st.session_state.df_population = None
            st.rerun()
    else:
        # No file uploaded and no previous file - use example
        df_population = example_population
        using_defaults = True
        st.info("ℹ️ Using example population. Upload a file to use your own data.")
        st.write(df_population)

# Button is always enabled now (using defaults if no files uploaded)
    disabled = False
        
st.divider()

if st.button("Confirm and Prepare Questionnaire", type="primary", disabled=disabled, use_container_width=True):
    questionnaires: list[LLMPrompt] = SurveyCreator.from_dataframe(df_population, df_questionnaire)
    st.session_state.questionnaires = questionnaires
    st.switch_page("pages/01_Option_Prompt.py")
