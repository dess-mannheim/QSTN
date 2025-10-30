import streamlit as st
import pandas as pd
from io import StringIO

from surveygen.survey_manager import SurveyCreator

from surveygen.llm_interview import LLMInterview

st.set_page_config(layout="wide")
st.title("SurveyGen")

col1, col2 = st.columns(2)

# def save_dataframe(uploaded_file):

#     try:
#         df = pd.read_csv(uploaded_file)
#         # Store the DataFrame in session state
#         st.session_state.df = df
#     except Exception as e:
#         st.error(f"Error reading the file: {e}")

#     return df

df_population = None
df_questionnaire = None

with col1:
    uploaded_questionnaire = st.file_uploader("Select a questionnaire to start with")
    if uploaded_questionnaire is not None:
        # bytes_data = uploaded_questionnaire.getvalue()
        # st.write(bytes_data)

        df_questionnaire = pd.read_csv(uploaded_questionnaire)
        #dataframe = save_dataframe(uploaded_questionnaire)

        st.write(df_questionnaire)

with col2:
    uploaded_population = st.file_uploader("Select a population to start with")
    if uploaded_population is not None:
        # bytes_data = uploaded_population.getvalue()
        # st.write(bytes_data)

        df_population = pd.read_csv(uploaded_population)
        #dataframe = save_dataframe(uploaded_population)

        st.write(df_population)

disabled = True

if df_population is not None and df_questionnaire is not None:
    disabled = False
        
st.divider()

if st.button("Confirm and Prepare Interview", type="primary", disabled=disabled, use_container_width=True):
    interviews: list[LLMInterview] = SurveyCreator.from_dataframe(df_population, df_questionnaire)
    st.session_state.interviews = interviews
    st.switch_page("pages/01_Basic_Prompt_Settings.py")
