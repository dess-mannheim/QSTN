import streamlit as st
from surveygen.llm_interview import LLMInterview
from surveygen.survey_manager import conduct_survey_question_by_question, conduct_survey_in_context, conduct_whole_survey_one_prompt, SurveyOptionGenerator, SurveyCreator
from surveygen.utilities.constants import InterviewType
from gui_elements.paginator import paginator
from gui_elements.stateful_widget import StatefulWidgets
import time


#CONSTANTS FOR FIELDS
system_prompt_field = "System prompt"
interview_instruction_field = "Interview instructions"
change_all_system_prompts_checkbox = "system_change_all"
change_all_interview_instructions_checkbox = "interview_instructions_change_all"

st.set_page_config(layout="wide")
st.title("Generate Prompt")
st.write(
    "This interface allows you to inspect and change the system prompt and primary instructions for the model."
)
st.divider()


@st.cache_data
def create_stateful_widget() -> StatefulWidgets:
    return StatefulWidgets()

state = create_stateful_widget()

#FOR DEBUGGING
# if "interviews" not in st.session_state:
#     st.session_state.interviews = [SurveyCreator().from_path(survey_path="/home/maxi/Documents/SurveyGen/surveys/ANES.csv", questionnaire_path="/home/maxi/Documents/SurveyGen/surveys/ANES_PERSONAS.csv")]
if "interviews" not in st.session_state:
    st.error("You need to first upload a questionnaire and the population you want to survey.")
    st.stop()

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

text_field_ids = [system_prompt_field, interview_instruction_field]

current_interview_id = paginator(st.session_state.interviews, "current_interview_index_prompt")

st.divider()

if "interviews" in st.session_state and st.session_state.interviews is not None:
    try:
        interview = st.session_state.interviews[current_interview_id].duplicate()
    except IndexError:
        st.error("Index is out of range. Resetting to the first item.")
        current_interview_id = 0
        interview = st.session_state.interviews[current_interview_id].duplicate()
        
    #st.session_state.preview_interview = interview

    col_options, col_prompt_display = st.columns(2)

    with col_options:
        st.subheader("⚙️ Configuration")

        new_system_prompt = st.text_area(
            label=system_prompt_field,
            key=f"{system_prompt_field}{current_interview_id}",
            value=interview.system_prompt,
            help="The system prompt the model is prompted with."
        )

        change_all_system = state.create(
            st.checkbox,
            key=change_all_system_prompts_checkbox,
            label="On update: change all System Prompts",
            help="If this is ticked, all system prompts will be changed to this.",
            initial_value=False
        )

        new_interview_instruction = st.text_area(
            label=interview_instruction_field,
            key=f"{interview_instruction_field}{current_interview_id}",
            value=interview.interview_instruction,
            help="Instructions that are given to the model before the questions."
        )

        change_all_interview = state.create(
            st.checkbox,
            key=change_all_interview_instructions_checkbox,
            label="On update: change all interview instructions",
            help="If this is ticked, all interview instructions will be changed to this.",
            initial_value=False
        )

    # Place the corresponding output in the second column
    with col_prompt_display:
        st.subheader("📄 Live Preview")

    # --- The Dynamic Preview Logic ---
    # This block re-runs on every widget interaction.
        with st.container(border=True):
            interview.system_prompt = new_system_prompt
            interview.interview_instruction = new_interview_instruction
            current_prompt = interview.get_prompt_for_interview_type(InterviewType.CONTEXT)
            current_prompt = current_prompt.replace("\n", "  \n")
            st.write(current_prompt)
    if st.button("Update Prompt(s)", type="secondary", use_container_width=True):
        if change_all_system:
            for interview in st.session_state.interviews:
                interview.system_prompt = new_system_prompt
        else:
            st.session_state.interviews[current_interview_id].system_prompt = new_system_prompt
        if change_all_interview:
            for interview in st.session_state.interviews:
                interview.interview_instruction = new_interview_instruction
        else:
            st.session_state.interviews[current_interview_id].interview_instruction = new_interview_instruction             
        st.success("Prompt(s) updated!")

    if st.button("Confirm Base Prompt", type="primary", use_container_width=True):
        st.switch_page("pages/02_Option_Prompt.py")
else:
    st.warning("No data found. Please upload a CSV file on the 'Start Page' first.")


