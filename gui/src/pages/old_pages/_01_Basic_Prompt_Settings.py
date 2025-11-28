import streamlit as st
from qstn.llm_questionnaire import LLMQuestionnaire
from qstn.survey_manager import conduct_survey_single_item, conduct_survey_sequential, conduct_survey_battery, SurveyOptionGenerator, SurveyCreator
from qstn.utilities.constants import QuestionnaireType
from gui_elements.paginator import paginator
from gui_elements.stateful_widget import StatefulWidgets
import time


#CONSTANTS FOR FIELDS
system_prompt_field = "System prompt"
prompt_field = "Prompt"
change_all_system_prompts_checkbox = "system_change_all"
change_all_prompts_checkbox = "prompts_change_all"

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
# if "questionnaires" not in st.session_state:
#     st.session_state.questionnaires = [SurveyCreator().from_path(survey_path="/home/maxi/Documents/SurveyGen/surveys/ANES.csv", questionnaire_path="/home/maxi/Documents/SurveyGen/surveys/ANES_PERSONAS.csv")]
if "questionnaires" not in st.session_state:
    st.error("You need to first upload a questionnaire and the population you want to survey.")
    st.stop()

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

text_field_ids = [system_prompt_field, prompt_field]

current_questionnaire_id = paginator(st.session_state.questionnaires, "current_questionnaire_index_prompt")

st.divider()

if "questionnaires" in st.session_state and st.session_state.questionnaires is not None:
    try:
        questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()
    except IndexError:
        st.error("Index is out of range. Resetting to the first item.")
        current_questionnaire_id = 0
        questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()
        
    #st.session_state.preview_questionnaire = questionnaire

    col_options, col_prompt_display = st.columns(2)

    with col_options:
        st.subheader("⚙️ Configuration")

        new_system_prompt = st.text_area(
            label=system_prompt_field,
            key=f"{system_prompt_field}{current_questionnaire_id}",
            value=questionnaire.system_prompt,
            help="The system prompt the model is prompted with."
        )

        change_all_system = state.create(
            st.checkbox,
            key=change_all_system_prompts_checkbox,
            label="On update: change all System Prompts",
            help="If this is ticked, all system prompts will be changed to this.",
            initial_value=False
        )

        new_prompt = st.text_area(
            label=prompt_field,
            key=f"{prompt_field}{current_questionnaire_id}",
            value=questionnaire.prompt,
            help="Instructions that are given to the model before the questions."
        )

        change_all_questionnaire = state.create(
            st.checkbox,
            key=change_all_prompts_checkbox,
            label="On update: change all questionnaire instructions",
            help="If this is ticked, all questionnaire instructions will be changed to this.",
            initial_value=False
        )

    # Place the corresponding output in the second column
    with col_prompt_display:
        st.subheader("📄 Live Preview")

    # --- The Dynamic Preview Logic ---
    # This block re-runs on every widget interaction.
        with st.container(border=True):
            questionnaire.system_prompt = new_system_prompt
            questionnaire.prompt = new_prompt
            current_system_prompt, current_prompt = questionnaire.get_prompt_for_questionnaire_type(QuestionnaireType.SEQUENTIAL)
            current_system_prompt = current_system_prompt.replace("\n", "  \n")
            current_prompt = current_prompt.replace("\n", "  \n")
            st.write(current_system_prompt)
            st.write(current_prompt)
    if st.button("Update Prompt(s)", type="secondary", use_container_width=True):
        if change_all_system:
            for questionnaire in st.session_state.questionnaires:
                questionnaire.system_prompt = new_system_prompt
        else:
            st.session_state.questionnaires[current_questionnaire_id].system_prompt = new_system_prompt

        if change_all_questionnaire:
            for questionnaire in st.session_state.questionnaires:
                questionnaire.prompt = new_prompt
        else:
            st.session_state.questionnaires[current_questionnaire_id].prompt = new_prompt             
        st.success("Prompt(s) updated!")

    if st.button("Confirm Base Prompt", type="primary", use_container_width=True):
        st.switch_page("pages/02_Option_Prompt.py")
else:
    st.warning("No data found. Please upload a CSV file on the 'Start Page' first.")


