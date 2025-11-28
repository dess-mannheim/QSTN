import streamlit as st
from qstn.survey_manager import SurveyOptionGenerator
from qstn.llm_questionnaire import LLMQuestionnaire
from qstn.utilities.constants import QuestionnaireType
from qstn.utilities import placeholder
from typing import Any

from gui_elements.paginator import paginator
from gui_elements.stateful_widget import StatefulWidgets

# CONSTANTS FOR FIELDS
question_stem_field = "Question Stem"
randomize_order_tick = "Randomize the order of items"
system_prompt_field = "System prompt"
prompt_field = "Prompt"
change_all_system_prompts_checkbox = "system_change_all"
change_all_prompts_checkbox = "prompts_change_all"

field_ids = [question_stem_field, randomize_order_tick]

st.set_page_config(layout="wide")
st.title("Prompt Configuration")
st.write(
    "This interface allows you configure how the questions are prompted to the LLM and the overall prompt structure. "
    "These options are applied to every questionnaire in your survey."
)
st.page_link("pages/02_Option_Prompt.py", label="Click here to adjust the answer options.")
st.divider()

@st.cache_data
def create_stateful_widget() -> StatefulWidgets:
    return StatefulWidgets()

state = create_stateful_widget()

if "questionnaires" not in st.session_state:
    st.error("You need to first upload a questionnaire and the population you want to survey.")
    st.stop()
    disabled = True
else:
    disabled = False

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

#current_questionnaire_id = paginator(st.session_state.questionnaires, "current_questionnaire_index_prepare")
current_questionnaire_id = paginator(st.session_state.questionnaires, "current_questionnaire_index_prompt")

if not "temporary_questionnaire" in st.session_state:
    st.session_state.temporary_questionnaire = st.session_state.questionnaires[0].duplicate()

if not "base_questionnaire" in st.session_state:
    st.session_state.base_questionnaire = st.session_state.temporary_questionnaire.duplicate()

def process_inputs(input: Any, field_id: str) -> str:
    if "survey_options" in st.session_state:
        survey_options = st.session_state.survey_options
    else:
        survey_options = None

    if field_id == question_stem_field:
        LLMQuestionnaire.prepare_questionnaire
        st.session_state.temporary_questionnaire.prepare_questionnaire(
            question_stem=input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
        st.session_state.base_questionnaire.prepare_questionnaire(
            question_stem=input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
    elif field_id == randomize_order_tick:
        if input == True:
            st.session_state.temporary_questionnaire.prepare_questionnaire(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=input,
            )
        else:
            st.session_state.temporary_questionnaire = st.session_state.base_questionnaire.duplicate()

def handle_change(field_id: str):
    """
    This single callback handles changes from any text field.
    It reads the input from session state using the unique key,
    processes it, and saves the output to session state.
    """
    input_key = f"input_{field_id}"

    with st.spinner(f"Processing {field_id}..."):
        # time.sleep(0.5) # Simulate work
        process_inputs(st.session_state[input_key], field_id)


if "questionnaires" in st.session_state and st.session_state.questionnaires is not None:
    try:
        questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()
    except IndexError:
        st.error("Index is out of range. Resetting to the first item.")
        current_questionnaire_id = 0
        questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("⚙️ Configuration")

        for field_id in field_ids:
            input_key = f"input_{field_id}"
            if not input_key in st.session_state:
                if field_id == question_stem_field:
                    st.session_state[input_key] = st.session_state.temporary_questionnaire ._questions[0].question_stem
                if field_id == randomize_order_tick:
                    st.session_state[input_key] = False

        # Handle placeholder replacement before widget is created
        input_key = f"input_{question_stem_field}"
        if "placeholder_to_replace" in st.session_state and st.session_state.placeholder_to_replace:
            current_text = st.session_state.get(input_key, "")
            placeholder_shortcut = st.session_state.placeholder_to_replace["shortcut"]
            placeholder_value = st.session_state.placeholder_to_replace["value"]
            
            # Replace all occurrences of the shortcut (e.g., -Q) with the placeholder
            if placeholder_shortcut in current_text:
                st.session_state[input_key] = current_text.replace(placeholder_shortcut, placeholder_value)
            else:
                # Shortcut not found, append at the end
                st.session_state[input_key] = current_text + f" {placeholder_value} "
            
            st.session_state.placeholder_to_replace = None
            st.rerun()

        # --- Input Widgets (No Form) ---
        question_stem_input = st.text_area(
            question_stem_field,
            key=f"input_{question_stem_field}",
            # placeholder="e.g., How would you rate the following aspects of our service?",
            #on_change=handle_change,
            kwargs={'field_id': question_stem_field},
            height=100,
        )

        # --- Placeholder Replacement Buttons ---
        st.write("**Insert Placeholder:**")
        
        # Define available placeholders with their shortcuts and character labels
        # Format: (placeholder_value, shortcut, character_label, description)
        available_placeholders = [
            (placeholder.QUESTION_CONTENT, "-Q", "Q", "Question Content"),
            (placeholder.PROMPT_OPTIONS, "-O", "O", "Prompt Options"),
        ]
        
        # Create shortcuts list for the tip
        shortcuts_list = ", ".join([f"`{shortcut}`" for _, shortcut, _, _ in available_placeholders])
        st.caption(f"💡 Tip: Type shortcuts {shortcuts_list} in the text, then click the button to replace them with placeholders.")
        
        # Create buttons in columns with consistent formatting
        cols = st.columns(len(available_placeholders))
        for i, (placeholder_value, shortcut, char_label, description) in enumerate(available_placeholders):
            button_label = description  # Use the actual placeholder name
            button_key = f"btn_placeholder_{char_label}"
            
            if cols[i].button(button_label, key=button_key, use_container_width=True, help=f"Replaces '{shortcut}' with {placeholder_value}"):
                st.session_state.placeholder_to_replace = {
                    "shortcut": shortcut,
                    "value": placeholder_value
                }
                st.rerun()

        randomize_order_bool = st.checkbox(
            randomize_order_tick,
            key=f"input_{randomize_order_tick}",
            value=False,
            #on_change=handle_change,
            kwargs={'field_id': randomize_order_tick} 
        )

        st.divider()

        # System prompt and main prompt section (from Basic Prompt Settings)
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

        # Handle placeholder replacement for main prompt before widget is created
        prompt_key = f"{prompt_field}{current_questionnaire_id}"
        if "main_prompt_placeholder_to_replace" in st.session_state and st.session_state.main_prompt_placeholder_to_replace:
            current_prompt_text = st.session_state.get(prompt_key, questionnaire.prompt)
            placeholder_shortcut = st.session_state.main_prompt_placeholder_to_replace["shortcut"]
            placeholder_value = st.session_state.main_prompt_placeholder_to_replace["value"]
            
            if placeholder_shortcut in current_prompt_text:
                st.session_state[prompt_key] = current_prompt_text.replace(placeholder_shortcut, placeholder_value)
            else:
                st.session_state[prompt_key] = current_prompt_text + f" {placeholder_value} "
            
            st.session_state.main_prompt_placeholder_to_replace = None
            st.rerun()

        new_prompt = st.text_area(
            label=prompt_field,
            key=prompt_key,
            value=questionnaire.prompt,
            help="Instructions that are given to the model before the questions."
        )

        # Placeholder insertion buttons for main prompt
        st.write("**Insert Placeholder in Main Prompt:**")
        main_prompt_placeholders = [
            (placeholder.PROMPT_QUESTIONS, "-P", "P", "Prompt Questions"),
            (placeholder.PROMPT_OPTIONS, "-O", "O", "Prompt Options"),
            (placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS, "-A", "A", "Automatic Output"),
            (placeholder.JSON_TEMPLATE, "-J", "J", "JSON Template"),
        ]
        
        main_shortcuts_list = ", ".join([f"`{shortcut}`" for _, shortcut, _, _ in main_prompt_placeholders])
        st.caption(f"💡 Tip: Type shortcuts {main_shortcuts_list} in the main prompt, then click the button to replace them.")
        
        cols_main = st.columns(len(main_prompt_placeholders))
        for i, (placeholder_value, shortcut, char_label, description) in enumerate(main_prompt_placeholders):
            button_key = f"btn_main_placeholder_{char_label}"
            if cols_main[i].button(description, key=button_key, use_container_width=True, help=f"Replaces '{shortcut}' with {placeholder_value}"):
                st.session_state.main_prompt_placeholder_to_replace = {
                    "shortcut": shortcut,
                    "value": placeholder_value
                }
                st.rerun()

        change_all_questionnaire = state.create(
            st.checkbox,
            key=change_all_prompts_checkbox,
            label="On update: change all questionnaire instructions",
            help="If this is ticked, all questionnaire instructions will be changed to this.",
            initial_value=False
        )

    # Place the corresponding output in the second column
    with col2:
        st.subheader("📄 Live Preview")

    # --- The Dynamic Preview Logic ---
    # This block re-runs on every widget interaction.
        with st.container(border=True):
            # Update temporary questionnaire with question stem
            if "survey_options" in st.session_state:
                survey_options = st.session_state.survey_options
            else:
                survey_options = None

            if randomize_order_bool:
                st.session_state.temporary_questionnaire.prepare_questionnaire(
                    question_stem=question_stem_input,
                    answer_options=survey_options,
                    randomized_item_order=randomize_order_bool,
                )
            st.session_state.base_questionnaire.prepare_questionnaire(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=False,
            )

            if not randomize_order_bool:
                st.session_state.temporary_questionnaire = st.session_state.base_questionnaire.duplicate()

            # Update system prompt and main prompt for preview (apply to temporary_questionnaire)
            st.session_state.temporary_questionnaire.system_prompt = new_system_prompt
            st.session_state.temporary_questionnaire.prompt = new_prompt
            current_system_prompt, current_prompt = st.session_state.temporary_questionnaire.get_prompt_for_questionnaire_type(QuestionnaireType.SEQUENTIAL)
            current_system_prompt = current_system_prompt.replace("\n", "  \n")
            current_prompt = current_prompt.replace("\n", "  \n")
            st.write(current_system_prompt)
            st.write(current_prompt)

    st.divider()

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

    if st.button("Confirm and Prepare Questionnaire", type="primary", use_container_width=True):
        for questionnaire in st.session_state.questionnaires:
            questionnaire.prepare_questionnaire(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=randomize_order_bool,
            )
        st.success("Changed the prompts!")
        st.switch_page("pages/02_Option_Prompt.py")
else:
    st.warning("No data found. Please upload a CSV file on the 'Start Page' first.")
