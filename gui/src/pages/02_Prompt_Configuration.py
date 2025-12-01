import streamlit as st
from qstn.prompt_builder import LLMPrompt
from qstn.utilities.constants import QuestionnairePresentation
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
st.page_link("pages/01_Option_Prompt.py", label="Click here to adjust the answer options.")
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
        LLMPrompt.prepare_prompt
        st.session_state.temporary_questionnaire.prepare_prompt(
            question_stem=input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
        st.session_state.base_questionnaire.prepare_prompt(
            question_stem=input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
    elif field_id == randomize_order_tick:
        if input == True:
            st.session_state.temporary_questionnaire.prepare_prompt(
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
        
        # Global settings checkboxes at the top
        change_all_system = state.create(
            st.checkbox,
            key=change_all_system_prompts_checkbox,
            label="On update: change all System Prompts",
            help="If this is ticked, all system prompts will be changed to this.",
            initial_value=True
        )
        
        change_all_questionnaire = state.create(
            st.checkbox,
            key=change_all_prompts_checkbox,
            label="On update: change all questionnaire instructions",
            help="If this is ticked, all questionnaire instructions will be changed to this.",
            initial_value=True
        )
        
        st.divider()

        # System prompt and main prompt section (from Basic Prompt Settings)
        system_prompt_key = f"{system_prompt_field}{current_questionnaire_id}"
        prompt_key = f"{prompt_field}{current_questionnaire_id}"
        
        # Handle placeholder replacement for both textboxes before widgets are created
        if "unified_placeholder_to_replace" in st.session_state and st.session_state.unified_placeholder_to_replace:
            current_system_text = st.session_state.get(system_prompt_key, questionnaire.system_prompt)
            current_prompt_text = st.session_state.get(prompt_key, questionnaire.prompt)
            placeholder_shortcut = st.session_state.unified_placeholder_to_replace["shortcut"]
            placeholder_value = st.session_state.unified_placeholder_to_replace["value"]
            target_textbox = st.session_state.unified_placeholder_to_replace.get("target", "prompt")  # "system" or "prompt"
            
            # Check if placeholder already exists in EITHER textbox
            if placeholder_value in current_system_text or placeholder_value in current_prompt_text:
                st.session_state.unified_placeholder_warning = f"⚠️ The placeholder `{placeholder_value}` already exists in one of the textboxes. Please remove it first if you want to insert it again."
            else:
                # Check for shortcut in both textboxes - replace where found
                if placeholder_shortcut in current_system_text:
                    st.session_state[system_prompt_key] = current_system_text.replace(placeholder_shortcut, placeholder_value)
                elif placeholder_shortcut in current_prompt_text:
                    st.session_state[prompt_key] = current_prompt_text.replace(placeholder_shortcut, placeholder_value)
                else:
                    # No shortcut found, add to the target textbox (default based on which was clicked)
                    if target_textbox == "system":
                        st.session_state[system_prompt_key] = current_system_text + f" {placeholder_value} "
                    else:
                        st.session_state[prompt_key] = current_prompt_text + f" {placeholder_value} "
            
            st.session_state.unified_placeholder_to_replace = None
            st.rerun()
        
        # Display warnings if they exist
        if "unified_placeholder_warning" in st.session_state and st.session_state.unified_placeholder_warning:
            st.warning(st.session_state.unified_placeholder_warning)
            st.session_state.unified_placeholder_warning = None  # Clear after displaying
        
        new_system_prompt = st.text_area(
            label=system_prompt_field,
            key=system_prompt_key,
            value=questionnaire.system_prompt,
            help="The system prompt the model is prompted with."
        )

        # Display warning for main prompt if it exists
        if "main_prompt_warning" in st.session_state and st.session_state.main_prompt_warning:
            st.warning(st.session_state.main_prompt_warning)
            st.session_state.main_prompt_warning = None  # Clear after displaying
        
        new_prompt = st.text_area(
            label=prompt_field,
            key=prompt_key,
            value=questionnaire.prompt,
            help="Instructions that are given to the model before the questions."
        )

        # Unified placeholder insertion buttons (work for both system prompt and main prompt)
        st.write("**Insert Placeholder:**")
        
        # Check if options were configured (needed for both main prompt and question stem sections)
        options_configured = "survey_options" in st.session_state and st.session_state.survey_options is not None
        
        unified_placeholders = [
            (placeholder.PROMPT_QUESTIONS, "-P", "P", "Prompt Questions"),
            (placeholder.PROMPT_OPTIONS, "-O", "O", "Prompt Options"),
            (placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS, "-A", "A", "Automatic Output"),
            (placeholder.JSON_TEMPLATE, "-J", "J", "JSON Template"),
        ]
        
        shortcuts_list = ", ".join([f"`{shortcut}`" for _, shortcut, _, _ in unified_placeholders])
        st.caption(f"💡 Tip: Type shortcuts {shortcuts_list} in either the system prompt or main prompt, then click the button to replace them. The placeholder will be inserted where the shortcut is found, or in the main prompt if not found.")
        
        cols_unified = st.columns(len(unified_placeholders))
        for i, (placeholder_value, shortcut, char_label, description) in enumerate(unified_placeholders):
            button_key = f"btn_unified_placeholder_{char_label}"
            
            # Disable "Prompt Options" button if options weren't configured
            is_options_button = placeholder_value == placeholder.PROMPT_OPTIONS
            button_disabled = is_options_button and not options_configured
            button_help = f"Replaces '{shortcut}' with {placeholder_value} in either textbox"
            if button_disabled:
                button_help = "⚠️ You need to configure answer options first. Go back to the Options page to set them up."
            
            if cols_unified[i].button(description, key=button_key, use_container_width=True, disabled=button_disabled, help=button_help):
                st.session_state.unified_placeholder_to_replace = {
                    "shortcut": shortcut,
                    "value": placeholder_value,
                    "target": "prompt"  # Default to main prompt if no shortcut found
                }
                st.rerun()

        st.divider()

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
            
            # Check if placeholder already exists in the text
            if placeholder_value in current_text:
                st.session_state.question_stem_warning = f"⚠️ The placeholder `{placeholder_value}` already exists in the text. Please remove it first if you want to insert it again."
            else:
                # Replace all occurrences of the shortcut (e.g., -Q) with the placeholder
                if placeholder_shortcut in current_text:
                    st.session_state[input_key] = current_text.replace(placeholder_shortcut, placeholder_value)
                else:
                    # Shortcut not found, append at the end
                    st.session_state[input_key] = current_text + f" {placeholder_value} "
            
            st.session_state.placeholder_to_replace = None
            st.rerun()

        # --- Input Widgets (No Form) ---
        # Display warning for question stem if it exists
        if "question_stem_warning" in st.session_state and st.session_state.question_stem_warning:
            st.warning(st.session_state.question_stem_warning)
            st.session_state.question_stem_warning = None  # Clear after displaying
        
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
            
            # Disable "Prompt Options" button if options weren't configured
            is_options_button = placeholder_value == placeholder.PROMPT_OPTIONS
            button_disabled = is_options_button and not options_configured
            button_help = f"Replaces '{shortcut}' with {placeholder_value}"
            if button_disabled:
                button_help = "⚠️ You need to configure answer options first. Go back to the Options page to set them up."
            
            if cols[i].button(button_label, key=button_key, use_container_width=True, disabled=button_disabled, help=button_help):
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

        

    # Place the corresponding output in the second column
    with col2:
        st.subheader("📄 Live Preview")
    # --- The Dynamic Preview Logic ---
    # This block re-runs on every widget interaction.
    # --- The Preview Logic ---
    # Preview only updates when "Update Prompt(s)" is clicked
        with st.container(border=True):

            
            #             # Update temporary questionnaire with question stem
            # if "survey_options" in st.session_state:
            #     survey_options = st.session_state.survey_options
            # else:
            #     survey_options = None

            # if randomize_order_bool:
            #     st.session_state.temporary_questionnaire.prepare_prompt(
            #         question_stem=question_stem_input,
            #         answer_options=survey_options,
            #         randomized_item_order=randomize_order_bool,
            #     )
            # st.session_state.base_questionnaire.prepare_prompt(
            #     question_stem=question_stem_input,
            #     answer_options=survey_options,
            #     randomized_item_order=False,
            # )

            # if not randomize_order_bool:
            #     st.session_state.temporary_questionnaire = st.session_state.base_questionnaire.duplicate()

            # # Update system prompt and main prompt for preview (apply to temporary_questionnaire)
            # st.session_state.temporary_questionnaire.system_prompt = new_system_prompt
            # st.session_state.temporary_questionnaire.prompt = new_prompt
            # current_system_prompt, current_prompt = st.session_state.temporary_questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.SEQUENTIAL)
            # current_system_prompt = current_system_prompt.replace("\n", "  \n")
            # current_prompt = current_prompt.replace("\n", "  \n")
            # st.write(current_system_prompt)
            # st.write(current_prompt)
            
            
            # Initialize preview state if it doesn't exist
            preview_key = f"preview_{current_questionnaire_id}"
            if preview_key not in st.session_state:
                # Create initial preview
                if "survey_options" in st.session_state:
                    survey_options = st.session_state.survey_options
                else:
                    survey_options = None
                
                temp_questionnaire = st.session_state.temporary_questionnaire.duplicate()
                temp_questionnaire.prepare_prompt(
                    question_stem=question_stem_input,
                    answer_options=survey_options,
                    randomized_item_order=False,
                )
                temp_questionnaire.system_prompt = questionnaire.system_prompt
                temp_questionnaire.prompt = questionnaire.prompt
                current_system_prompt, current_prompt = temp_questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.SEQUENTIAL)
                current_system_prompt = current_system_prompt.replace("\n", "  \n")
                current_prompt = current_prompt.replace("\n", "  \n")
                st.session_state[preview_key] = {
                    "system_prompt": current_system_prompt,
                    "prompt": current_prompt
                }
            
            # Display the stored preview
            st.write(st.session_state[preview_key]["system_prompt"])
            st.write(st.session_state[preview_key]["prompt"])

    st.divider()

    if st.button("Update Prompt(s)", type="secondary", use_container_width=True):
        # Get current values from text areas (they're already in session state via the keys)
        current_system_value = st.session_state.get(system_prompt_key, new_system_prompt)
        current_prompt_value = st.session_state.get(prompt_key, new_prompt)
        
        if change_all_system:
            for questionnaire in st.session_state.questionnaires:
                questionnaire.system_prompt = current_system_value
        else:
            st.session_state.questionnaires[current_questionnaire_id].system_prompt = current_system_value

        if change_all_questionnaire:
            for questionnaire in st.session_state.questionnaires:
                questionnaire.prompt = current_prompt_value
        else:
            st.session_state.questionnaires[current_questionnaire_id].prompt = current_prompt_value
        
        # Update the preview when Update Prompt(s) is clicked
        preview_key = f"preview_{current_questionnaire_id}"
        if "survey_options" in st.session_state:
            survey_options = st.session_state.survey_options
        else:
            survey_options = None
        
        temp_questionnaire = st.session_state.temporary_questionnaire.duplicate()
        if randomize_order_bool:
            temp_questionnaire.prepare_prompt(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=randomize_order_bool,
            )
        else:
            temp_questionnaire.prepare_prompt(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=False,
            )
        
        temp_questionnaire.system_prompt = current_system_value
        temp_questionnaire.prompt = current_prompt_value
        preview_system_prompt, preview_prompt = temp_questionnaire.get_prompt_for_questionnaire_type(QuestionnairePresentation.SEQUENTIAL)
        preview_system_prompt = preview_system_prompt.replace("\n", "  \n")
        preview_prompt = preview_prompt.replace("\n", "  \n")
        st.session_state[preview_key] = {
            "system_prompt": preview_system_prompt,
            "prompt": preview_prompt
        }
             
        st.success("Prompt(s) updated!")
        st.rerun()

    if st.button("Confirm and Prepare Questionnaire", type="primary", use_container_width=True):
        # Get survey options if they exist
        if "survey_options" in st.session_state:
            survey_options = st.session_state.survey_options
        else:
            survey_options = None
        
        # Get current system prompt and main prompt values
        current_system_value = st.session_state.get(system_prompt_key, new_system_prompt)
        current_prompt_value = st.session_state.get(prompt_key, new_prompt)
        
        for questionnaire in st.session_state.questionnaires:
            # Update system prompt and main prompt
            if change_all_system:
                questionnaire.system_prompt = current_system_value
            else:
                # Only update if it's the current questionnaire or if we're updating all
                if questionnaire == st.session_state.questionnaires[current_questionnaire_id]:
                    questionnaire.system_prompt = current_system_value
            
            if change_all_questionnaire:
                questionnaire.prompt = current_prompt_value
            else:
                # Only update if it's the current questionnaire or if we're updating all
                if questionnaire == st.session_state.questionnaires[current_questionnaire_id]:
                    questionnaire.prompt = current_prompt_value
            
            # Update question stem and answer options
            questionnaire.prepare_prompt(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=randomize_order_bool,
            )
        st.success("Changed the prompts!")
        st.switch_page("pages/03_Inference_Setting.py")
else:
    st.warning("No data found. Please upload a CSV file on the 'Start Page' first.")
