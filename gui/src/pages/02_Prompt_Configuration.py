import streamlit as st
from qstn.prompt_builder import LLMPrompt
from qstn.utilities.constants import QuestionnairePresentation
from qstn.utilities import placeholder
from typing import Any
import json

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

current_questionnaire_id = paginator(st.session_state.questionnaires, "current_questionnaire_index_prompt")

# Helper functions
def get_survey_options():
    """Helper to get survey options from session state."""
    return st.session_state.get("survey_options")

def get_randomize_order_bool():
    """Helper to get randomize order boolean."""
    return st.session_state.get(f"input_{randomize_order_tick}", False)

def clear_questionnaire_keys(questionnaire_id):
    """Helper to clear session state keys for a questionnaire."""
    keys_to_clear = [
        f"system_prompt_textarea_{questionnaire_id}",
        f"prompt_textarea_{questionnaire_id}",
        f"input_{question_stem_field}_{questionnaire_id}",
        f"preview_{questionnaire_id}"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def extract_placeholders(text):
    """Extract all placeholder strings from text."""
    all_placeholders = [
        placeholder.PROMPT_QUESTIONS,
        placeholder.PROMPT_OPTIONS,
        placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS,
        placeholder.QUESTION_CONTENT,
        placeholder.JSON_TEMPLATE,
    ]
    found_placeholders = []
    for ph in all_placeholders:
        if ph in text:
            found_placeholders.append(ph)
    return found_placeholders

def add_missing_placeholders(target_text, placeholders_to_add, source_text):
    """Add or update placeholders in target text to match source formatting."""
    if not placeholders_to_add:
        return target_text
    
    result = target_text.rstrip()
    
    # First, remove any existing placeholders from the target to avoid duplicates
    # and to allow re-formatting them
    for ph in placeholders_to_add:
        if ph in result:
            # Remove the placeholder and any surrounding whitespace/newlines
            # This allows us to re-add it in the correct format
            result = result.replace(ph, '')
            # Clean up extra whitespace/newlines that might be left
            result = result.replace('\n\n\n', '\n\n')  # Remove triple newlines
            result = result.replace('  ', ' ')  # Remove double spaces
            result = result.rstrip()
    
    # Determine formatting from source text
    # Check if any placeholder in source appears on the same line (not after a newline)
    same_line_format = False
    for ph in placeholders_to_add:
        if ph in source_text:
            # Find the placeholder in source
            ph_index = source_text.find(ph)
            # Check the character immediately before the placeholder
            # If it's a newline, the placeholder is on a new line
            # If it's a space or other character (not newline), it's on the same line
            if ph_index > 0:
                char_before = source_text[ph_index - 1]
                if char_before != '\n':
                    # Placeholder is on same line (not preceded by newline)
                    same_line_format = True
                    break
            # If placeholder is at start of text (ph_index == 0), treat as new line
            # If preceded by newline, also treat as new line
    
    # Add placeholders in the correct format
    if same_line_format:
        # Add placeholders on same line with space before
        if result:
            result += ' '
        result += ' '.join(placeholders_to_add)
    else:
        # Add placeholders on new line(s)
        if result and not result.endswith('\n'):
            result += '\n'
        result += '\n'.join(placeholders_to_add)
    
    return result

def generate_preview(questionnaire, survey_options, randomize_order):
    """Helper to generate preview content."""
    temp_q = questionnaire.duplicate()
    current_question_stem = (
        questionnaire._questions[0].question_stem 
        if questionnaire._questions else ""
    )
    temp_q.prepare_prompt(
        question_stem=current_question_stem,
        answer_options=survey_options,
        randomized_item_order=randomize_order,
    )
    temp_q.system_prompt = questionnaire.system_prompt
    temp_q.prompt = questionnaire.prompt
    system_prompt, prompt = temp_q.get_prompt_for_questionnaire_type(QuestionnairePresentation.SEQUENTIAL)
    return system_prompt.replace("\n", "  \n"), prompt.replace("\n", "  \n")

# Initialize or update temporary_questionnaire to match current questionnaire
if "temporary_questionnaire" not in st.session_state:
    st.session_state.temporary_questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()
    st.session_state.temporary_questionnaire_id = current_questionnaire_id
elif st.session_state.get("temporary_questionnaire_id") != current_questionnaire_id:
    # User switched questionnaires, store old ID before updating
    old_questionnaire_id = st.session_state.get("temporary_questionnaire_id")
    st.session_state.temporary_questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()
    st.session_state.temporary_questionnaire_id = current_questionnaire_id
    
    # Clear old text area keys so they get re-initialized from the new temporary_questionnaire
    if old_questionnaire_id is not None:
        clear_questionnaire_keys(old_questionnaire_id)
    
    # Clear keys for the NEW questionnaire so they get re-initialized from temporary_questionnaire
    # This ensures we use fresh values from the actual questionnaire, not stale session state
    new_system_key = f"system_prompt_textarea_{current_questionnaire_id}"
    new_prompt_key = f"prompt_textarea_{current_questionnaire_id}"
    new_question_stem_key = f"input_{question_stem_field}_{current_questionnaire_id}"
    clear_questionnaire_keys(current_questionnaire_id)
    
    # Explicitly set the keys to values from temporary_questionnaire after clearing
    # This ensures the widgets initialize with the correct values, not stale session state
    st.session_state[new_system_key] = st.session_state.temporary_questionnaire.system_prompt
    st.session_state[new_prompt_key] = st.session_state.temporary_questionnaire.prompt
    new_question_stem_default = st.session_state.temporary_questionnaire._questions[0].question_stem if st.session_state.temporary_questionnaire._questions else ""
    st.session_state[new_question_stem_key] = new_question_stem_default

if not "base_questionnaire" in st.session_state:
    st.session_state.base_questionnaire = st.session_state.temporary_questionnaire.duplicate()
elif st.session_state.get("temporary_questionnaire_id") != current_questionnaire_id:
    # Update base_questionnaire when switching questionnaires
    st.session_state.base_questionnaire = st.session_state.temporary_questionnaire.duplicate()

if "questionnaires" in st.session_state and st.session_state.questionnaires is not None:
    try:
        # Validate questionnaire index
        _ = st.session_state.questionnaires[current_questionnaire_id]
    except IndexError:
        st.error("Index is out of range. Resetting to the first item.")
        current_questionnaire_id = 0
        st.session_state.temporary_questionnaire = st.session_state.questionnaires[current_questionnaire_id].duplicate()
        st.session_state.temporary_questionnaire_id = current_questionnaire_id

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

        # System prompt and main prompt section
        # Handle placeholder replacement for both textboxes before widgets are created
        if "unified_placeholder_to_replace" in st.session_state and st.session_state.unified_placeholder_to_replace:
            current_system_text = st.session_state.temporary_questionnaire.system_prompt
            current_prompt_text = st.session_state.temporary_questionnaire.prompt
            placeholder_shortcut = st.session_state.unified_placeholder_to_replace["shortcut"]
            placeholder_value = st.session_state.unified_placeholder_to_replace["value"]
            target_textbox = st.session_state.unified_placeholder_to_replace.get("target", "prompt")  # "system" or "prompt"
            
            # Check if placeholder already exists in EITHER textbox
            if placeholder_value in current_system_text or placeholder_value in current_prompt_text:
                st.session_state.unified_placeholder_warning = f"⚠️ The placeholder `{placeholder_value}` already exists in one of the textboxes. Please remove it first if you want to insert it again."
            else:
                # Check for shortcut in both textboxes - replace where found
                if placeholder_shortcut in current_system_text:
                    st.session_state.temporary_questionnaire.system_prompt = current_system_text.replace(placeholder_shortcut, placeholder_value)
                elif placeholder_shortcut in current_prompt_text:
                    st.session_state.temporary_questionnaire.prompt = current_prompt_text.replace(placeholder_shortcut, placeholder_value)
                else:
                    # No shortcut found, add to the target textbox (default based on which was clicked)
                    if target_textbox == "system":
                        st.session_state.temporary_questionnaire.system_prompt = current_system_text + f" {placeholder_value} "
                    else:
                        st.session_state.temporary_questionnaire.prompt = current_prompt_text + f" {placeholder_value} "
            
            st.session_state.unified_placeholder_to_replace = None
            st.rerun()
        
        # Display warnings if they exist
        if "unified_placeholder_warning" in st.session_state and st.session_state.unified_placeholder_warning:
            st.warning(st.session_state.unified_placeholder_warning)
            st.session_state.unified_placeholder_warning = None  # Clear after displaying
        
        # Use temporary_questionnaire for text areas (temporary edits)
        system_prompt_key = f"system_prompt_textarea_{current_questionnaire_id}"
        prompt_key = f"prompt_textarea_{current_questionnaire_id}"
        
        # Initialize session state keys from temporary_questionnaire if they don't exist
        # (This happens on first load or when switching to a new questionnaire)
        if system_prompt_key not in st.session_state:
            st.session_state[system_prompt_key] = st.session_state.temporary_questionnaire.system_prompt
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = st.session_state.temporary_questionnaire.prompt
        
        new_system_prompt = st.text_area(
            label=system_prompt_field,
            help="The system prompt the model is prompted with.",
            key=system_prompt_key,
        )
        # Sync from session state to temporary_questionnaire (captures user edits)
        st.session_state.temporary_questionnaire.system_prompt = st.session_state[system_prompt_key]

        # Display warning for main prompt if it exists
        if "main_prompt_warning" in st.session_state and st.session_state.main_prompt_warning:
            st.warning(st.session_state.main_prompt_warning)
            st.session_state.main_prompt_warning = None  # Clear after displaying
        
        new_prompt = st.text_area(
            label=prompt_field,
            help="Instructions that are given to the model before the questions.",
            key=prompt_key,
        )
        # Sync from session state to temporary_questionnaire (captures user edits)
        st.session_state.temporary_questionnaire.prompt = st.session_state[prompt_key]

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

        # Initialize field keys if they don't exist
        for field_id in field_ids:
            if field_id == question_stem_field:
                input_key = f"input_{field_id}_{current_questionnaire_id}"
            else:
                input_key = f"input_{field_id}"
            if input_key not in st.session_state:
                if field_id == question_stem_field:
                    st.session_state[input_key] = st.session_state.temporary_questionnaire._questions[0].question_stem
                if field_id == randomize_order_tick:
                    st.session_state[input_key] = False

        # Handle placeholder replacement before widget is created
        input_key = f"input_{question_stem_field}_{current_questionnaire_id}"
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

        # --- Input Widgets ---
        # Display warning for question stem if it exists
        if "question_stem_warning" in st.session_state and st.session_state.question_stem_warning:
            st.warning(st.session_state.question_stem_warning)
            st.session_state.question_stem_warning = None  # Clear after displaying
        
        # Use questionnaire-specific key (consistent with system_prompt and prompt)
        # Streamlit will automatically use the session_state value when key is provided
        input_key = f"input_{question_stem_field}_{current_questionnaire_id}"
        question_stem_input = st.text_area(
            question_stem_field,
            key=input_key,
            height=100,
        )
        # Sync from session state to temporary_questionnaire (captures user edits)
        # This ensures consistency with system_prompt and prompt fields
        survey_options = get_survey_options()
        randomize_order_bool = get_randomize_order_bool()
        st.session_state.temporary_questionnaire.prepare_prompt(
            question_stem=st.session_state[input_key],
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
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
        )

        st.divider()

    # Place the corresponding output in the second column
    with col2:
        st.subheader("📄 Live Preview")
        # Preview only updates when "Update Prompt(s)" is clicked
        with st.container(border=True):
            # Initialize preview state if it doesn't exist
            preview_key = f"preview_{current_questionnaire_id}"
            if preview_key not in st.session_state:
                # Create initial preview
                survey_options = get_survey_options()
                system_prompt, prompt = generate_preview(
                    st.session_state.temporary_questionnaire,
                    survey_options,
                    False
                )
                st.session_state[preview_key] = {
                    "system_prompt": system_prompt,
                    "prompt": prompt
                }
            
            # Display the stored preview
            # Add a visible questionnaire identifier to force Streamlit to update when switching
            # This ensures the preview updates even when "change all" makes content identical
            st.caption(f"📋 Questionnaire {current_questionnaire_id + 1} of {len(st.session_state.questionnaires)}")
            # Use a unique container key for each questionnaire to force Streamlit to update when switching
            # This ensures the preview updates even when content is identical
            with st.container(key=f"preview_container_{current_questionnaire_id}"):
                st.markdown(st.session_state[preview_key]["system_prompt"])
                st.write(st.session_state[preview_key]["prompt"])

    st.divider()

    if st.button("Update Preview", type="secondary", use_container_width=True):
        # Update the preview when Update Prompt(s) is clicked
        # Use temporary_questionnaire directly (already has all the edits)
        preview_key = f"preview_{current_questionnaire_id}"
        survey_options = get_survey_options()
        randomize_order_bool = get_randomize_order_bool()
        
        system_prompt, prompt = generate_preview(
            st.session_state.temporary_questionnaire,
            survey_options,
            randomize_order_bool
        )
        st.session_state[preview_key] = {
            "system_prompt": system_prompt,
            "prompt": prompt
        }
             
        st.success("Prompt(s) updated!")
        st.rerun()

    if st.button("Confirm and Prepare Questionnaire", type="primary", use_container_width=True):
        # Get survey options if they exist
        survey_options = get_survey_options()
        randomize_order_bool = get_randomize_order_bool()
        
        # Copy from temporary_questionnaire to actual questionnaires
        # All values come from temporary_questionnaire (now fully synced with all edits)
        current_system_value = st.session_state.temporary_questionnaire.system_prompt
        current_prompt_value = st.session_state.temporary_questionnaire.prompt
        current_question_stem = (
            st.session_state.temporary_questionnaire._questions[0].question_stem 
            if st.session_state.temporary_questionnaire._questions 
            else ""
        )
        
        # Extract placeholders from current system prompt if "change all System Prompts" is checked
        placeholders_to_add = []
        if change_all_system:
            placeholders_to_add = extract_placeholders(current_system_value)
        
        for idx, questionnaire in enumerate(st.session_state.questionnaires):
            # Determine if this questionnaire should be updated
            # System prompt: current questionnaire gets full update, others get placeholders added (if checkbox checked)
            should_update_system = (idx == current_questionnaire_id)
            should_update_prompt = change_all_questionnaire or (idx == current_questionnaire_id)
            should_update = should_update_system or should_update_prompt or (change_all_system and idx != current_questionnaire_id)
            
            if should_update:
                # Update system prompt: full update for current, add placeholders for others
                if should_update_system:
                    # Current questionnaire: update entire system prompt
                    questionnaire.system_prompt = current_system_value
                elif change_all_system and idx != current_questionnaire_id:
                    # Other questionnaires: add missing placeholders while preserving base content
                    questionnaire.system_prompt = add_missing_placeholders(
                        questionnaire.system_prompt,
                        placeholders_to_add,
                        current_system_value
                    )
                
                if should_update_prompt:
                    questionnaire.prompt = current_prompt_value
                
                # Call prepare_prompt for questionnaires being updated
                # Use question stem from temporary_questionnaire (consistent with other fields)
                questionnaire.prepare_prompt(
                    question_stem=current_question_stem,
                    answer_options=survey_options,
                    randomized_item_order=randomize_order_bool,
                )
        st.success("Changed the prompts!")
        
        # Auto-save session
        from gui_elements.session_cache import save_session_state
        save_session_state()
        
        st.switch_page("pages/03_Inference_Setting.py")
else:
    st.warning("No data found. Please upload a CSV file on the 'Start Page' first.")
