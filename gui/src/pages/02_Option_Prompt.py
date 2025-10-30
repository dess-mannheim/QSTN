import streamlit as st
from surveygen.survey_manager import SurveyOptionGenerator
from surveygen.llm_interview import LLMInterview
from surveygen.utilities.prompt_templates import (
    LIST_OPTIONS_DEFAULT,
    SCALE_OPTIONS_DEFAULT,
)

from gui_elements.stateful_widget import StatefulWidgets

st.set_page_config(layout="wide")
st.title("Likert Scale Options Generator")
st.write(
    "This interface allows you to configure and generate Likert scale answer options by adjusting the parameters below."
)
st.divider()

if "interviews" not in st.session_state:
    st.error("You need to first upload a questionnaire and the population you want to survey.")
    st.stop()
    disabled = True
else:
    disabled = False

#if 'answer_texts_input' not in st.session_state:
    #st.session_state.answer_texts_input = "Strongly Disagree\nDisagree\nNeutral\nAgree\nStrongly Agree"

state = StatefulWidgets()

# Use a form to batch all inputs together
with st.container(border=True):
    # --- Main Configuration ---
    st.subheader("Main Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        n_options = state.create(
            st.number_input,
            "n_options",
            "Number of Options (n)",
            initial_value=5,
            min_value=2,
            step=1,
            help="The total number of choices in the scale.",
        )

    with col2:
        idx_type = state.create(
            st.selectbox,
            "idx_type",
            "Index Type",
            initial_value="integer",
            options=["integer", "char_low", "char_up"],
            help="The type of index to use for the options.",
        )

    with col3:
        start_idx = state.create(
            st.number_input,
            "start_idx",
            "Starting Index",
            initial_value=1,
            step=1,
            help="The number to start counting from (e.g., 1).",
        )

    # --- Order and Structure Options ---
    st.subheader("Ordering and Structure")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        only_from_to_scale = state.create(
            st.checkbox,
            "only_from_to_scale",
            "From-To Scale Only",
            initial_value=False,
            help="If checked, only the first and last answer labels are display e.g. 1 Strongly Disagree to 5 Strongly agree.",
        )

    with col2:
        random_order = state.create(
            st.checkbox,
            "random_order",
            "Random Order",
            initial_value=False,
            help="Randomize the order of options.",
        )

    with col3:
        reversed_order = state.create(
            st.checkbox,
            "reversed_order",
            "Reversed Order",
            initial_value=False,
            help="Reverse the order of options.",
        )

    with col4:
        even_order = state.create(
            st.checkbox,
            "even_order",
            "Even Order",
            initial_value=False,
            help="If there is an uneven number of answer texts, the middle section is automatically removed.",
        )

    # --- Answer Texts Input ---
    st.subheader("Answer Texts")

    answer_texts = state.create(
        st.text_area,
        "answer_texts",
        "Enter Answer Texts (one per line)",
        initial_value="Strongly Disagree\nDisagree\nNeutral\nAgree\nStrongly Agree",
        height=150,
        help="Enter the labels for each answer option.",
    )

    # --- Advanced Configuration ---
    with st.expander("Advanced Configuration"):
        options_separator = state.create(
            st.text_input,
            "options_separator",
            "Options Separator",
            initial_value=", ",
            help="The character(s) used to separate options in the final string.",
        )
        list_prompt_template = state.create(
            st.text_area,
            "list_prompt_template",
            "List Prompt Template",
            initial_value=LIST_OPTIONS_DEFAULT,
            height=100,
            help="Write how the options should be presented to the model.",
        )
        scale_prompt_template = state.create(
            st.text_area,
            "scale_prompt_template",
            "Scale Prompt Template",
            initial_value=SCALE_OPTIONS_DEFAULT,
            height=100,
            help="Write how the options should be presented to the model.",
        )


    # The submit button for the form
    submitted = st.button("Confirm and Generate Options", disabled=disabled, type="primary", use_container_width=True)

    if st.button("Remove all options", use_container_width=True, icon="❌"):
        st.session_state.survey_options = None
        st.switch_page("pages/03_Prepare_Prompts.py")

# --- Processing and Output ---
if submitted:
    #print("Session state answer texts "+ st.session_state.answer_texts_input)
    #print(answer_texts_input)
    # Convert the raw text area string into a list of strings.
    answer_texts_list = [
        text.strip() for text in answer_texts.split("\n") if text.strip()
    ]

    # --- Input Validation ---
    validation_ok = True
    if only_from_to_scale and len(answer_texts_list) != 2:
        st.error(
            f"Error: When 'From-To Scale Only' is selected, you must provide exactly 2 answer texts. You provided {len(answer_texts_list)}."
        )
        validation_ok = False

    if not only_from_to_scale and len(answer_texts_list) != n_options:
        st.error(
            f"Error: The number of answer texts ({len(answer_texts_list)}) must match the 'Number of Options' ({n_options})."
        )
        validation_ok = False

    if reversed_order and random_order:
        st.error(f"Error: Reversed Order and Random Order cannot both be true.")
        validation_ok = False

    if validation_ok:
        survey_options = SurveyOptionGenerator.generate_likert_options(
            n=n_options,
            answer_texts=answer_texts_list,
            only_from_to_scale=only_from_to_scale,
            random_order=random_order,
            reversed_order=reversed_order,
            even_order=even_order,
            start_idx=start_idx,
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            options_separator=options_separator,
            idx_type=idx_type,
        )

        st.session_state.survey_options = survey_options
        st.switch_page("pages/03_Prepare_Prompts.py")


