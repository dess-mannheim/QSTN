import streamlit as st
from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.utilities.prompt_templates import (
    LIST_OPTIONS_DEFAULT,
    SCALE_OPTIONS_DEFAULT,
)
from qstn.inference.response_generation import (
    JSONSingleResponseGenerationMethod,
    JSONVerbalizedDistribution,
    JSONReasoningResponseGenerationMethod,
    ChoiceResponseGenerationMethod,
    LogprobResponseGenerationMethod,
)
from qstn.utilities import constants

from gui_elements.stateful_widget import StatefulWidgets

st.set_page_config(layout="wide")
st.title("Likert Scale Options Generator")
st.write(
    "This interface allows you to configure and generate Likert scale answer options by adjusting the parameters below."
)
st.divider()

if "questionnaires" not in st.session_state:
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

    # --- Response Generation Method ---
    st.subheader("Response Generation Method")
    st.write("Choose how the model output should be constrained. This controls the format of responses.")
    
    # Global option for all response generation methods
    output_index_only = state.create(
        st.checkbox,
        "output_index_only",
        "Output Indices Only",
        initial_value=False,
        help="If checked, the model will output only indices (e.g., '1', '2', '3') instead of full text (e.g., '1: Strongly Disagree'). Applies to all response generation methods.",
    )
    
    rgm_type = state.create(
        st.selectbox,
        "rgm_type",
        "Response Generation Method",
        options=["None", "JSON Single Answer", "JSON All Options (Probabilities)", "JSON with Reasoning", "Choice"],  # "Logprob" commented out - not fully implemented
        initial_value="None",
        help="None: Free text output (requires parsing). JSON: Structured JSON output. Choice: Guided choice selection.",
    )
    
    # Show additional options based on selected method
    rgm_config = {}
    
    # Choice configuration - allow user to specify choices
    if rgm_type == "Choice":
        with st.container(border=True):
            st.write("**Choice Configuration:**")
            # Pre-fill with answer texts if available
            if answer_texts:
                answer_texts_list_preview = [text.strip() for text in answer_texts.split("\n") if text.strip()]
                # Format as full text by default: "1: Option 1, 2: Option 2"
                if output_index_only:
                    # If indices only, just use numbers
                    default_choices = "\n".join([str(i+1) for i in range(len(answer_texts_list_preview))])
                else:
                    # Full text format
                    default_choices = "\n".join([f"{i+1}: {text}" for i, text in enumerate(answer_texts_list_preview)])
            else:
                default_choices = ""
            
            choice_input = state.create(
                st.text_area,
                "choice_allowed_choices",
                "Allowed Choices (one per line)",
                initial_value=default_choices,
                height=100,
                help="Enter the choices the model can output. Format: indices only (1, 2, 3) or full text (1: Strongly Disagree, 2: Disagree, etc.).",
            )
            rgm_config["allowed_choices"] = choice_input
    
    # Logprob configuration - commented out until fully implemented
    # if rgm_type == "Logprob":
    #     with st.container(border=True):
    #         st.write("**Logprob Configuration:**")
    #         rgm_config["token_position"] = state.create(
    #             st.number_input,
    #             "logprob_token_position",
    #             "Token Position",
    #             initial_value=0,
    #             min_value=0,
    #             help="Position in output to capture logprobs (0 = first token).",
    #         )
    #         rgm_config["token_limit"] = state.create(
    #             st.number_input,
    #             "logprob_token_limit",
    #             "Token Limit",
    #             initial_value=1,
    #             min_value=1,
    #             help="Maximum number of tokens to generate.",
    #         )
    #         rgm_config["top_logprobs"] = state.create(
    #             st.number_input,
    #             "logprob_top_logprobs",
    #             "Top Logprobs",
    #             initial_value=20,
    #             min_value=1,
    #             max_value=20,
    #             help="Number of top probabilities to return (max 20).",
    #         )
    #         rgm_config["ignore_reasoning"] = state.create(
    #             st.checkbox,
    #             "logprob_ignore_reasoning",
    #             "Ignore Reasoning",
    #             initial_value=True,
    #             help="Only consider tokens after reasoning output.",
    #         )


    # The submit button for the form
    submitted = st.button("Confirm and Generate Options", disabled=disabled, type="primary", use_container_width=True)

    if st.button("Skip", use_container_width=True, icon="❌"):
        st.session_state.survey_options = None
        st.switch_page("pages/02_Prompt_Configuration.py")

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
        # Create response generation method based on user selection
        response_generation_method = None
        if rgm_type == "JSON Single Answer":
            response_generation_method = JSONSingleResponseGenerationMethod(
                automatic_output_instructions=True,
                output_index_only=output_index_only,
            )
        elif rgm_type == "JSON All Options (Probabilities)":
            response_generation_method = JSONVerbalizedDistribution(
                automatic_system_prompt=False,
                output_index_only=output_index_only,
            )
        elif rgm_type == "JSON with Reasoning":
            response_generation_method = JSONReasoningResponseGenerationMethod(
                automatic_output_instructions=True,
                output_index_only=output_index_only,
            )
        elif rgm_type == "Choice":
            # Parse the choices from the text input
            choice_text = rgm_config.get("allowed_choices", "")
            if choice_text:
                # Split by newlines and clean up
                allowed_choices_list = [choice.strip() for choice in choice_text.split("\n") if choice.strip()]
            else:
                # Fallback to OPTIONS_ADJUST if no input (will be auto-configured)
                allowed_choices_list = constants.OPTIONS_ADJUST
            
            response_generation_method = ChoiceResponseGenerationMethod(
                allowed_choices=allowed_choices_list,
                automatic_output_instructions=True,
                output_index_only=output_index_only,
            )
        # Logprob - commented out until fully implemented
        # elif rgm_type == "Logprob":
        #     response_generation_method = LogprobResponseGenerationMethod(
        #         token_position=rgm_config.get("token_position", 0),
        #         token_limit=rgm_config.get("token_limit", 1),
        #         top_logprobs=rgm_config.get("top_logprobs", 20),
        #         ignore_reasoning=rgm_config.get("ignore_reasoning", True),
        #         automatic_output_instructions=True,
        #         output_index_only=False,
        #     )
        # If rgm_type == "None", response_generation_method stays None
        
        survey_options = generate_likert_options(
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
            response_generation_method=response_generation_method,
        )

        st.session_state.survey_options = survey_options
        st.switch_page("pages/02_Prompt_Configuration.py")


