import streamlit as st
from surveygen.survey_manager import SurveyOptionGenerator
from surveygen.llm_interview import LLMInterview
from surveygen.utilities.constants import InterviewType
from typing import Any

from gui_elements.paginator import paginator

# CONSTANTS FOR FIELDS
question_stem_field = "Question Stem"
randomize_order_tick = "Randomize the order of items"

field_ids = [question_stem_field, randomize_order_tick]

st.title("Questions Preparation")
st.write(
    "This interface allows you configure how the questions are prompted to the LLM. These options are applied to every questionnaire in your survey."
)
st.page_link("pages/02_Option_Prompt.py", label="Click here to adjust the answer options.")
st.divider()

#current_interview_id = paginator(st.session_state.interviews, "current_interview_index_prepare")

if "interviews" not in st.session_state:
    st.error("You need to first upload a questionnaire and the population you want to survey.")
    st.stop()
    disabled = True
else:
    disabled = False
if not "temporary_interview" in st.session_state:
    st.session_state.temporary_interview = st.session_state.interviews[0].duplicate()

    #print(st.session_state.temporary_interview._questions)

if not "base_interview" in st.session_state:
    st.session_state.base_interview = st.session_state.temporary_interview.duplicate()

def process_inputs(input: Any, field_id: str) -> str:
    if "survey_options" in st.session_state:
        survey_options = st.session_state.survey_options
    else:
        survey_options = None

    if field_id == question_stem_field:
        LLMInterview.prepare_interview
        st.session_state.temporary_interview.prepare_interview(
            question_stem=input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
        st.session_state.base_interview.prepare_interview(
            question_stem=input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
    # elif field_id == global_options_tick:
    #     option_behavior = input == "Give instruction in the beginning"
    #     st.session_state.temporary_interview.prepare_interview(
    #         question_stem=question_stem_input,
    #         answer_options=survey_options,
    #         global_options=option_behavior,
    #         randomized_item_order=randomize_order_bool,
    #     )
    #     st.session_state.base_interview.prepare_interview(
    #         question_stem=question_stem_input,
    #         answer_options=survey_options,
    #         global_options=option_behavior,
    #         randomized_item_order=False,
    #     )
    elif field_id == randomize_order_tick:
        if input == True:
            st.session_state.temporary_interview.prepare_interview(
                question_stem=question_stem_input,
                answer_options=survey_options,
                randomized_item_order=input,
            )
        else:
            st.session_state.temporary_interview = st.session_state.base_interview.duplicate()

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


col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("⚙️ Configuration")

    for field_id in field_ids:
        input_key = f"input_{field_id}"
        if not input_key in st.session_state:
            if field_id == question_stem_field:
                st.session_state[input_key] = st.session_state.temporary_interview ._questions[0].question_stem
            if field_id == randomize_order_tick:
                st.session_state[input_key] = False

    # --- Input Widgets (No Form) ---
    question_stem_input = st.text_area(
        question_stem_field,
        key=f"input_{question_stem_field}",
        # placeholder="e.g., How would you rate the following aspects of our service?",
        #on_change=handle_change,
        kwargs={'field_id': question_stem_field},
        height=100,
    )

    # option_behavior = st.radio(
    #     global_options_tick,
    #     key=f"input_{global_options_tick}",
    #     options=[
    #         "Give instruction in the beginning",
    #         "Give options after each question",
    #     ],
    #     index=0,
    #     #on_change=handle_change,
    #     kwargs={'field_id': global_options_tick},
    #     help="Choose how answer options are applied.",
    # )

    randomize_order_bool = st.checkbox(
        randomize_order_tick,
        key=f"input_{randomize_order_tick}",
        value=False,
        #on_change=handle_change,
        kwargs={'field_id': randomize_order_tick} 
    )

if "survey_options" in st.session_state:
    survey_options = st.session_state.survey_options
else:
    survey_options = None

if randomize_order_bool:
    st.session_state.temporary_interview.prepare_interview(
        question_stem=question_stem_input,
        answer_options=survey_options,
        randomized_item_order=randomize_order_bool,
    )
st.session_state.base_interview.prepare_interview(
    question_stem=question_stem_input,
    answer_options=survey_options,
    randomized_item_order=False,
)

if not randomize_order_bool:
    st.session_state.temporary_interview = st.session_state.base_interview.duplicate()

with col2:
    st.subheader("📄 Live Preview")

    #@Ahmed All of these could be a resusable function (it is used on almost all pages) Maybe split up the container in system prompt/ prompt
    with st.container(border=True):
        system_prompt, current_prompt = st.session_state.temporary_interview.get_prompt_for_interview_type(InterviewType.ONE_PROMPT)
        # markdown newlines
        system_prompt = system_prompt.replace("\n", "  \n")
        current_prompt = current_prompt.replace("\n", "  \n")
        st.write(system_prompt)
        st.write(current_prompt)

st.divider()

if st.button("Confirm and Prepare Interview", type="primary", use_container_width=True):
    for interview in st.session_state.interviews:
        interview.prepare_interview(
            question_stem=question_stem_input,
            answer_options=survey_options,
            randomized_item_order=randomize_order_bool,
        )
    st.success("Changed the prompts!")
