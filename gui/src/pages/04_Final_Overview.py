import streamlit as st
from gui_elements.paginator import paginator
from gui_elements.stateful_widget import StatefulWidgets
from gui_elements.output_manager import st_capture, TqdmToStreamlit

import io
import queue
import time
import threading
import asyncio
import pandas as pd

import logging

from contextlib import redirect_stderr, redirect_stdout

from qstn.parser.llm_answer_parser import raw_responses
from qstn.utilities.constants import QuestionnairePresentation
from qstn.utilities.utils import create_one_dataframe
from qstn.survey_manager import (
    conduct_survey_sequential,
    conduct_survey_battery,
    conduct_survey_single_item,
)

from streamlit.runtime.scriptrunner import add_script_run_ctx

from openai import AsyncOpenAI

# Set OpenAI's API key and API base to use vLLM's API server.


if "questionnaires" not in st.session_state:
    st.error(
        "You need to first upload a questionnaire and the population you want to survey."
    )
    st.stop()
    disabled = True
else:
    disabled = False


@st.cache_data
def create_stateful_widget() -> StatefulWidgets:
    return StatefulWidgets()


state = create_stateful_widget()

current_index = paginator(st.session_state.questionnaires, "overview_page")

questionnaires = st.session_state.questionnaires[current_index]

col_llm, col_prompt_display = st.columns(2)

with col_llm:
    st.subheader("⚙️ Inference Parameters")

    with st.container(border=True):
        st.subheader("Core Settings")
        model_name = state.create(
            st.text_input,
            "model_name",
            "Model Name",
            # initial_value="meta-llama/Llama-3.1-70B-Instruct",
            # placeholder="meta-llama/Llama-3.1-70B-Instruct",
            disabled=True,
            help="The model to use for the inference call.",
        )

        temperature = state.create(
            st.slider,
            "temperature",
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            initial_value=1.0,
            disabled=True,
            help="Controls randomness. Lower values are more deterministic and less creative.",
        )

        max_tokens = state.create(
            st.number_input,
            "max_tokens",
            "Max Tokens",
            initial_value=1024,
            min_value=1,
            disabled=True,
            help="The maximum number of tokens to generate in the completion.",
        )

        top_p = state.create(
            st.slider,
            "top_p",
            "Top P",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            initial_value=1.0,
            disabled=True,
            help="Controls nucleus sampling. The model considers tokens with top_p probability mass.",
        )

        seed = state.create(
            st.number_input,
            "seed",
            "Seed",
            initial_value=42,
            min_value=0,
            disabled=True,
            help="A specific seed for reproducibility of results.",
        )

        with st.expander("Advanced Inference Settings (JSON)"):
            advanced_inference_params_str = state.create(
                st.text_area,
                "advanced_inference_params_str",
                "JSON for other inference parameters",
                initial_value="",
                # placeholder='{\n  "stop": ["\\n", " Human:"],\n  "presence_penalty": 0\n}',
                height=150,
                disabled=True,
                help='Enter any other valid inference parameters like "stop", "logit_bias", or "frequency_penalty" as a JSON object.',
            )


with col_prompt_display:
    st.subheader("📄 Live Preview")
    
    # Survey method selector
    survey_method_options = {
        "Single item": ("single_item", QuestionnairePresentation.SINGLE_ITEM),
        "Battery": ("battery", QuestionnairePresentation.BATTERY),
        "Sequential": ("sequential", QuestionnairePresentation.SEQUENTIAL),
    }
    
    survey_method_display = state.create(
        st.selectbox,
        "survey_method",
        "Questionnaire Method",
        options=list(survey_method_options.keys()),
        initial_value="Single item",
        help="Choose how to conduct the questionnaire: Single item (one at a time), Battery (all questions together), or Sequential (with conversation history)."
    )
    
    # Get the method name and questionnaire type from selection
    selected_method_name, selected_questionnaire_type = survey_method_options[survey_method_display]

    with st.container(border=True):
        # For single item mode, show multiple previews (up to 3 items)
        if selected_questionnaire_type == QuestionnairePresentation.SINGLE_ITEM:
            num_questions = len(questionnaires._questions)
            num_previews = min(3, num_questions)  # Show up to 3 previews
            
            if num_previews > 1:
                st.write(f"**Preview of first {num_previews} items:**")
            else:
                st.write("**Preview:**")
            
            for i in range(num_previews):
                if num_previews > 1:
                    st.write(f"**Item {i+1}:**")
                
                current_system_prompt, current_prompt = questionnaires.get_prompt_for_questionnaire_type(
                    selected_questionnaire_type, 
                    item_id=i
                )
                current_system_prompt = current_system_prompt.replace("\n", "  \n")
                current_prompt = current_prompt.replace("\n", "  \n")
                st.write(current_system_prompt)
                st.write(current_prompt)
                
                # Add separator between items (except for the last one)
                if i < num_previews - 1:
                    st.divider()
        else:
            # For battery and sequential, show single preview as before
            current_system_prompt, current_prompt = questionnaires.get_prompt_for_questionnaire_type(selected_questionnaire_type)
            current_system_prompt = current_system_prompt.replace("\n", "  \n")
            current_prompt = current_prompt.replace("\n", "  \n")
            st.write(current_system_prompt)
            st.write(current_prompt)


if st.button("Confirm and Run Questionnaire", type="primary", use_container_width=True):
    st.write("Starting inference...")

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = AsyncOpenAI(**st.session_state.client_config)

    inference_config = st.session_state.inference_config.copy()

    model_name = inference_config.pop("model")

    progress_text = st.empty()

    log_queue = queue.Queue()
    result_queue = queue.Queue()

    class QueueWriter:
        def __init__(self, q):
            self.q = q

        def write(self, message):
            if message.strip():
                self.q.put(message)

        def flush(self):
            # This function is needed to match the file-like object interface
            # but we don't need to do anything here.
            pass

    # Helper function for asyncronous runs
    def run_async_in_thread(
        result_q, client, questionnaires, model_name, survey_method_name, **inference_config
    ):
        queue_writer = QueueWriter(log_queue)

        # We need to redirect the output to a queue, as streamlit does not support multithreading
        # API concurrency should be  configurable in the GUI
        try:
            with redirect_stderr(queue_writer):
                # Select the appropriate survey method based on user choice
                if survey_method_name == "single_item":
                    survey_func = conduct_survey_single_item
                elif survey_method_name == "battery":
                    survey_func = conduct_survey_battery
                elif survey_method_name == "sequential":
                    survey_func = conduct_survey_sequential
                else:
                    survey_func = conduct_survey_single_item  # Default fallback
                
                result = survey_func(
                    client,
                    llm_prompts=questionnaires,
                    client_model_name=model_name,
                    api_concurrency=100,
                    **inference_config,
                )

        except Exception as e:
            result = e
            st.error(e)
        finally:
            result_q.put(result)

    while not log_queue.empty():
        log_queue.get()
    while not result_queue.empty():
        result_queue.get()  

    # Get the selected survey method
    survey_method_display = st.session_state.get("survey_method", "Single item")
    survey_method_options = {
        "Single item": ("single_item", QuestionnairePresentation.SINGLE_ITEM),
        "Battery": ("battery", QuestionnairePresentation.BATTERY),
        "Sequential": ("sequential", QuestionnairePresentation.SEQUENTIAL),
    }
    selected_method_name, _ = survey_method_options.get(survey_method_display, ("single_item", QuestionnairePresentation.SINGLE_ITEM))
    
    thread = threading.Thread(
        target=run_async_in_thread,
        args=(result_queue, client, st.session_state.questionnaires, model_name, selected_method_name),
        kwargs=inference_config,
    )
    thread.start()

    all_questions_placeholder = st.empty()
    progress_placeholder = st.empty()

    while thread.is_alive():
        try:
            # Here we can write directly to the UI, as it is the main thread
            # TQDM uses carriage returns (\r) to animate in the console, we only show clear lines
            log_message = log_queue.get_nowait()
            # This is quite a hacky solution for now, we should adjust QSTN to make the messages clearly parsable.
            if "[A" not in log_message and "Processing Prompts" not in log_message:
                all_questions_placeholder.text(log_message.strip().replace("\r", ""))

            elif "Processing Prompts" in log_message:
                progress_placeholder.text(log_message.strip().replace("\r", ""))

        except queue.Empty:
            pass
        time.sleep(0.1)
    thread.join()

    all_questions_placeholder.empty()
    progress_placeholder.empty()

    try:
        final_output = result_queue.get_nowait()
    except queue.Empty:
        st.error("Could not retrieve result from the asynchronous task.")

    st.success("Finished inferencing!")

    responses = raw_responses(final_output)

    df = create_one_dataframe(responses)

    # Store the dataframe in session state for saving later
    st.session_state.results_dataframe = df
    st.session_state.inference_completed = True

    st.dataframe(df)

# Show save button if inference is completed
if "inference_completed" in st.session_state and st.session_state.inference_completed:
    st.divider()
    st.subheader("💾 Save Results")
    
    # Text input for filename
    if "save_filename" not in st.session_state:
        st.session_state.save_filename = "questionnaire_results.csv"
    
    save_filename = st.text_input(
        "Save File",
        value=st.session_state.save_filename,
        key="save_filename_input",
        help="Enter the filename for the results. Should be a CSV file (e.g., results.csv)."
    )
    
    # Ensure filename ends with .csv
    if save_filename and not save_filename.endswith('.csv'):
        save_filename = save_filename + '.csv'
    
    # Convert dataframe to CSV string for download
    csv = st.session_state.results_dataframe.to_csv(index=False)
    
    st.download_button(
        label="Save Results",
        data=csv,
        file_name=save_filename if save_filename else "questionnaire_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        help="Click to save the results to your computer. You can choose the directory and filename in the save dialog."
    )
