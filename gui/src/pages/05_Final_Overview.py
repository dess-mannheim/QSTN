import streamlit as st
from gui_elements.paginator import paginator
from gui_elements.stateful_widget import StatefulWidgets
from gui_elements.output_manager import st_capture, TqdmToStreamlit

import io
import queue
import time
import threading
import asyncio

import logging

from contextlib import redirect_stderr, redirect_stdout

from surveygen.parser.llm_answer_parser import raw_responses
from surveygen.utilities.constants import InterviewType
from surveygen.utilities.utils import create_one_dataframe
from surveygen.survey_manager import (
    conduct_survey_in_context,
    conduct_whole_survey_one_prompt,
    conduct_survey_question_by_question,
)

from streamlit.runtime.scriptrunner import add_script_run_ctx

from openai import AsyncOpenAI

# Set OpenAI's API key and API base to use vLLM's API server.


if "interviews" not in st.session_state:
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

current_index = paginator(st.session_state.interviews, "overview_page")

interview = st.session_state.interviews[current_index]

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

    with st.container(border=True):
        current_system_prompt, current_prompt = interview.get_prompt_for_interview_type(InterviewType.CONTEXT)
        current_system_prompt = current_system_prompt.replace("\n", "  \n")
        current_prompt = current_prompt.replace("\n", "  \n")
        st.write(current_system_prompt)
        st.write(current_prompt)

model_name = state.create(
    st.text_input,
    "save_file",
    "Save File",
    # initial_value="meta-llama/Llama-3.1-70B-Instruct",
    # placeholder="meta-llama/Llama-3.1-70B-Instruct",
    help="The save file to write your results to. Should be a csv file.",
)


if st.button("Confirm and Run Survey", type="primary", use_container_width=True):
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
        result_q, client, interviews, model_name, **inference_config
    ):
        queue_writer = QueueWriter(log_queue)

        # We need to redirect the output to a queue, as streamlit does not support multithreading
        # API concurrency should be  configurable in the GUI
        try:
            with redirect_stderr(queue_writer):
                result = conduct_survey_question_by_question(
                    client,
                    interviews=interviews,
                    client_model_name=model_name,
                    api_concurrency=100,
                    **inference_config,
                )

        except Exception as e:
            result = e
        finally:
            result_q.put(result)

    while not log_queue.empty():
        log_queue.get()
    while not result_queue.empty():
        result_queue.get()

    thread = threading.Thread(
        target=run_async_in_thread,
        args=(result_queue, client, st.session_state.interviews, model_name),
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
            # This is quite a hacky solution for now, we should adjust surveygen to make the messages clearly parsable.
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

    st.success("Finished inferencing! Saving results...")

    responses = raw_responses(final_output)

    df = create_one_dataframe(responses)

    st.dataframe(df)

    df.to_csv(st.session_state.save_file, index=False)

    st.success(f"File saved to {st.session_state.save_file}!")
