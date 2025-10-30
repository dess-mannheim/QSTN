import streamlit as st
import json
from gui_elements.stateful_widget import StatefulWidgets

# --- Page Configuration ---
st.set_page_config(
    page_title="Inference Settings",
    layout="wide"
)

st.title("AsyncOpenAI API Client & Inference Configurator")
st.markdown("Use the widgets below to configure the `AsyncOpenAI` client and the inference parameters for an API call. Advanced or less common options can be added as a JSON object.")

st.divider()

# --- Column Layout ---
col1, col2 = st.columns(2)

defaults = {
    # Client Config
    "api_key": "", "organization": "", "project": "", "base_url": "",
    "timeout": 20, "max_retries": 2,
    "advanced_client_params_str": '',
    # Inference Config
    "model_name": "", "temperature": 1.0, "max_tokens": 1024,
    "top_p": 1.0, "seed": 42,
    "advanced_inference_params_str": ''
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


state = StatefulWidgets()

# ==============================================================================
# COLUMN 1: OPENAI CLIENT CONFIGURATION
# ==============================================================================
with col1:
    st.header("1. Client Configuration")

with col2:
    st.header("2. Inference Configuration")

with col1:
    with st.container(border=True):
        st.subheader("Core Settings")

        api_key = state.create(
            st.text_input,
            "api_key",
            "API Key",
            initial_value="",
            type="password",
            placeholder="sk-...",
            help="Your OpenAI API key. It is handled securely by Streamlit."
        )

        organization = state.create(
            st.text_input,
            "organization",
            "Organization ID",
            initial_value="",
            placeholder="org-...",
            help="Optional identifier for your organization."
        )

        project = state.create(
            st.text_input,
            "project",
            "Project ID",
            initial_value="",
            placeholder="proj_...",
            help="Optional identifier for your project."
        )

        base_url = state.create(
            st.text_input,
            "base_url",
            "Base URL",
            initial_value="",
            placeholder="https://api.openai.com/v1",
            help="The base URL for the API. Leave empty for the default."
        )

        timeout = state.create(
            st.number_input,
            "timeout",
            "Timeout (seconds)",
            initial_value=20,
            min_value=1,
            help="The timeout for API requests in seconds."
        )

        max_retries = state.create(
            st.number_input,
            "max_retries",
            "Max Retries",
            initial_value=2,
            min_value=0,
            help="The maximum number of times to retry a failed request."
        )

        with st.expander("Advanced Client Settings (JSON)"):
            advanced_client_params_str = state.create(
                st.text_area,
                "advanced_client_params_str",
                "JSON for other client parameters",
                initial_value="",
                placeholder='{\n  "default_headers": {"X-Custom-Header": "value"}\n}',
                height=150,
                help='Enter any other client init parameters like "default_headers" or "default_query" as a valid JSON object.'
            )

# ==============================================================================
# COLUMN 2: INFERENCE PARAMETERS
# ==============================================================================

with col2:
    with st.container(border=True):
        st.subheader("Core Settings")
        model_name = state.create(
            st.text_input,
            "model_name",
            "Model Name",
            #initial_value="meta-llama/Llama-3.1-70B-Instruct",
            placeholder="meta-llama/Llama-3.1-70B-Instruct",
            help="The model to use for the inference call."
        )

        temperature = state.create(
            st.slider,
            "temperature",
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            initial_value=1.0,
            help="Controls randomness. Lower values are more deterministic and less creative."
        )

        max_tokens = state.create(
            st.number_input,
            "max_tokens",
            "Max Tokens",
            initial_value=1024,
            min_value=1,
            help="The maximum number of tokens to generate in the completion."
        )

        top_p = state.create(
            st.slider,
            "top_p",
            "Top P",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            initial_value=1.0,
            help="Controls nucleus sampling. The model considers tokens with top_p probability mass."
        )

        seed = state.create(
            st.number_input,
            "seed",
            "Seed",
            initial_value=42,
            min_value=0,
            help="A specific seed for reproducibility of results."
        )

        with st.expander("Advanced Inference Settings (JSON)"):
            advanced_inference_params_str = state.create(
                st.text_area,
                "advanced_inference_params_str",
                "JSON for other inference parameters",
                initial_value="",
                placeholder='{\n  "stop": ["\\n", " Human:"],\n  "presence_penalty": 0\n}',
                height=150,
                help='Enter any other valid inference parameters like "stop", "logit_bias", or "frequency_penalty" as a JSON object.'
            )


# ==============================================================================
# GENERATION AND DISPLAY LOGIC
# ==============================================================================
st.divider()

if st.button("Generate Configuration & Code", type="primary", use_container_width=True):
    # --- Process Client Config ---
    client_config = {
        "api_key": api_key
    }
    # Add optional string parameters if they are not empty
    if organization: client_config["organization"] = organization
    if project: client_config["project"] = project
    if base_url: client_config["base_url"] = base_url
    # Add numeric parameters
    client_config["timeout"] = timeout
    client_config["max_retries"] = max_retries

    try:
        if advanced_client_params_str:
            advanced_client_params = json.loads(advanced_client_params_str)
            client_config.update(advanced_client_params)
    except json.JSONDecodeError:
        st.error("Invalid JSON detected in Advanced Client Settings. Please correct it.")
        st.stop()

    # --- Process Inference Config ---
    inference_config = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed
    }

    try:
        if advanced_inference_params_str:
            advanced_inference_params = json.loads(advanced_inference_params_str)
            inference_config.update(advanced_inference_params)
    except json.JSONDecodeError:
        st.error("Invalid JSON detected in Advanced Inference Settings. Please correct it.")
        st.stop()

    st.session_state.client_config = client_config
    st.session_state.inference_config = inference_config

    st.success("Configuration generated successfully!")