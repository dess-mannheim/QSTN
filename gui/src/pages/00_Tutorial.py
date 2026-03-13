"""
Tutorial & Help page for the QSTN GUI.
Use ?section=start|options|prompts|inference|overview to deep-link to a section.
"""

import streamlit as st

st.set_page_config(page_title="Tutorial & Help", layout="wide")
st.title("📖 QSTN Tutorial & Help")
st.markdown(
    "Full documentation of every option in the GUI. Use the sections below "
    "or jump to a step from the links."
)

# Section keys for deep linking
SECTION_START = "start"
SECTION_OPTIONS = "options"
SECTION_PROMPTS = "prompts"
SECTION_INFERENCE = "inference"
SECTION_OVERVIEW = "overview"

section_param = st.query_params.get("section", SECTION_START)

# Quick jump links
st.markdown("**Jump to:** ")
cols = st.columns(5)
with cols[0]:
    st.page_link(
        "pages/00_Tutorial.py",
        label="Start",
        query_params={"section": SECTION_START},
    )
with cols[1]:
    st.page_link(
        "pages/00_Tutorial.py",
        label="Answer options",
        query_params={"section": SECTION_OPTIONS},
    )
with cols[2]:
    st.page_link(
        "pages/00_Tutorial.py",
        label="Prompts",
        query_params={"section": SECTION_PROMPTS},
    )
with cols[3]:
    st.page_link(
        "pages/00_Tutorial.py",
        label="Inference",
        query_params={"section": SECTION_INFERENCE},
    )
with cols[4]:
    st.page_link(
        "pages/00_Tutorial.py",
        label="Run & results",
        query_params={"section": SECTION_OVERVIEW},
    )

st.divider()

# =============================================================================
# 1. START PAGE
# =============================================================================
with st.expander(
    "**1. Start page – Sessions, questionnaire & population**",
    expanded=(section_param == SECTION_START),
):
    st.markdown(
        "On the **Start** page you create or load a session, upload your data, "
        "and build the survey. All work is saved per session."
    )
    st.markdown(
        "**Session options** (no replica): Continue Last Session, Start New "
        "Session, Switch to another session (dropdown), Create New, View Saved "
        "Sessions (Load/Delete)."
    )
    st.subheader("Data upload")
    with st.container(border=True):
        _c1, _c2 = st.columns(2)
        with _c1:
            st.text_input(
                "Select a questionnaire to start with",
                value="",
                disabled=True,
                key="tutorial_start_quest",
            )
            st.caption(
                "File upload for the **questionnaire** CSV. One row per "
                "question. Required columns: `questionnaire_item_id`, "
                "`question_content`, `question_stem` (optional)."
            )
        with _c2:
            st.text_input(
                "Select a population to start with",
                value="",
                disabled=True,
                key="tutorial_start_pop",
            )
            st.caption(
                "File upload for the **population** CSV. One row per persona. "
                "Required columns: `questionnaire_name`, `system_prompt`, "
                "`questionnaire_instruction`."
            )
        st.button(
            "Confirm and Prepare Questionnaire",
            disabled=True,
            key="tutorial_start_btn",
        )
        st.caption(
            "Builds the survey and goes to **Answer options**. Use **Clear** "
            "under each upload to reset to example data."
        )
    st.subheader("Next step")
    st.page_link("Start_Page.py", label="→ Go to Start page")

# =============================================================================
# 2. ANSWER OPTIONS (LIKERT SCALE OPTIONS GENERATOR)
# =============================================================================
with st.expander(
    "**2. Answer options – Likert scales & response format**",
    expanded=(section_param == SECTION_OPTIONS),
):
    st.markdown(
        "This page configures how the LLM sees answer options and how the "
        "model outputs its answer (free text, JSON, or constrained choice)."
    )
    with st.container(border=True):
        st.subheader("Main Configuration")
        _col1, _col2, _col3 = st.columns(3)
        with _col1:
            st.number_input(
                "Number of Options (n)",
                value=5,
                min_value=2,
                step=1,
                disabled=True,
                key="tutorial_opt_n",
            )
            st.caption(
                "Total number of choices in the scale (e.g. 5 for a 5-point " "scale). Minimum 2."
            )
        with _col2:
            st.selectbox(
                "Index Type",
                options=["integer", "char_low", "char_up"],
                index=0,
                disabled=True,
                key="tutorial_opt_idx",
            )
            st.caption(
                "How options are labeled: integer (1,2,3…), char_low (a,b,c…), "
                "or char_up (A,B,C…)."
            )
        with _col3:
            st.number_input(
                "Starting Index",
                value=1,
                step=1,
                disabled=True,
                key="tutorial_opt_start",
            )
            st.caption("Number to start counting from (e.g. 1).")
        st.subheader("Ordering and Structure")
        _co1, _co2, _co3, _co4 = st.columns(4)
        with _co1:
            st.checkbox(
                "From-To Scale Only",
                value=False,
                disabled=True,
                key="tutorial_opt_ft",
            )
            st.caption(
                "Only first and last labels shown (e.g. “1 Strongly Disagree "
                "to 5 Strongly Agree”). Requires exactly 2 answer texts."
            )
        with _co2:
            st.checkbox(
                "Random Order",
                value=False,
                disabled=True,
                key="tutorial_opt_rand",
            )
            st.caption("Randomize option order per prompt. Cannot combine with " "Reversed Order.")
        with _co3:
            st.checkbox(
                "Reversed Order",
                value=False,
                disabled=True,
                key="tutorial_opt_rev",
            )
            st.caption("Reverse option order (e.g. 5→1).")
        with _co4:
            st.checkbox(
                "Even Order",
                value=False,
                disabled=True,
                key="tutorial_opt_even",
            )
            st.caption("If odd number of labels, middle option is removed.")
        st.subheader("Answer Texts")
        st.text_area(
            "Enter Answer Texts (one per line)",
            value="Strongly Disagree\nDisagree\nNeutral\nAgree\nStrongly Agree",
            height=120,
            disabled=True,
            key="tutorial_opt_texts",
        )
        st.caption(
            "One label per line. Number of lines must match **Number of "
            "Options** (or 2 if From-To Scale Only)."
        )
        st.subheader("Response Generation Method")
        st.checkbox(
            "Output Indices Only",
            value=False,
            disabled=True,
            key="tutorial_opt_outidx",
        )
        st.caption(
            "If checked, model outputs only indices (e.g. 1, 2) instead of "
            "full text (e.g. 1: Strongly Disagree). Applies to all methods "
            "below."
        )
        st.selectbox(
            "Response Generation Method",
            options=[
                "None",
                "JSON Single Answer",
                "JSON All Options (Probabilities)",
                "JSON with Reasoning",
                "Choice",
            ],
            index=0,
            disabled=True,
            key="tutorial_opt_rgm",
        )
        st.markdown("**Details by option:**")
        st.markdown("""
        - **None** — Model returns free text; no structured output.
          You must parse the response yourself. No extra configuration on this page.
        - **JSON Single Answer** — Model returns a single selected answer in JSON
          (e.g. one chosen option per question). Parsing is automatic; use when
          you want one answer per item.
        - **JSON All Options (Probabilities)** — Model returns a probability
          (or score) for each option. Use when you want a distribution over
          choices rather than a single pick. Parsing expects this structure.
        - **JSON with Reasoning** — Model returns reasoning text plus the chosen
          answer in JSON. Use when you need both explanation and a structured
          answer. Parsing extracts both.
        - **Choice** — Model must pick exactly one value from a fixed list.
          When selected, **Allowed Choices** appears: one choice per line,
          either indices only (e.g. 1, 2, 3) or with labels
          (e.g. 1: Strongly Disagree). **Output Indices Only** controls whether
          the model returns indices or full labels. Parsing uses the same list.
        """)
    st.subheader("Next step")
    st.page_link("pages/01_Option_Prompt.py", label="→ Go to Answer options")

# =============================================================================
# 3. PROMPT CONFIGURATION
# =============================================================================
with st.expander(
    "**3. Prompt configuration – System prompt & instructions**",
    expanded=(section_param == SECTION_PROMPTS),
):
    st.markdown(
        "Edit the system and user prompts the LLM sees. Use the checkboxes to "
        "copy the current prompts to all questionnaires on confirm."
    )
    with st.container(border=True):
        st.subheader("Configuration")
        st.checkbox(
            "On update: change all System Prompts",
            value=True,
            disabled=True,
            key="tutorial_prompt_chg_sys",
        )
        st.caption(
            "When you click Confirm and Prepare Questionnaire, the current "
            "system prompt is applied to **all** questionnaires (missing "
            "placeholders added to others)."
        )
        st.checkbox(
            "On update: change all questionnaire instructions",
            value=True,
            disabled=True,
            key="tutorial_prompt_chg_inst",
        )
        st.caption("Same for the main prompt (instructions); all questionnaires get " "this text.")
        st.text_area(
            "System prompt",
            value="You are a student.",
            height=80,
            disabled=True,
            key="tutorial_prompt_sys",
        )
        st.caption("The system message sent to the model (e.g. role or context).")
        st.text_area(
            "Prompt",
            value="Please answer the following questions.",
            height=80,
            disabled=True,
            key="tutorial_prompt_user",
        )
        st.caption("User-facing instructions given before the questions.")
        st.write("**Insert Placeholder:**")
        _pb1, _pb2, _pb3, _pb4 = st.columns(4)
        with _pb1:
            st.button("Prompt Questions", disabled=True, key="tutorial_prompt_btn_p")
        with _pb2:
            st.button("Prompt Options", disabled=True, key="tutorial_prompt_btn_o")
        with _pb3:
            st.button("Automatic Output", disabled=True, key="tutorial_prompt_btn_a")
        with _pb4:
            st.button("JSON Template", disabled=True, key="tutorial_prompt_btn_j")
        st.caption(
            "Type shortcut (-P, -O, -A, -J) in system or main prompt, then "
            "click to replace with placeholder. **Prompt Questions**: where "
            "question text goes. **Prompt Options**: where answer options go "
            "(configure options first). **Automatic Output**: JSON/output "
            "instructions. **JSON Template**: JSON schema for response."
        )
        st.text_area(
            "Question Stem",
            value="How do you feel about?",
            height=80,
            disabled=True,
            key="tutorial_prompt_stem",
        )
        st.caption(
            "Template for each question. Use **Question Content** placeholder "
            "(-Q) so the actual item (e.g. “Coffee”) is inserted."
        )
        st.checkbox(
            "Randomize the order of items",
            value=False,
            disabled=True,
            key="tutorial_prompt_rand",
        )
        st.caption("If checked, question order is randomized per questionnaire.")
    st.caption(
        "**Live Preview** (on the real page) shows prompts with placeholders "
        "filled; use **Update Preview** to refresh. The **paginator** switches "
        "which questionnaire you edit."
    )
    st.subheader("Next step")
    st.page_link(
        "pages/02_Prompt_Configuration.py",
        label="→ Go to Prompt configuration",
    )

# =============================================================================
# 4. INFERENCE SETTINGS
# =============================================================================
with st.expander(
    "**4. Inference settings – API client & model**",
    expanded=(section_param == SECTION_INFERENCE),
):
    st.markdown(
        "Configure where the LLM is called (API URL, key) and how (model, "
        "temperature, etc.). Values are used on Final overview when you run "
        "the survey."
    )
    _inf_col1, _inf_col2 = st.columns(2)
    with _inf_col1:
        with st.container(border=True):
            st.subheader("1. Client Configuration")
            st.text_input(
                "API Key",
                value="",
                type="password",
                disabled=True,
                key="tutorial_inf_apikey",
            )
            st.caption("Your OpenAI API key (or leave empty for local vLLM). Handled " "securely.")
            st.text_input(
                "Base URL",
                value="",
                placeholder="https://api.openai.com/v1",
                disabled=True,
                key="tutorial_inf_base",
            )
            st.caption(
                "API base URL. For **local vLLM** use `http://localhost:8000/v1`. "
                "Optional: Organization ID, Project ID."
            )
            st.number_input(
                "Timeout (seconds)",
                value=20,
                min_value=1,
                disabled=True,
                key="tutorial_inf_timeout",
            )
            st.caption("Request timeout in seconds.")
            st.number_input(
                "Max Retries",
                value=2,
                min_value=0,
                disabled=True,
                key="tutorial_inf_retries",
            )
            st.caption(
                "Number of retries for failed requests. Advanced: JSON for "
                "default_headers, default_query, etc."
            )
    with _inf_col2:
        with st.container(border=True):
            st.subheader("2. Inference Configuration")
            st.text_input(
                "Model Name",
                value="",
                placeholder="meta-llama/Llama-3.1-70B-Instruct",
                disabled=True,
                key="tutorial_inf_model",
            )
            st.caption(
                "Model ID. Must match your API/vLLM (e.g. "
                "`meta-llama/Llama-3.2-3B-Instruct` for vLLM)."
            )
            st.slider(
                "Temperature",
                0.0,
                2.0,
                1.0,
                0.01,
                disabled=True,
                key="tutorial_inf_temp",
            )
            st.caption("Randomness (0–2). Lower = more deterministic.")
            st.number_input(
                "Max Tokens",
                value=1024,
                min_value=1,
                disabled=True,
                key="tutorial_inf_maxtok",
            )
            st.caption("Maximum tokens to generate per completion.")
            st.slider(
                "Top P",
                0.0,
                1.0,
                1.0,
                0.01,
                disabled=True,
                key="tutorial_inf_topp",
            )
            st.caption(
                "Nucleus sampling: consider tokens with cumulative probability " "up to this value."
            )
            st.number_input(
                "Seed",
                value=42,
                min_value=0,
                disabled=True,
                key="tutorial_inf_seed",
            )
            st.caption(
                "Seed for reproducible sampling. Advanced: JSON for stop, "
                "presence_penalty, frequency_penalty, logit_bias."
            )
    st.subheader("Using local vLLM")
    st.markdown("""
    1. Start the vLLM server, e.g.:
       ```bash
       python -m vllm.entrypoints.openai.api_server \
         --model meta-llama/Llama-3.2-3B-Instruct \
         --max-model-len 8192
       ```
    2. In the GUI: set **Base URL** to `http://localhost:8000/v1`
       and **Model name** to the same model ID
       (e.g. `meta-llama/Llama-3.2-3B-Instruct`).
    3. Click **Generate Configuration & Code** to save;
       then go to **Final overview** to run the survey.
    """)
    st.subheader("Next step")
    st.markdown(
        "Click **Generate Configuration & Code** to save client and inference "
        "config and go to **Final overview**."
    )
    st.page_link("pages/03_Inference_Setting.py", label="→ Go to Inference settings")

# =============================================================================
# 5. FINAL OVERVIEW & RUN
# =============================================================================
with st.expander(
    "**5. Final overview – Run survey & save results**",
    expanded=(section_param == SECTION_OVERVIEW),
):
    st.markdown(
        "Read-only inference summary, prompt preview for the selected method, "
        "then run the survey and save results."
    )
    with st.container(border=True):
        st.subheader("⚙️ Inference Parameters")
        st.text_input(
            "Model Name",
            value="meta-llama/Llama-3.2-3B-Instruct",
            disabled=True,
            key="tutorial_overview_model",
        )
        st.caption(
            "Shown from Inference settings (not editable here). Change on "
            "**Inference settings** page."
        )
        st.slider(
            "Temperature",
            0.0,
            2.0,
            1.0,
            0.01,
            disabled=True,
            key="tutorial_overview_temp",
        )
        st.number_input(
            "Max Tokens",
            value=1024,
            min_value=1,
            disabled=True,
            key="tutorial_overview_maxtok",
        )
        st.selectbox(
            "Questionnaire Method",
            options=["Single item", "Battery", "Sequential"],
            index=0,
            disabled=True,
            key="tutorial_overview_method",
        )
        st.caption(
            "**Single item**: one question per API call. **Battery**: all "
            "questions in one call. **Sequential**: all in one thread with "
            "conversation history. Affects prompts and result parsing."
        )
        st.markdown("**Details by option:**")
        st.markdown("""
        - **Single item** — One question is sent per API call; the model sees
          only that question (and system/instruction context). Preview on the
          page shows the first few items so you can check how each is prompted.
        - **Battery** — All questions are sent in one API call; the model sees
          the full list and responds once. Preview shows one combined prompt.
        - **Sequential** — All questions are sent in one conversation thread
          with history; the model sees prior Q&A. Preview shows one combined
          flow. Parsing respects the conversation structure.
        """)
    st.caption(
        "**Live preview** (on the real page): for Single item, first few items "
        "shown; for Battery/Sequential, one combined preview. **Paginator** "
        "switches questionnaire."
    )
    with st.container(border=True):
        st.subheader("💾 Save Results")
        st.text_input(
            "Save File",
            value="questionnaire_results.csv",
            disabled=True,
            key="tutorial_overview_savefile",
        )
        st.caption("Filename for the CSV. If you omit `.csv`, it is appended.")
        st.button("Save Results", disabled=True, key="tutorial_overview_savebtn")
        st.caption(
            "Downloads the results table as CSV. Appears after you click "
            "**Confirm and Run Questionnaire** and inference finishes."
        )
    st.page_link("pages/04_Final_Overview.py", label="→ Go to Final overview")

st.divider()
st.subheader("Workflow summary")
st.markdown(
    "**Start** (upload data, confirm) → **Answer options** (scale + response "
    "method, confirm) → **Prompt configuration** (prompts + placeholders, "
    "confirm) → **Inference settings** (API + model, generate config) → "
    "**Final overview** (run & save)."
)
