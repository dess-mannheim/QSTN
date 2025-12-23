import streamlit as st
import pandas as pd
from io import StringIO

from qstn.survey_manager import SurveyCreator

from qstn.prompt_builder import LLMPrompt
from gui_elements.session_cache import (
    load_session_state, 
    save_session_state, 
    list_available_sessions,
    delete_session,
    generate_session_id,
    require_hf_login,
    handle_oauth_callback,
    login_button,
    get_user_id,
)

st.set_page_config(layout="wide")
st.title("QSTN")

# Handle OAuth callback first (before checking login)
handle_oauth_callback()

# Check if login is required (for Hugging Face Spaces)
login_required, login_error_message = require_hf_login()
if login_required:
    # Center the login UI in a container
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("### 🔐 Sign In Required")
            st.markdown("This application requires Hugging Face authentication to provide personalized session management.")
            st.markdown("")
            login_button()
    st.stop()  # Stop execution - user must log in

# User is logged in - show username (optional)
user_id = get_user_id()
if user_id:
    st.caption(f"👤 Logged in as: {user_id}")

# Check for saved sessions on startup
if "session_loaded" not in st.session_state:
    available_sessions = list_available_sessions()
    
    if available_sessions:
        st.info("💾 Saved session(s) found!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Continue Last Session", type="primary", use_container_width=True):
                # Load the most recent session
                if load_session_state(available_sessions[0]["id"]):
                    st.session_state.session_loaded = True
                    st.success("Session loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load session.")
        
        with col2:
            # Check if dialog should be open
            if "show_new_session_dialog_start" not in st.session_state:
                st.session_state.show_new_session_dialog_start = False
            
            if st.button("🆕 Start New Session", use_container_width=True):
                # Open the dialog
                st.session_state.show_new_session_dialog_start = True
                st.rerun()
        
        # Show dialog if requested (BEFORE st.stop())
        if st.session_state.get("show_new_session_dialog_start", False):
            with st.container(border=True):
                st.subheader("Create New Session")
                st.write("**Enter session name:**")
                new_session_name = st.text_input(
                    "Session Name",
                    value=f"Session {generate_session_id()}",
                    key="new_session_dialog_name_start_with_sessions"
                )
                
                col_dialog1, col_dialog2 = st.columns(2)
                with col_dialog1:
                    if st.button("Cancel", use_container_width=True, key="cancel_new_session_start_with_sessions"):
                        st.session_state.show_new_session_dialog_start = False
                        st.rerun()
                with col_dialog2:
                    if st.button("Create", type="primary", use_container_width=True, key="create_new_session_start_with_sessions"):
                        if new_session_name and new_session_name.strip():
                            # Generate new session ID
                            new_session_id = generate_session_id()
                            st.session_state.current_session_id = new_session_id
                            st.session_state.current_session_name = new_session_name.strip()
                            st.session_state.session_loaded = True
                            
                            # Save to default location
                            save_session_state(new_session_id, new_session_name.strip())
                            
                            st.session_state.show_new_session_dialog_start = False
                            st.rerun()
                        else:
                            st.error("Please enter a session name.")
            st.stop()  # Stop execution when dialog is open
        
        # Show session details
        with st.expander("📋 View Saved Sessions"):
            for session in available_sessions:
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    session_name = session.get('name', session['id'])
                    st.write(f"**{session_name}**")
                    st.caption(f"ID: {session['id']} | Last saved: {session.get('timestamp', 'Unknown')}")
                with col_b:
                    if st.button("Load", key=f"load_{session['id']}"):
                        if load_session_state(session['id']):
                            st.session_state.session_loaded = True
                            st.rerun()
                with col_c:
                    if st.button("Delete", key=f"delete_{session['id']}"):
                        delete_session(session['id'])
                        st.rerun()
        
        st.stop()  # Stop execution until user chooses
    else:
        # No saved sessions - show create session button
        st.info("No saved sessions found. Create a new session to get started.")
        
        # Initialize dialog state if not present
        if "show_new_session_dialog_start" not in st.session_state:
            st.session_state.show_new_session_dialog_start = False
        
        if st.button("🆕 Create New Session", type="primary", use_container_width=True):
            # Open the dialog
            st.session_state.show_new_session_dialog_start = True
            st.rerun()
        
        # Show dialog if requested
        if st.session_state.get("show_new_session_dialog_start", False):
            with st.container(border=True):
                st.subheader("Create New Session")
                st.write("**Enter session name:**")
                new_session_name = st.text_input(
                    "Session Name",
                    value=f"Session {generate_session_id()}",
                    key="new_session_dialog_name_start_no_sessions"
                )
                
                col_dialog1, col_dialog2 = st.columns(2)
                with col_dialog1:
                    if st.button("Cancel", use_container_width=True, key="cancel_new_session_start_no_sessions"):
                        st.session_state.show_new_session_dialog_start = False
                        st.rerun()
                with col_dialog2:
                    if st.button("Create", type="primary", use_container_width=True, key="create_new_session_start_no_sessions"):
                        if new_session_name and new_session_name.strip():
                            # Generate new session ID
                            new_session_id = generate_session_id()
                            st.session_state.current_session_id = new_session_id
                            st.session_state.current_session_name = new_session_name.strip()
                            st.session_state.session_loaded = True
                            
                            # Save to default location
                            save_session_state(new_session_id, new_session_name.strip())
                            
                            st.session_state.show_new_session_dialog_start = False
                            st.rerun()
                        else:
                            st.error("Please enter a session name.")
        
        st.stop()  # Stop execution until user creates a session

# Define example dataframes once (used for both demo and defaults)
example_questionnaire = pd.DataFrame({
    'questionnaire_item_id': [1, 2, 3],
    'question_content': ['Coffee', 'Pizza', 'Ice cream'],
    'question_stem': [
        'How do you feel about?',
        'How do you feel about?',
        'How do you feel about?'
    ]
})

example_population = pd.DataFrame({
    'questionnaire_name': ['Student', 'Teacher'],
    'system_prompt': ['You are a student.', 'You are a teacher.'],
    'questionnaire_instruction': [
        'Please answer the following questions.',
        'Please answer the following questions.'
    ]
})


current_session_id = st.session_state.get("current_session_id")
current_session_name = st.session_state.get("current_session_name", "Unnamed Session")

# Initialize dialog state if not present
if "show_new_session_dialog_start" not in st.session_state:
    st.session_state.show_new_session_dialog_start = False

# Dialog for naming new session (accessible from anywhere)
if st.session_state.get("show_new_session_dialog_start", False):
    with st.container(border=True):
        st.subheader("Create New Session")
        st.write("**Enter session name:**")
        new_session_name = st.text_input(
            "Session Name",
            value=f"Session {generate_session_id()}",
            key="new_session_dialog_name_start"
        )
        
        col_dialog1, col_dialog2 = st.columns(2)
        with col_dialog1:
            if st.button("Cancel", use_container_width=True, key="cancel_new_session_start"):
                st.session_state.show_new_session_dialog_start = False
                st.rerun()
        with col_dialog2:
            if st.button("Create", type="primary", use_container_width=True, key="create_new_session_start"):
                if new_session_name and new_session_name.strip():
                    # Generate new session ID
                    new_session_id = generate_session_id()
                    st.session_state.current_session_id = new_session_id
                    st.session_state.current_session_name = new_session_name.strip()
                    st.session_state.session_loaded = True
                    
                    # Save to default location
                    save_session_state(new_session_id, new_session_name.strip())
                    
                    st.session_state.show_new_session_dialog_start = False
                    st.rerun()
                else:
                    st.error("Please enter a session name.")
    
    # Stop execution here if dialog is open - don't show main content
    st.stop()

# Switch session section (only show if session is loaded)
if st.session_state.get("session_loaded", False):
    available_sessions = list_available_sessions()
    if available_sessions:
        # Header with button for creating new session
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.subheader("Switch to another session:")
        with col_header2:
            if st.button("➕ Create New", key="create_new_from_switcher", use_container_width=True):
                # Open the dialog for creating a new session
                st.session_state.show_new_session_dialog_start = True
                st.rerun()
        
        # Reorder sessions: current session first, then rest sorted by name for consistency
        current_session_in_list = None
        other_sessions = []
        
        for session in available_sessions:
            if session['id'] == current_session_id:
                current_session_in_list = session
            else:
                other_sessions.append(session)
        
        # Sort other sessions by name (alphabetically) for consistent order
        other_sessions.sort(key=lambda x: x['name'].lower())
        
        # Build final list: current session first, then others
        if current_session_in_list:
            ordered_sessions = [current_session_in_list] + other_sessions
        else:
            ordered_sessions = other_sessions
        
        # Create a selectbox with session names only
        session_names = [s['name'] for s in ordered_sessions]
        # Create a mapping from name to ID (handle potential duplicates by using first match)
        name_to_id = {s['name']: s['id'] for s in ordered_sessions}
        
        # Track the currently loaded session to avoid unnecessary reloads
        if "last_loaded_session_id" not in st.session_state:
            st.session_state.last_loaded_session_id = current_session_id
        
        # Current session should always be at index 0
        default_index = 0
        
        def on_session_change():
            """Callback when session selection changes."""
            selected_name = st.session_state.session_switcher_select
            selected_id = name_to_id[selected_name]
            
            # Only switch if it's a different session
            if selected_id != st.session_state.get("last_loaded_session_id"):
                # Save current session before switching
                if current_session_id:
                    save_session_state()
                
                # Load new session
                if load_session_state(selected_id):
                    st.session_state.last_loaded_session_id = selected_id
                    st.session_state.session_switch_trigger = True
        
        st.selectbox(
            "Select Session",
            options=session_names,
            index=default_index,
            key="session_switcher_select",
            label_visibility="collapsed",
            on_change=on_session_change
        )
        
        # Check if we need to rerun after session switch
        if st.session_state.get("session_switch_trigger", False):
            st.session_state.session_switch_trigger = False
            st.rerun()
        
        st.divider()
    else:
        # No sessions available - show button to create first session
        st.subheader("Create a new session:")
        if st.button("➕ Create New Session", key="create_new_no_sessions", use_container_width=True, type="primary"):
            # Open the dialog for creating a new session
            st.session_state.show_new_session_dialog_start = True
            st.rerun()
        st.divider()

# Only show main content after a session is loaded or created
if st.session_state.get("session_loaded", False):
    # Demo section showing expected CSV format
    with st.expander("📋 View Example CSV Format", expanded=False):
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            st.subheader("Questionnaire CSV Format")
            st.write("**Required columns:** `questionnaire_item_id`, `question_content`, `question_stem`")
            st.write(example_questionnaire)
        
        with demo_col2:
            st.subheader("Population CSV Format")
            st.write("**Required columns:** `questionnaire_name`, `system_prompt`, `questionnaire_instruction`")
            st.write(example_population)

    st.divider()

    col1, col2 = st.columns(2)

    # Initialize session state for dataframes if not present
    if "df_questionnaire" not in st.session_state:
        st.session_state.df_questionnaire = None
    if "df_population" not in st.session_state:
        st.session_state.df_population = None

    df_population = None
    df_questionnaire = None
    using_defaults = False

    with col1:
        uploaded_questionnaire = st.file_uploader("Select a questionnaire to start with")
        if uploaded_questionnaire is not None:
            # New file uploaded - read and store it
            df_questionnaire = pd.read_csv(uploaded_questionnaire)
            st.session_state.df_questionnaire = df_questionnaire
            st.write(df_questionnaire)
        elif st.session_state.df_questionnaire is not None:
            # Use previously uploaded file from session state
            df_questionnaire = st.session_state.df_questionnaire
            st.write(df_questionnaire)
            if st.button("Clear", key="clear_questionnaire", help="Reset to example questionnaire"):
                st.session_state.df_questionnaire = None
                st.rerun()
        else:
            # No file uploaded and no previous file - use example
            df_questionnaire = example_questionnaire
            using_defaults = True
            st.info("ℹ️ Using example questionnaire. Upload a file to use your own data.")
            st.write(df_questionnaire)

    with col2:
        uploaded_population = st.file_uploader("Select a population to start with")
        if uploaded_population is not None:
            # New file uploaded - read and store it
            df_population = pd.read_csv(uploaded_population)
            st.session_state.df_population = df_population
            st.write(df_population)
        elif st.session_state.df_population is not None:
            # Use previously uploaded file from session state
            df_population = st.session_state.df_population
            st.write(df_population)
            if st.button("Clear", key="clear_population", help="Reset to example population"):
                st.session_state.df_population = None
                st.rerun()
        else:
            # No file uploaded and no previous file - use example
            df_population = example_population
            using_defaults = True
            st.info("ℹ️ Using example population. Upload a file to use your own data.")
            st.write(df_population)

    # Button is always enabled now (using defaults if no files uploaded)
    disabled = False
            
    st.divider()

    if st.button("Confirm and Prepare Questionnaire", type="primary", disabled=disabled, use_container_width=True):
        questionnaires: list[LLMPrompt] = SurveyCreator.from_dataframe(df_population, df_questionnaire)
        st.session_state.questionnaires = questionnaires
        
        # Auto-save session
        save_session_state()
        
        st.switch_page("pages/01_Option_Prompt.py")
