import streamlit as st
import pickle
import os
import shutil
from pathlib import Path
from datetime import datetime

# Directory to store session cache files
CACHE_DIR = Path(".session_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_session_file_path(session_id: str = "default") -> Path:
    """Get the file path for a session cache."""
    return CACHE_DIR / f"session_{session_id}.pkl"

def generate_session_id() -> str:
    """Generate a unique session ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_current_session_id() -> str:
    """Get the current active session ID, creating one if it doesn't exist."""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    return st.session_state.current_session_id

def save_session_state(session_id: str = None, session_name: str = None) -> bool:
    """
    Save current session state to disk.
    If session_id is None, uses the current active session ID.
    If session_name is None, uses the current session name or generates a default.
    Returns True if successful, False otherwise.
    """
    try:
        # Use provided session_id or get current active one
        if session_id is None:
            session_id = get_current_session_id()
        
        # Use provided session_name or get current one
        if session_name is None:
            session_name = st.session_state.get("current_session_name", f"Session {session_id}")
        
        # Store the session name in session state
        st.session_state.current_session_name = session_name
        
        # Prepare session data
        session_data = _prepare_session_data(session_id, session_name)
        
        # Save using pickle (for complex objects like LLMPrompt)
        cache_file = get_session_file_path(session_id)
        with open(cache_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False

def load_session_state(session_id: str = "default") -> bool:
    """
    Load session state from disk.
    Sets the loaded session as the current active session.
    Returns True if successful, False otherwise.
    """
    try:
        cache_file = get_session_file_path(session_id)
        if not cache_file.exists():
            return False
        
        with open(cache_file, 'rb') as f:
            session_data = pickle.load(f)
        
        # Set this as the current active session
        st.session_state.current_session_id = session_id
        st.session_state.current_session_name = session_data.get("_session_name", f"Session {session_id}")
        
        # Restore session state (skip internal metadata)
        for key, value in session_data.items():
            if not key.startswith("_"):
                st.session_state[key] = value
        
        return True
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return False

def list_available_sessions() -> list[dict]:
    """List all available saved sessions."""
    sessions = []
    for cache_file in CACHE_DIR.glob("session_*.pkl"):
        try:
            with open(cache_file, 'rb') as f:
                session_data = pickle.load(f)
                session_id = session_data.get("_session_id", cache_file.stem.replace("session_", ""))
                session_name = session_data.get("_session_name", f"Session {session_id}")
                timestamp = session_data.get("_timestamp", "Unknown")
                sessions.append({
                    "id": session_id,
                    "name": session_name,
                    "timestamp": timestamp,
                    "file": cache_file
                })
        except Exception:
            continue
    return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

def delete_session(session_id: str = "default") -> bool:
    """Delete a saved session."""
    try:
        cache_file = get_session_file_path(session_id)
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False

def clear_all_sessions():
    """Clear all saved sessions."""
    for cache_file in CACHE_DIR.glob("session_*.pkl"):
        try:
            cache_file.unlink()
        except Exception:
            continue

def rename_session(session_id: str, new_name: str) -> bool:
    """Rename a session by updating its metadata."""
    try:
        cache_file = get_session_file_path(session_id)
        if not cache_file.exists():
            return False
        
        # Load existing session
        with open(cache_file, 'rb') as f:
            session_data = pickle.load(f)
        
        # Update name
        session_data["_session_name"] = new_name
        
        # Save back
        with open(cache_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        # Update current session name if this is the active session
        if st.session_state.get("current_session_id") == session_id:
            st.session_state.current_session_name = new_name
        
        return True
    except Exception as e:
        st.error(f"Error renaming session: {e}")
        return False

def _prepare_session_data(session_id: str, session_name: str) -> dict:
    """Helper function to prepare session data for saving."""
    session_data = {}
    
    # Save dataframes
    if "df_questionnaire" in st.session_state:
        session_data["df_questionnaire"] = st.session_state.df_questionnaire
    if "df_population" in st.session_state:
        session_data["df_population"] = st.session_state.df_population
    
    # Save questionnaires (LLMPrompt objects - need pickle)
    if "questionnaires" in st.session_state:
        session_data["questionnaires"] = st.session_state.questionnaires
    
    # Save inference configs
    if "client_config" in st.session_state:
        session_data["client_config"] = st.session_state.client_config
    if "inference_config" in st.session_state:
        session_data["inference_config"] = st.session_state.inference_config
    
    # Save survey options
    if "survey_options" in st.session_state:
        session_data["survey_options"] = st.session_state.survey_options
    
    # Save other important state
    important_keys = [
        "model_name", "temperature", "max_tokens", "top_p", "seed",
        "api_key", "base_url", "organization", "project",
        "advanced_client_params_str", "advanced_inference_params_str",
        "timeout", "max_retries"
    ]
    for key in important_keys:
        if key in st.session_state:
            session_data[key] = st.session_state[key]
    
    # Save timestamp
    session_data["_timestamp"] = datetime.now().isoformat()
    session_data["_session_id"] = session_id
    session_data["_session_name"] = session_name
    
    return session_data

def save_session_state_to_path(session_id: str = None, session_name: str = None, save_path: Path = None) -> bool:
    """
    Save current session state to a specific path.
    If session_id is None, uses the current active session ID.
    If session_name is None, uses the current session name or generates a default.
    If save_path is None, uses the default cache directory.
    Returns True if successful, False otherwise.
    """
    try:
        # Use provided session_id or get current active one
        if session_id is None:
            session_id = get_current_session_id()
        
        # Use provided session_name or get current one
        if session_name is None:
            session_name = st.session_state.get("current_session_name", f"Session {session_id}")
        
        # Store the session name in session state
        st.session_state.current_session_name = session_name
        
        # Prepare session data
        session_data = _prepare_session_data(session_id, session_name)
        
        # Determine save path
        if save_path is None:
            save_path = get_session_file_path(session_id)
        else:
            # Ensure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # If it's a directory, append the filename
            if save_path.is_dir():
                save_path = save_path / f"session_{session_id}.pkl"
            # Ensure it has .pkl extension
            elif not save_path.suffix == ".pkl":
                save_path = save_path.with_suffix(".pkl")
        
        # Save using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(session_data, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False

def load_session_state_from_path(file_path: Path) -> bool:
    """
    Load session state from a specific file path.
    Sets the loaded session as the current active session.
    Returns True if successful, False otherwise.
    """
    try:
        if not file_path.exists():
            return False
        
        with open(file_path, 'rb') as f:
            session_data = pickle.load(f)
        
        # Get session ID from metadata or generate one
        session_id = session_data.get("_session_id", generate_session_id())
        
        # Set this as the current active session
        st.session_state.current_session_id = session_id
        st.session_state.current_session_name = session_data.get("_session_name", f"Session {session_id}")
        
        # Restore session state (skip internal metadata)
        for key, value in session_data.items():
            if not key.startswith("_"):
                st.session_state[key] = value
        
        # Optionally copy to default cache directory for easy access
        default_path = get_session_file_path(session_id)
        if file_path != default_path:
            try:
                shutil.copy2(file_path, default_path)
            except Exception:
                pass  # If copy fails, that's okay - session is still loaded
        
        return True
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return False

