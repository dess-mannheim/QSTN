import streamlit as st
import pickle
import os
import shutil
import secrets
import urllib.parse
import requests
import re
from pathlib import Path
from datetime import datetime

# OAuth helper functions
def get_space_url() -> str:
    """Get the base URL for the Space."""
    space_base_url = os.getenv("SPACE_BASE_URL")
    if space_base_url:
        return space_base_url.rstrip("/")
    # Default to hf.space URL
    space_host = os.getenv("SPACE_HOST", "")
    if space_host:
        return f"https://{space_host}".rstrip("/")
    return ""

def oidc_config():
    """Get OpenID Connect configuration."""
    provider = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co").rstrip("/")
    try:
        return requests.get(f"{provider}/.well-known/openid-configuration", timeout=10).json()
    except Exception as e:
        st.error(f"Failed to fetch OIDC config: {e}")
        return {}

def login_button():
    """Display a login button that redirects to HF OAuth."""
    if not os.environ.get("OAUTH_CLIENT_ID"):
        st.error("OAuth is not configured. Please enable hf_oauth: true in README.md")
        return
    
    try:
        cfg = oidc_config()
        if not cfg:
            st.error("Could not fetch OpenID configuration. Please try again later.")
            return
            
        auth_endpoint = cfg.get("authorization_endpoint")
        if not auth_endpoint:
            st.error("Could not get authorization endpoint from OpenID configuration.")
            return

        state = secrets.token_urlsafe(24)
        st.session_state["oauth_state"] = state

        redirect_uri = get_space_url() + "/"
        if not redirect_uri:
            st.warning("Could not determine Space URL. OAuth may not work correctly.")
            redirect_uri = "https://" + os.getenv("SPACE_HOST", "") + "/"
        
        params = {
            "client_id": os.environ["OAUTH_CLIENT_ID"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": os.getenv("OAUTH_SCOPES", "openid profile"),
            "state": state,
        }
        url = auth_endpoint + "?" + urllib.parse.urlencode(params)
        # Make button larger and more prominent
        st.link_button("🔐 Sign in with Hugging Face", url, use_container_width=True, type="primary")
    except Exception as e:
        st.error(f"Error setting up OAuth login: {e}")
        st.info("Please check the Space logs for more details.")

def handle_oauth_callback():
    """Handle OAuth callback and exchange code for token."""
    qp = st.query_params
    if "code" not in qp:
        return

    # CSRF protection
    returned_state = qp.get("state")
    expected_state = st.session_state.get("oauth_state")
    if expected_state and returned_state != expected_state:
        st.error("OAuth state mismatch. Please try signing in again.")
        return

    cfg = oidc_config()
    token_endpoint = cfg.get("token_endpoint")
    userinfo_endpoint = cfg.get("userinfo_endpoint")

    if not token_endpoint:
        st.error("Could not get token endpoint")
        return

    redirect_uri = get_space_url() + "/"

    data = {
        "grant_type": "authorization_code",
        "code": qp["code"],
        "redirect_uri": redirect_uri,
        "client_id": os.environ["OAUTH_CLIENT_ID"],
        "client_secret": os.environ["OAUTH_CLIENT_SECRET"],
    }
    
    try:
        token_response = requests.post(token_endpoint, data=data, timeout=10)
        token_response.raise_for_status()
        token = token_response.json()
        
        if "access_token" not in token:
            st.error(f"Token exchange failed: {token}")
            return

        # Fetch user profile
        profile = {}
        if userinfo_endpoint:
            profile_response = requests.get(
                userinfo_endpoint,
                headers={"Authorization": f"Bearer {token['access_token']}"},
                timeout=10,
            )
            profile_response.raise_for_status()
            profile = profile_response.json()
        elif "id_token" in token:
            # Decode ID token if userinfo not available
            import base64
            import json
            try:
                # JWT is base64url encoded (3 parts separated by .)
                id_token_parts = token["id_token"].split(".")
                if len(id_token_parts) >= 2:
                    # Decode payload (second part)
                    payload = id_token_parts[1]
                    # Add padding if needed
                    payload += "=" * (4 - len(payload) % 4)
                    decoded = base64.urlsafe_b64decode(payload)
                    profile = json.loads(decoded)
            except Exception:
                pass

        st.session_state["hf_profile"] = profile
        st.session_state["hf_access_token"] = token["access_token"]

        # Clean the URL (remove code/state)
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"OAuth callback failed: {e}")

def get_user_id() -> str | None:
    """Get the current user's ID from OAuth profile."""
    profile = st.session_state.get("hf_profile") or {}
    # Common OpenID fields: preferred_username, name, sub
    return profile.get("preferred_username") or profile.get("name") or profile.get("sub")

def safe_user_id(uid: str) -> str:
    """Make user ID filesystem-safe."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", uid)[:80]

def is_huggingface_space() -> bool:
    """
    Detect if running on Hugging Face Spaces.
    Checks for HF-specific environment variables.
    """
    # Hugging Face Spaces set these environment variables
    return bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"))

def is_user_logged_in() -> bool:
    """
    Check if user is logged in via OAuth.
    """
    return get_user_id() is not None

def require_hf_login() -> tuple[bool, str]:
    """
    Check if login is required and if user is logged in.
    Returns (is_required, error_message)
    - If on Hugging Face: login is required
    - If not on Hugging Face: login is not required
    """
    if is_huggingface_space():
        if not is_user_logged_in():
            return True, (
                "🔒 **Authentication Required**\n\n"
                "This application requires you to be logged in to Hugging Face to use it. "
                "Please log in using the button in the top right corner of this page."
            )
    return False, ""

def _get_local_user_identifier_file() -> Path:
    """Get the path to the file storing the local user identifier."""
    return Path(".session_cache") / ".local_user_id"

def get_user_identifier() -> str:
    """
    Get a unique identifier for the current user.
    - On Hugging Face Spaces with OAuth: uses OAuth username
    - Locally: uses a persistent file-based identifier that survives refreshes
    """
    # Check for OAuth user first
    user_id = get_user_id()
    if user_id:
        return safe_user_id(user_id)
    
    # For local development: use a persistent file-based identifier
    # This persists across page refreshes and app restarts
    identifier_file = _get_local_user_identifier_file()
    
    # Try to load from file first
    if identifier_file.exists():
        try:
            with open(identifier_file, 'r') as f:
                stored_id = f.read().strip()
            if stored_id:
                st.session_state.user_identifier = stored_id
                return stored_id
        except Exception:
            pass
    
    # File doesn't exist or couldn't be read - create new identifier
    # Generate a stable identifier for this local session
    new_id = f"local_{secrets.token_hex(8)}"
    
    # Save to file for persistence
    try:
        identifier_file.parent.mkdir(parents=True, exist_ok=True)
        with open(identifier_file, 'w') as f:
            f.write(new_id)
    except Exception:
        pass
    
    st.session_state.user_identifier = new_id
    return new_id

def get_user_cache_dir() -> Path:
    """
    Get a user-specific cache directory.
    Uses /data if persistent storage is enabled, otherwise .session_cache
    """
    # Use persistent storage if available, otherwise use .session_cache
    base_dir = Path("/data") if Path("/data").exists() else Path(".session_cache")
    user_cache_base = base_dir / "user_cache"
    
    # Get user ID
    user_id = get_user_id()
    if user_id:
        safe_uid = safe_user_id(user_id)
        user_cache_dir = user_cache_base / safe_uid
        user_cache_dir.mkdir(parents=True, exist_ok=True)
        return user_cache_dir
    
    # Locally: use shared cache directory
    cache_dir = base_dir / "session_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

# Directory to store session cache files (now user-specific)
def get_cache_dir() -> Path:
    """Get the cache directory for the current user."""
    return get_user_cache_dir()

def get_session_file_path(session_id: str = "default") -> Path:
    """Get the file path for a session cache (user-specific)."""
    return get_cache_dir() / f"session_{session_id}.pkl"

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
    """List all available saved sessions for the current user."""
    sessions = []
    cache_dir = get_cache_dir()  # Now user-specific
    for cache_file in cache_dir.glob("session_*.pkl"):
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
    """Delete a saved session (user-specific)."""
    try:
        cache_file = get_session_file_path(session_id)  # Already user-specific
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False

def clear_all_sessions():
    """Clear all saved sessions for the current user."""
    cache_dir = get_cache_dir()
    for cache_file in cache_dir.glob("session_*.pkl"):
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

