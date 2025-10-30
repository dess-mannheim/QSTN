import streamlit as st

class StatefulWidgets:
    """
    A class to create Streamlit widgets with encapsulated session state management.
    """
    def __init__(self, prefix: str = "_"):
        """
        Initializes the StatefulWidgets class.

        Args:
            prefix (str): A prefix to use for the widget's internal session state key.
        """
        self.prefix = prefix

    def _store_value(self, key: str) -> None:
        """
        Callback function to store the widget's value from the internal state
        to the main session state.
        """
        st.session_state[key] = st.session_state[f"{self.prefix}{key}"]

    def _initialize_and_load(self, key: str, initial_value) -> None:
        """
        Initializes the session state for a given key if it doesn't exist,
        and then loads this value into the widget's internal state.
        """
        if key not in st.session_state:
            st.session_state[key] = initial_value
        st.session_state[f"{self.prefix}{key}"] = st.session_state[key]

    def create(self, widget_func, key: str, *args, initial_value=None, **kwargs):
        """
        Creates a Streamlit widget and binds its state.

        Args:
            widget_func: The Streamlit widget function (e.g., st.text_input).
            key (str): The key for the main session state.
            *args: Positional arguments to pass to the widget function.
            initial_value: The initial value to set in the session state if not present.
            **kwargs: Keyword arguments to pass to the widget function.

        Returns:
            The created Streamlit widget.
        """
        self._initialize_and_load(key, initial_value)
        return widget_func(
            *args,
            key=f"{self.prefix}{key}",
            on_change=self._store_value,
            args=(key,),
            **kwargs,
        )