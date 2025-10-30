import streamlit as st

def paginator(items, session_state_key):
    """
    Creates a reusable paginator component for a list of items.

    Args:
        items (list): The list of items to paginate through.
        session_state_key (str): A unique key for storing the current index in 
                                 st.session_state.
    
    Returns:
        Any: The currently selected item from the list.
    """
    list_length = len(items)
    
    # Initialize the session state for the current index if it doesn't exist.
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = 0

    def next_item():
        """Increments the index, wrapping around to the start if it reaches the end."""
        if list_length > 0:
            st.session_state[session_state_key] = (st.session_state[session_state_key] + 1) % list_length

    def prev_item():
        """Decrements the index, wrapping around to the end if it goes below zero."""
        if list_length > 0:
            st.session_state[session_state_key] = (st.session_state[session_state_key] - 1 + list_length) % list_length

    # Create the navigation columns.
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.button("⬅️ Previous", on_click=prev_item, use_container_width=True, disabled=(list_length <= 1))

    # Display the current position and the popover for jumping to a specific item.
    with col2:
        if list_length > 0:
            current_num = st.session_state[session_state_key] + 1
            
            popover = st.popover(f"Item {current_num} of {list_length}", use_container_width=True)

            with popover:
                st.markdown("Jump to a specific item:")
                target_index_input = st.number_input(
                    "Item Number",
                    min_value=1,
                    max_value=list_length,
                    value=current_num,
                    step=1,
                    label_visibility="collapsed"
                )
                if st.button("Go"):
                    st.session_state[session_state_key] = target_index_input - 1
                    st.rerun()
        else:
            st.text("No items to display")

    # "Next" button in the third column.
    with col3:
        st.button("Next ➡️", on_click=next_item, use_container_width=True, disabled=(list_length <= 1))

    # Return the currently selected item.
    if list_length > 0:
        return st.session_state[session_state_key]
    else:
        return None