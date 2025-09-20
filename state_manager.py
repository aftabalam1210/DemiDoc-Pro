import streamlit as st

def initialize_session_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "document_uploaded": False,
        "document_metadata": {},
        "processing_history": [],
        "current_document_hash": None,
        "sidebar_visible": False,
        "debug_mode": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
