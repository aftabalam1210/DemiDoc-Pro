import streamlit as st
from state_manager import initialize_session_state
from ui_components import render_header, render_chat, render_footer
from processor import get_enhanced_processor
from ui_components import inject_custom_css
from ui_components import render_document_section



# Configure Streamlit page
st.set_page_config(
    page_title="DemiDoc Pro | Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_custom_css()


# Initialize state and processor
initialize_session_state()
processor = get_enhanced_processor()

# --- UI Layout ---
render_header()
render_document_section(processor)
render_chat(processor)
render_footer()
