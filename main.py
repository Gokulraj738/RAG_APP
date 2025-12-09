import streamlit as st
from ui import (
    init_app_state,
    render_sidebar,
    render_input_section,
    process_and_build_rag,
    render_summary_section,
    render_chat_section,
)


st.set_page_config(page_title="Universal RAG Summarizer + Chatbot", page_icon="📚")
st.title("Universal RAG Summarizer + Q&A Chatbot")

init_app_state()
render_sidebar()

st.markdown("---")
url, uploaded_files = render_input_section()

if st.button("Process & Summarize"):
    process_and_build_rag(url, uploaded_files)

render_summary_section()
render_chat_section()
