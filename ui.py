import streamlit as st
import validators

from rag_utils import (
    get_llm,
    load_from_url,
    load_from_files,
    split_and_embed,
    generate_summary,
    init_db,
    get_user_sessions,
    create_session,
    get_chat_history,
    save_message,
    get_rag_answer,
)


def init_app_state():
    if "user_id" not in st.session_state:
        from uuid import uuid4

        st.session_state.user_id = str(uuid4())

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = {}

    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""


def render_sidebar():
    with st.sidebar:
        st.subheader("Settings")
        groq_key = st.text_input("Groq API Key", type="password")
        st.session_state.groq_api_key = groq_key

        st.markdown("---")
        st.markdown("### Sessions")

        init_db()
        user_id = st.session_state.user_id
        sessions = get_user_sessions(user_id)

        session_labels = []
        session_ids = []
        for row in sessions:
            label = f"{row['name']} ({row['created_at']})"
            session_labels.append(label)
            session_ids.append(row["session_id"])

        selected_session = None
        if sessions:
            idx = st.selectbox(
                "Select a session",
                options=list(range(len(session_labels))),
                format_func=lambda i: session_labels[i],
            )
            selected_session = session_ids[idx]
        else:
            st.info("No sessions yet. Create one below.")

        new_name = st.text_input("New session name", value="My Session")
        if st.button("Create new session"):
            sid = create_session(user_id, new_name)
            st.session_state.current_session_id = sid
            st.rerun()

        if selected_session and st.session_state.current_session_id is None:
            st.session_state.current_session_id = selected_session

        if selected_session and st.session_state.current_session_id != selected_session:
            st.session_state.current_session_id = selected_session
            st.rerun()


def render_input_section():
    st.subheader("Input")
    url = st.text_input("Enter URL (YouTube, website, etc.)")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, CSV, Excel, JSON)",
        type=["pdf", "txt", "csv", "xlsx", "json"],
        accept_multiple_files=True,
    )
    return url, uploaded_files


def process_and_build_rag(url, uploaded_files):
    if not st.session_state.current_session_id:
        st.error("Please create or select a session in the sidebar.")
        return

    groq_api_key = st.session_state.groq_api_key
    llm = get_llm(groq_api_key)
    if llm is None:
        st.error("Enter a valid Groq API key in the sidebar.")
        return

    all_docs = []

    if url:
        if validators.url(url):
            try:
                with st.spinner("Loading from URL..."):
                    url_docs = load_from_url(url, use_whisper=True)
                    all_docs.extend(url_docs)
                    st.success(f"Loaded {len(url_docs)} document(s) from URL.")
            except Exception as e:
                st.error(f"Error loading from URL: {e}")
        else:
            st.error("Invalid URL.")

    if uploaded_files:
        try:
            with st.spinner("Loading uploaded files..."):
                file_docs = load_from_files(uploaded_files)
                all_docs.extend(file_docs)
                st.success(f"Loaded {len(file_docs)} document(s) from files.")
        except Exception as e:
            st.error(f"Error loading files: {e}")

    if not all_docs:
        st.error("No content found from URL/files.")
        return

    with st.spinner("Building vector store..."):
        vectorstore, splits = split_and_embed(all_docs)

    summary = ""
    with st.spinner("Generating summary with LLM..."):
        summary = generate_summary(llm, splits)

    sid = st.session_state.current_session_id
    st.session_state.vectorstores[sid] = vectorstore
    st.session_state.summaries[sid] = summary

    st.success("RAG store built and summary generated.")


def render_summary_section():
    sid = st.session_state.current_session_id
    if not sid:
        return

    summary = st.session_state.summaries.get(sid)
    if not summary:
        return

    st.subheader("Summary")
    st.write(summary)

    st.download_button(
        "Download Summary as .txt",
        data=summary,
        file_name="summary.txt",
        mime="text/plain",
    )


def render_chat_section():
    sid = st.session_state.current_session_id
    if not sid:
        return

    vectorstore = st.session_state.vectorstores.get(sid)
    if not vectorstore:
        st.info("No RAG index for this session. Process content first.")
        return

    groq_api_key = st.session_state.groq_api_key
    llm = get_llm(groq_api_key)
    if llm is None:
        st.error("Groq API key missing.")
        return

    st.subheader("Chat with your data")

    user_id = st.session_state.user_id
    history_rows = get_chat_history(user_id, sid)

    for row in history_rows:
        with st.chat_message(row["role"]):
            st.markdown(row["content"])

    user_question = st.chat_input("Ask a question about the uploaded content or URL")

    if user_question:
        save_message(user_id, sid, "user", user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_rag_answer(vectorstore, llm, history_rows + [
                    {"role": "user", "content": user_question, "ts": ""}
                ], user_question)
                st.markdown(answer)

        save_message(user_id, sid, "assistant", answer)
        st.rerun()
