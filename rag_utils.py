import os
import json
import sqlite3
from uuid import uuid4
from tempfile import NamedTemporaryFile

import pandas as pd
from faster_whisper import WhisperModel
import yt_dlp

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
    PyPDFLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DB_PATH = "chat_history.db"
parser = StrOutputParser()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def create_session(user_id: str, name: str | None = None) -> str:
    if not name:
        name = "New Session"
    session_id = str(uuid4())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (user_id, session_id, name) VALUES (?, ?, ?)",
        (user_id, session_id, name),
    )
    conn.commit()
    conn.close()
    return session_id


def get_user_sessions(user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id, name, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def save_message(user_id: str, session_id: str, role: str, content: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
        (user_id, session_id, role, content),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: str, session_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content, ts
        FROM chat_messages
        WHERE user_id=? AND session_id=?
        ORDER BY id ASC
        """,
        (user_id, session_id),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def format_chat_history_for_llm(history_rows):
    if not history_rows:
        return "No previous conversation."
    lines = []
    for row in history_rows:
        role = row["role"]
        content = row["content"]
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def get_llm(groq_api_key: str | None):
    if not groq_api_key:
        # Try environment secret (HuggingFace)
        groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return None

    return ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0)



def transcribe_audio(video_url: str) -> str:
    audio_pattern = "temp_audio.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_pattern,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    audio_file = None
    for f in os.listdir():
        if f.startswith("temp_audio."):
            audio_file = f
            break

    if audio_file is None:
        raise RuntimeError("Audio download failed.")

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    transcript = ""
    segments, _ = model.transcribe(audio_file, beam_size=1, vad_filter=True)
    for segment in segments:
        transcript += segment.text + " "

    if os.path.exists(audio_file):
        os.remove(audio_file)

    return transcript.strip()


def load_from_url(input_url: str, use_whisper: bool = True):
    docs = []

    if "youtube.com" in input_url or "youtu.be" in input_url:
        clean_url = input_url.split("&")[0]
        try:
            loader = YoutubeLoader.from_youtube_url(clean_url, add_video_info=False)
            youtube_docs = loader.load()
            if youtube_docs:
                docs.extend(youtube_docs)
            else:
                raise ValueError("Empty transcript.")
        except Exception:
            if not use_whisper:
                raise
            transcript = transcribe_audio(clean_url)
            if transcript:
                docs.append(
                    Document(
                        page_content=transcript,
                        metadata={"source": clean_url, "type": "youtube_audio"},
                    )
                )
    else:
        loader = UnstructuredURLLoader(
            urls=[input_url],
            ssl_verify=False,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        web_docs = loader.load()
        docs.extend(web_docs)

    return docs


def load_from_files(files):
    docs = []

    for file in files:
        name = file.name.lower()

        if name.endswith(".pdf"):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)

            os.remove(tmp_path)

        elif name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_string(index=False)
            docs.append(
                Document(page_content=text, metadata={"source": file.name, "type": "csv"})
            )

        elif name.endswith(".xlsx"):
            df = pd.read_excel(file)
            text = df.to_string(index=False)
            docs.append(
                Document(page_content=text, metadata={"source": file.name, "type": "excel"})
            )

        elif name.endswith(".json"):
            data = json.load(file)
            text = json.dumps(data, indent=2)
            docs.append(
                Document(page_content=text, metadata={"source": file.name, "type": "json"})
            )

    return docs


def split_and_embed(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore, splits


def build_summary_chain(llm):
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
You are a helpful assistant.

Summarize the following content in 200–300 words.
Then provide exactly 10 concise bullet-point key takeaways.

Content:
{text}
""",
    )
    chain = summary_prompt | llm | parser
    return chain


def generate_summary(llm, splits, max_chars: int = 15000) -> str:
    joined_text = "\n".join(d.page_content for d in splits)[:max_chars]
    summary_chain = build_summary_chain(llm)
    summary = summary_chain.invoke({"text": joined_text})
    return summary


def format_docs_for_context(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_rag_answer(vectorstore, llm, history_rows, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer only using the provided context. "
                "If the answer is not in the context, say you don't know.",
            ),
            (
                "human",
                "Chat history:\n{chat_history}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}",
            ),
        ]
    )

    chain = (
        {
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(
                lambda _: format_chat_history_for_llm(history_rows)
            ),
            "context": retriever | RunnableLambda(format_docs_for_context),
        }
        | chat_prompt
        | llm
        | parser
    )

    answer = chain.invoke(question)
    return answer
