📚 Universal RAG Summarizer + AI Chatbot

A powerful Streamlit-based application that allows users to:

Extract and summarize content from:

📄 PDF, TXT, CSV, JSON, Excel files

🔗 Website URLs

🎥 YouTube or social media videos (with transcript or Whisper fallback)

Build a RAG (Retrieval-Augmented Generation) index

Chat interactively with the extracted content

Save conversations by user session

Download generated summaries

This project supports multi-session chat history and can act as a personal knowledge assistant.

| Feature                                         | Supported |
| ----------------------------------------------- | :-------: |
| YouTube transcript extraction                   |     ✅     |
| Whisper speech-to-text if no transcript         |     ✅     |
| Website content extraction                      |     ✅     |
| Multi-file support (PDF, JSON, TXT, CSV, Excel) |     ✅     |
| RAG-powered Q&A chatbot                         |     ✅     |
| Saved history + session management              |     ✅     |
| Downloadable summaries                          |     ✅     |
| LCEL (LangChain Expression Language) support    |     ✅     |
| Multi-user session memory                       |     ✅     |


| Component    | Library               |
| ------------ | --------------------- |
| UI           | Streamlit             |
| LLM          | Groq (Llama Models)   |
| RAG          | LangChain + FAISS     |
| Embeddings   | Sentence Transformers |
| Audio → Text | Faster Whisper        |
| Storage      | SQLite Local DB       |


How it works

User Uploads Files or Enters URL
              ↓
Extract text (crawler / transcript / whisper)
              ↓
Chunk + embed content using vector DB
              ↓
Generate structured summary using Groq LLM
              ↓
Start Q&A chatbot with memory + RAG search
              ↓
Save messages and allow session switching

