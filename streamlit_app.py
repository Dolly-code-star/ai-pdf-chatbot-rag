import os
import shutil
import streamlit as st

from utils.ingest import ingest_pdfs
from utils.embeddings import generate_embeddings
from utils.vector_store import search_documents
from chatbot import ask_llm

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI PDF Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)

    if os.path.exists("db"):
        shutil.rmtree("db")

    for uploaded_file in uploaded_files:
        file_path = os.path.join(
            "data",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    ingest_pdfs("data")
    st.success("✅ PDFs uploaded and processed successfully!")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input(
    "Ask a question about your PDFs..."
)

if question:
    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # Retrieve context
    query_embedding = generate_embeddings([question])[0]
    results = search_documents(query_embedding)

    if results["documents"] and results["documents"][0]:
        context = results["documents"][0][0]
        source = results["metadatas"][0][0]["source"]

        answer = ask_llm(context, question)

        bot_response = (
            f"{answer}\n\n📄 Source: {source}"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": bot_response
            }
        )

        with st.chat_message("assistant"):
            st.write(bot_response)
    else:
        with st.chat_message("assistant"):
            st.write(
                "No relevant documents found. Please upload PDFs first."
            )