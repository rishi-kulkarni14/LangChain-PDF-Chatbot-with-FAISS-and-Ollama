import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from vectorstore_utils import create_vectorstore, load_vectorstore
from memory_utils import save_chat_history, load_chat_history

import sys
import types
class DummyModule(types.ModuleType):
    __path__ = []

sys.modules['torch.classes'] = DummyModule('torch.classes')

# --- Constants ---
VECTORSTORE_PATH = "vectorstore"

st.set_page_config(page_title="PDF Chat with LangChain & Ollama", page_icon="📄")

st.title("📄 PDF Chatbot with Memory and OllamaLLM")

# Upload PDF file
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if "chat_history" not in st.session_state:
    # Load existing chat history or initialize empty list
    st.session_state.chat_history = load_chat_history()

if pdf_file is not None:
    # Save uploaded PDF temporarily
    pdf_path = f"temp_uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Load or create vectorstore
    try:
        vectorstore = load_vectorstore()
        st.success("Loaded existing vectorstore.")
    except FileNotFoundError:
        with st.spinner("Creating vectorstore from PDF..."):
            vectorstore = create_vectorstore(pdf_path)
        st.success("Vectorstore created and saved!")

    # Create QA chain
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="gemma3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # User input form
    query = st.text_input("Ask a question about your PDF:")

    if query:
        # Run the query through the chain
        result = qa({"query": query})

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Append new message to chat history
        st.session_state.chat_history.append({"user": query, "bot": answer})

        # Save chat history after each interaction
        save_chat_history(st.session_state.chat_history)

        # Display chat
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")

        # Optionally show sources
        if sources:
            st.markdown("**Sources:**")
            for i, doc in enumerate(sources):
                st.markdown(f"- Source {i+1}: {doc.metadata.get('source', 'unknown')}")

else:
    st.info("Please upload a PDF to start chatting.")


# Optional: button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    save_chat_history(st.session_state.chat_history)
    st.experimental_rerun()
