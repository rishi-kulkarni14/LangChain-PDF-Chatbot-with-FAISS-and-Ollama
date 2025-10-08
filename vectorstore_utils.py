from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

PERSIST_DIR = "vectorstore"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vectorstore(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectorstore.save_local(PERSIST_DIR)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    if os.path.exists(os.path.join(PERSIST_DIR, "index.faiss")):
        return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"Vectorstore not found at {PERSIST_DIR}. Please create it first.")
