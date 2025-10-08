from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vectorstore(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="gemma3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa


import sys
import types

# Workaround for Streamlit watcher breaking on torch.classes
class DummyModule(types.ModuleType):
    __path__ = []

sys.modules['torch.classes'] = DummyModule('torch.classes')
