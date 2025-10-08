from langchain.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents
