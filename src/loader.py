from selector import select_pdf
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(len(documents))