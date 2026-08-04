from selector import select_pdf
from loader import load_pdf
from splitter import split_documents
from vectordb import create_vector_db
from retriever import create_retriever
from chat import start_chat
from embedder import embeddings


path = r"C:\Users\Issa\Desktop\pdf rag\data\chroma"


pdf = select_pdf()

documents = load_pdf(pdf)

chunks = split_documents(documents)


vector_db = create_vector_db(
    chunks,
    embeddings,
    path
)


retriever = create_retriever(vector_db)


start_chat(retriever)