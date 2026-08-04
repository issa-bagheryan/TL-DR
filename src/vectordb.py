from langchain_chroma import Chroma

path = r"C:\Users\Issa\Desktop\pdf rag\data\chroma"

def create_vector_db(chunks, embeddings, path):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=path)

    return vector_db