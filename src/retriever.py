from langchain_chroma import Chroma


def create_retriever(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    return retriever

