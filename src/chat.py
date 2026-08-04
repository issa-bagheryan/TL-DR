from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from query_rewriter import rewrite_query


def start_chat(retriever):

    
    chat_history = []


    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0.2
    )


    prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant answering questions about a document.

    Use ONLY the context below.
    If the answer is not in the context, say:
    "I don't have enough information in the document."

    Context:
    {context}

    Conversation history:
    {history}

    Question:
    {question}

    Answer:
    """
    )


    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "history": lambda x: chat_history
        }
        | prompt
        | llm
    )



    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        search_query = rewrite_query(
        question,
        chat_history
        )

        response = chain.invoke(search_query)

        print("\nAI:", response.content)

        chat_history.append(
        HumanMessage(content=question)
        )

        chat_history.append(
        AIMessage(content=response.content)
        )