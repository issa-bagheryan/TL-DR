from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


rewrite_prompt = ChatPromptTemplate.from_template(
"""
Rewrite the user's question into a standalone search query.

Use the conversation history to understand references like:
"he", "it", "that", "this".

Return only the rewritten question.

Conversation history:
{history}

User question:
{question}
"""
)


def rewrite_query(question, history):

    chain = rewrite_prompt | llm

    response = chain.invoke(
        {
            "history": history,
            "question": question
        }
    )

    return response.content