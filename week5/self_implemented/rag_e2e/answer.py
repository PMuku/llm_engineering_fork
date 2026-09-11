"""
1. fetch_context(question)
2. answer_question(question, history)
"""

from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document


DB_NAME = str(Path(__file__).parent.parent / "vector_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma4"

RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vector_store.as_retriever()
llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url="http://localhost:11434")


def fetch_context(question: str) -> list[Document]:
    """
    Return relevant chunks to user's query
    """
    return retriever.invoke(input=question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer queries and provide citations in the form tuple[response, cited chunks]
    """
    comb = combined_question(question=question, history=history)
    docs = fetch_context(question=comb)
    context = "\n\n".join(doc.page_content for doc in docs)
    sys_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=sys_prompt)]
    messages.extend(convert_to_messages(messages=history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(input=messages)
    return response.content, docs
