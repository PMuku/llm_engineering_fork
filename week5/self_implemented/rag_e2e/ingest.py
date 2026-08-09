"""
1. Read knowledge base
2. Turn docs into chunks
3. Turn chunks into vectors
4. Store in chroma
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


KNOWLEDGE_BASE_DIR = Path("../../knowledge-base")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_NAME = "vector_db"


def _read_knowledge_base() -> list[Document]:
    folders = KNOWLEDGE_BASE_DIR.glob("*")
    docs: list[Document] = []
    for folder in folders:
        doc_type = folder.stem
        loader = DirectoryLoader(
            path=folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            docs.append(doc)
    return docs

def _chunk_docs(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents=docs)
    return chunks

def _embed_chunks(chunks: list[Document]) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)

    collection = vector_store._collection
    count = collection.count()
    sample_embeddings = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dims = len(sample_embeddings)
    print(f"there are {count:,} vectors with {dims:,} dimensions in the vector store")

    return vector_store

def main():
    docs = _read_knowledge_base()
    chunks = _chunk_docs(docs=docs)
    _embed_chunks(chunks=chunks)
    print("Ingestion complete")

if __name__ == "__main__":
    main()