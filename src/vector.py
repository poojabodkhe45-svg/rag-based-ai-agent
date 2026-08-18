"""
Vector store setup using ChromaDB + Ollama embeddings.
Builds the store on first run; reuses it on subsequent runs.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from config import EMBEDDING_MODEL, DB_LOCATION, CSV_PATH


def _build_documents_from_csv(csv_path: str) -> tuple[list[Document], list[str]]:
    """Load reviews CSV and convert rows to LangChain Documents."""
    df = pd.read_csv(csv_path)
    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i),
        )
        ids.append(str(i))
        documents.append(doc)

    return documents, ids


def get_retriever(k: int = 5):
    """
    Returns a LangChain retriever backed by ChromaDB.
    Populates the vector store from the CSV if it doesn't exist yet.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    add_documents = not os.path.exists(DB_LOCATION)

    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )

    if add_documents:
        print(f"📂  Building vector store from '{CSV_PATH}'...")
        documents, ids = _build_documents_from_csv(CSV_PATH)
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"✅  Indexed {len(documents)} reviews.\n")

    return vector_store.as_retriever(search_kwargs={"k": k})
