"""
Configuration settings for the Restaurant Review AI Agent.
Edit these values to change models, paths, or retrieval behaviour.
"""

# Ollama model used for generating answers
LLM_MODEL = "phi3"

# Ollama model used for generating embeddings
EMBEDDING_MODEL = "mxbai-embed-large"

# Path to the customer reviews CSV
CSV_PATH = "data/restaurant_reviews.csv"

# Directory where ChromaDB persists the vector store
DB_LOCATION = "./chroma_langchain_db"

# Number of review chunks returned by the retriever
RETRIEVER_K = 5
