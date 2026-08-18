# RAG-Based AI Agent

A Retrieval-Augmented Generation (RAG) chatbot that answers natural language questions about a pizza restaurant using real customer reviews, powered by [Ollama](https://ollama.com).

---

## How It Works

```
User Question
     │
     ▼
ChromaDB Vector Store  ──►  Top-K relevant reviews
     │
     ▼
Ollama LLM (phi3)  ──►  Answer grounded in real reviews
```

1. **Ingestion** - Customer reviews from `data/restaurant_reviews.csv` are embedded using `mxbai-embed-large` and stored in a local ChromaDB vector store (built once, reused every run).
2. **Retrieval** - For each question, the 5 most semantically relevant reviews are fetched.
3. **Generation** - A local LLM (`phi3`) reads those reviews and answers the question.

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running

Pull the required models:

```bash
ollama pull phi3
ollama pull mxbai-embed-large
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/restaurant-review-agent.git
cd restaurant-review-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

On the **first run**, the vector store is built automatically from the CSV. Subsequent runs reuse the cached store and start instantly.

---

## Example

```
   Restaurant Review AI Agent
   Powered by Ollama + phi3
   Type 'q' or 'quit' to exit

--------------------------------------------------
Ask a question about the restaurant: Is the pizza worth the price?

Answer: Based on the reviews, opinions are mixed. Several customers found the
pizza exceptional and worth every penny, praising the quality of ingredients
and wood-fired crust. However, a few reviewers felt the pricing was high
for the portion size. Overall, the majority lean positive on value.
```

---

## Project Structure

```
restaurant-review-agent/
├── scripts/
│   ├── main.py                       # CLI entry point & conversation loop
│   └── vector.py                     # Vector store setup and retriever
│   └── config.py                     # Centralized configuration
├── data/
│   └── restaurant_reviews.csv  # Source reviews dataset
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuration

All settings live in `config.py`:

| Setting          | Default              | Description                          |
|------------------|----------------------|--------------------------------------|
| `LLM_MODEL`      | `phi3`               | Ollama model for answer generation   |
| `EMBEDDING_MODEL`| `mxbai-embed-large`  | Ollama model for embeddings          |
| `CSV_PATH`       | `data/restaurant_reviews.csv` | Path to reviews dataset    |
| `DB_LOCATION`    | `./chroma_langchain_db` | ChromaDB persistence directory    |
| `RETRIEVER_K`    | `5`                  | Number of reviews retrieved per query|

You can swap in any Ollama-compatible model - for example, replace `phi3` with `llama3` for more capable responses.

---

## Tech Stack

| Component       | Library / Tool          |
|-----------------|-------------------------|
| LLM             | Ollama (`phi3`)          |
| Embeddings      | Ollama (`mxbai-embed-large`) |
| Vector Store    | ChromaDB                 |
| Orchestration   | LangChain                |
| Data            | Pandas                   |

---

## License

MIT
