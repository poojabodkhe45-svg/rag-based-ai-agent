"""
Restaurant Review AI Agent
----------------------------
A RAG-based conversational agent that answers questions about a pizza restaurant
using real customer reviews and a locally-running LLM via Ollama.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever
from config import LLM_MODEL, RETRIEVER_K

# ── Model & Prompt ─────────────────────────────────────────────────────────────

model = OllamaLLM(model=LLM_MODEL)

template = """
You are a helpful assistant that answers questions about a pizza restaurant
based on real customer reviews.

Relevant reviews:
{reviews}

Customer question: {question}

Answer clearly and concisely based only on the reviews provided. If the reviews
don't contain enough information to answer, say so honestly.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# ── CLI Loop ───────────────────────────────────────────────────────────────────

def main():
    retriever = get_retriever(k=RETRIEVER_K)

    print("\n🍕  Restaurant Review AI Agent")
    print("   Powered by Ollama +", LLM_MODEL)
    print("   Type 'q' or 'quit' to exit\n")

    while True:
        print("-" * 50)
        question = input("Ask a question about the restaurant: ").strip()

        if question.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        print()
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})
        print("Answer:", result)
        print()


if __name__ == "__main__":
    main()
