import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Disable tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text to ask.")
    args = parser.parse_args()
    query_text = args.query_text

    # --- Load vector DB ---
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # --- Search similar chunks ---
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"No good matches found. Got {len(results)} results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\nQuerying NVIDIA model...\n")

    # --- NVIDIA Chat Client ---
    client = ChatNVIDIA(
        model="deepseek-ai/deepseek-v3.1-terminus",
        api_key=os.getenv("OPENAI_API_KEY"),  # use same .env key
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )

    # --- Stream reasoning & content ---
    print("Model reasoning and response:\n")
    for chunk in client.stream([{"role": "user", "content": prompt}]):
        if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
            print(chunk.additional_kwargs["reasoning_content"], end="")
        if chunk.content:
            print(chunk.content, end="")
    print("\n")

    # --- Display sources ---
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
    print(f"\n📚 Sources: {sources}")


if __name__ == "__main__":
    main()
