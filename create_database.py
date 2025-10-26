from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["BASE_URL"]

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"   # 📂 Folder containing your PDF files


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """Load all PDF files from DATA_PATH directory."""
    documents = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.lower().endswith(".pdf"):
            print("----", file_name)
            pdf_path = os.path.join(DATA_PATH, file_name)
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            documents.extend(pdf_docs)
    print(f"Loaded {len(documents)} documents from {DATA_PATH}.")
    return documents


def split_text(documents: list[Document]):
    """Split documents into smaller chunks for vector embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 📏 Increased chunk size for PDF text
        chunk_overlap=200,    # 🔁 Overlap for context continuity
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    print("Example chunk:")
    print(chunks[0].page_content[:300])
    print(chunks[0].metadata)
    return chunks


def save_to_chroma(chunks: list[Document]):
    """Save document chunks to Chroma vector store."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
