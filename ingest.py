import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)


def build_faiss_index(texts, model_name: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_dir)
    print(f"✅ FAISS index saved to {index_dir}")


def main():
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    pdf_path = os.path.join(os.path.dirname(__file__), "data", "Jafi_Shoes_Collection.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    build_faiss_index(chunks, model_name, os.path.join(os.path.dirname(__file__), "faiss_index"))
    print("✅ FAISS index created successfully")


if __name__ == "__main__":
    main()
