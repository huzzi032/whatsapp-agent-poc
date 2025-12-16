import os
import json
import pickle
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

load_dotenv()


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= length:
            break
        start = end - overlap
    return chunks


def build_faiss_index(texts, model_name: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    # ensure float32
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # ensure embeddings are contiguous float32 array and pass as keyword to satisfy type checkers
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    # FAISS SWIG wrapper expects a positional ndarray argument; calling with
    # keywords can raise a TypeError at runtime. Use the positional form.
    # (Pylance may warn about missing named params; ignore that for runtime.)
    index.add(embeddings)  # type: ignore[arg-type]
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    # store texts mapping
    with open(os.path.join(index_dir, "index.pkl"), "wb") as f:
        pickle.dump(texts, f)


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
