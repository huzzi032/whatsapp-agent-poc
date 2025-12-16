"""Lightweight RAG implementation without LangChain.

Uses SentenceTransformers for embeddings, FAISS for vector search, and
Azure OpenAI REST API for chat completions. Exposes `get_rag_chain()`
that returns an object with `invoke(query)` -> str to keep the existing
interface used by `main.py` and tests.
"""

import os
import pickle
import json
from typing import List
from dotenv import load_dotenv
import requests

from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()


class SimpleRAG:
    def __init__(self, index_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.emb_model_name = embedding_model
        self.embedder = SentenceTransformer(self.emb_model_name)

        # load faiss index
        index_path = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)

        # load metadata / documents
        pkl_path = os.path.join(self.index_dir, "index.pkl")
        self.docs = []
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
            # try to extract a list of texts from common LangChain/FAISS structures
            self.docs = self._extract_texts(obj)

    def _extract_texts(self, obj) -> List[str]:
        texts: List[str] = []
        # list of Document-like objects
        if isinstance(obj, list):
            for e in obj:
                if hasattr(e, "page_content"):
                    texts.append(e.page_content)
                elif isinstance(e, dict) and "page_content" in e:
                    texts.append(e["page_content"])
                else:
                    texts.append(str(e))
            return texts

        if isinstance(obj, dict):
            # LangChain FAISS saved format often contains 'docstore' and 'index_to_docstore_id'
            if "docstore" in obj and "index_to_docstore_id" in obj:
                docstore = obj["docstore"]
                idx_map = obj["index_to_docstore_id"]
                # idx_map is typically a list mapping index -> key
                for key in idx_map:
                    doc = docstore.get(key) if hasattr(docstore, "get") else docstore[key]
                    if hasattr(doc, "page_content"):
                        texts.append(doc.page_content)
                    elif isinstance(doc, dict) and "page_content" in doc:
                        texts.append(doc["page_content"])
                    else:
                        texts.append(str(doc))
                return texts

            # other dicts that may contain documents
            for k, v in obj.items():
                if isinstance(v, list):
                    for e in v:
                        if hasattr(e, "page_content"):
                            texts.append(e.page_content)
                        elif isinstance(e, dict) and "page_content" in e:
                            texts.append(e["page_content"])
                        else:
                            texts.append(str(e))
                    if texts:
                        return texts

        # fallback: represent object
        return [str(obj)]

    def _embed(self, texts: List[str]):
        return self.embedder.encode(texts, convert_to_numpy=True)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_emb = self._embed([query])
        D, I = self.index.search(q_emb.astype("float32"), k)
        results = []
        for idx in I[0]:
            if idx < 0:
                continue
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results

    def generate_answer(self, query: str, context_texts: List[str]) -> str:
        # Build prompt
        context = "\n\n---\n\n".join(context_texts)
        prompt = f"Use the following context to answer the question. If the answer is not contained, be honest.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Call Azure OpenAI REST endpoint
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        if not api_key or not endpoint:
            raise EnvironmentError("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment")

        url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
        resp.raise_for_status()
        j = resp.json()
        # Extract text
        try:
            return j["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(j)

    def invoke(self, query: str) -> str:
        docs = self.retrieve(query, k=3)
        return self.generate_answer(query, docs)


def get_rag_chain():
    project_root = os.path.dirname(__file__)
    index_dir = os.path.join(project_root, "faiss_index")
    return SimpleRAG(index_dir=index_dir)

