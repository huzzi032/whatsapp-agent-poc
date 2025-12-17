"""LangChain-based RAG implementation.

Uses LangChain components: FAISS vector store, HuggingFace embeddings,
Azure OpenAI LLM, and RetrievalQA chain. Keeps the same interface for main.py.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.language_models.fake import FakeListLLM  # type: ignore
from pydantic.v1 import SecretStr
from dotenv import load_dotenv
from pypdf import PdfReader
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


class LangChainRAG:
    def __init__(self, index_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.embedding_model = embedding_model

        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Load FAISS index
        index_path = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_path):
            self._build_index()
        self.vectorstore = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)

        # Set up LLM
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")),
                temperature=0.2,
            )
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI: {e}")
            raise

        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
        )

    def _build_index(self):
        pdf_path = os.path.join(os.path.dirname(__file__), "data", "Jafi_Shoes_Collection.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        build_faiss_index(chunks, self.embedding_model, self.index_dir)
        print("✅ FAISS index created successfully")

    def invoke(self, query: str) -> str:
        result = self.qa_chain.invoke({"query": query})
        return result["result"]


def get_rag_chain():
    project_root = os.path.dirname(__file__)
    index_dir = os.path.join(project_root, "faiss_index")
    return LangChainRAG(index_dir=index_dir)

