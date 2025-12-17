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

load_dotenv()


class LangChainRAG:
    def __init__(self, index_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.embedding_model = embedding_model

        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Load FAISS index
        index_path = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
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

    def invoke(self, query: str) -> str:
        result = self.qa_chain.invoke({"query": query})
        return result["result"]


def get_rag_chain():
    project_root = os.path.dirname(__file__)
    index_dir = os.path.join(project_root, "faiss_index")
    return LangChainRAG(index_dir=index_dir)

