from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import SecretStr
from dotenv import load_dotenv
import os

# Support both legacy and current LangChain layouts for RetrievalQA
try:
    from langchain_classic.chains import RetrievalQA  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - fallback for newer layouts
    from langchain.chains.retrieval_qa.base import RetrievalQA  # type: ignore[reportMissingImports]

load_dotenv()
# at the top of rag.py
from dotenv import load_dotenv
load_dotenv()
def get_rag_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    vectorstore = FAISS.load_local(
        os.path.join(os.path.dirname(__file__), "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True
    )

    api_key_value = os.getenv("AZURE_OPENAI_API_KEY")
    llm = AzureChatOpenAI(
        api_key=SecretStr(api_key_value) if api_key_value else None,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version="2025-01-01-preview",
        temperature=0.2
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    return qa
