from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
# at the top of ingest.py and rag.py

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

# Use the bundled catalog PDF; update the path if you swap the source file.
loader = PyPDFLoader("data/Jafi_Shoes_Collection.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("✅ FAISS index created successfully")
