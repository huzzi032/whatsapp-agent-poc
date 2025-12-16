import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rag_chain():
    # require faiss index
    if not (PROJECT_ROOT / "faiss_index").exists():
        pytest.skip("faiss_index not found; run `python ingest.py` first")

    required_env = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing Azure env vars: {', '.join(missing)}")

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from rag import get_rag_chain

    return get_rag_chain()


def test_simple_rag(rag_chain):
    query = os.getenv("TEST_QUERY", "What products do you have?")
    resp = rag_chain.invoke(query)
    assert isinstance(resp, str)
    assert resp.strip() != ""
