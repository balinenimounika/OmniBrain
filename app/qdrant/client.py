import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load env variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance cache to ensure singleton access outside Streamlit (e.g. tests or FastAPI)
_CLIENT_INSTANCE = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"

try:
    import streamlit as st
    is_streamlit = st.runtime.exists() if hasattr(st, "runtime") and hasattr(st.runtime, "exists") else False
except ImportError:
    is_streamlit = False

def _create_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()

    if not url:
        try:
            logger.info("Checking if Qdrant server is active on http://localhost:6333...")
            server_client = QdrantClient(host="localhost", port=6333, timeout=1.0)
            server_client.get_collections()
            logger.info("Connected to shared Qdrant server at localhost:6333")
            return server_client
        except Exception:
            pass

    if url:
        logger.info("Connecting to Qdrant instance at url: '%s'", url)
        return QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)

    LOCAL_QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Using local Qdrant storage at '%s'", LOCAL_QDRANT_PATH)
    return QdrantClient(path=str(LOCAL_QDRANT_PATH))


def _get_process_client() -> QdrantClient:
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = _create_qdrant_client()
    return _CLIENT_INSTANCE


if is_streamlit:
    @st.cache_resource(show_spinner=False)
    def get_qdrant_client() -> QdrantClient:
        """Return the one cached Qdrant client for the Streamlit process."""
        return _create_qdrant_client()
else:
    def get_qdrant_client() -> QdrantClient:
        """Return the one process-wide Qdrant client outside Streamlit."""
        return _get_process_client()
