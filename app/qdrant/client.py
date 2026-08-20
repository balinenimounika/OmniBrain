import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load env variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance cache to ensure singleton access outside Streamlit (e.g. tests or FastAPI)
_CLIENT_INSTANCE = None

try:
    import streamlit as st
    is_streamlit = st.runtime.exists() if hasattr(st, "runtime") and hasattr(st.runtime, "exists") else False
except ImportError:
    is_streamlit = False

if is_streamlit:
    @st.cache_resource
    def get_qdrant_client() -> QdrantClient:
        """
        Initializes and returns a cached QdrantClient instance (singleton) for Streamlit.
        Utilizes st.cache_resource to prevent concurrent database lock collisions.
        """
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        
        # Check if local Qdrant server is running on localhost:6333
        if not url:
            try:
                logger.info("Checking if Qdrant server is active on http://localhost:6333...")
                test_client = QdrantClient(host="localhost", port=6333, timeout=1.0)
                test_client.get_collections()
                logger.info("Successfully connected to shared Qdrant server at localhost:6333")
                return test_client
            except Exception:
                pass
                
        if url:
            logger.info(f"Connecting to Qdrant instance at url: '{url}'")
            if api_key:
                return QdrantClient(url=url, api_key=api_key)
            return QdrantClient(url=url)
        else:
            local_dir = os.path.join("data", "qdrant_db")
            logger.info(f"Using cached local disk storage at: '{local_dir}'")
            os.makedirs(os.path.dirname(local_dir), exist_ok=True)
            return QdrantClient(path=local_dir)
else:
    def get_qdrant_client() -> QdrantClient:
        """
        Initializes and returns a cached QdrantClient instance (singleton) outside Streamlit.
        """
        global _CLIENT_INSTANCE
        if _CLIENT_INSTANCE is not None:
            return _CLIENT_INSTANCE
            
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        
        # Check if local Qdrant server is running on localhost:6333
        if not url:
            try:
                test_client = QdrantClient(host="localhost", port=6333, timeout=1.0)
                test_client.get_collections()
                _CLIENT_INSTANCE = test_client
                return _CLIENT_INSTANCE
            except Exception:
                pass
                
        if url:
            if api_key:
                _CLIENT_INSTANCE = QdrantClient(url=url, api_key=api_key)
            else:
                _CLIENT_INSTANCE = QdrantClient(url=url)
        else:
            local_dir = os.path.join("data", "qdrant_db")
            os.makedirs(os.path.dirname(local_dir), exist_ok=True)
            _CLIENT_INSTANCE = QdrantClient(path=local_dir)
            
        return _CLIENT_INSTANCE
