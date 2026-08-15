import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load env variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a QdrantClient instance.
    
    If QDRANT_URL is set in environment variables, it connects to that remote/local server.
    If QDRANT_URL is empty or not set, it falls back to local disk-based persistence
    at 'data/qdrant_db' which does not require Docker or a cloud instance to be running.
    """
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    
    try:
        if url:
            logger.info(f"Connecting to Qdrant instance at url: '{url}'")
            if api_key:
                # remote secure setup (e.g. Qdrant cloud)
                client = QdrantClient(url=url, api_key=api_key)
            else:
                # local server setup (e.g. localhost docker container)
                client = QdrantClient(url=url)
        else:
            # Local disk persistence setup
            local_dir = os.path.join("data", "qdrant_db")
            logger.info(f"QDRANT_URL is empty. Falling back to local disk storage at: '{local_dir}'")
            # Ensure containing folder exists
            os.makedirs(os.path.dirname(local_dir), exist_ok=True)
            client = QdrantClient(path=local_dir)
            
        return client
        
    except Exception as e:
        logger.error(f"Error initializing Qdrant client: {e}")
        raise RuntimeError(f"Qdrant connection failure: {e}")
