import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Setup logging to show actions clearly in console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model config
MODEL_NAME = "all-MiniLM-L6-v2"

# Global variable to cache the loaded model
_TEXT_MODEL: Optional[SentenceTransformer] = None

def get_text_model() -> SentenceTransformer:
    """
    Lazy loads and caches the SentenceTransformer text embedding model.
    This prevents the model from being loaded from scratch on every function call.
    """
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        logger.info(f"Loading text embedding model '{MODEL_NAME}'... (This might take a moment the first time)")
        try:
            _TEXT_MODEL = SentenceTransformer(MODEL_NAME)
            logger.info("Text embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load text embedding model '{MODEL_NAME}': {e}")
            raise RuntimeError(f"Could not initialize text embedding model: {e}")
    return _TEXT_MODEL

def generate_text_embedding(text: str) -> List[float]:
    """
    Generates a numerical embedding vector for a given text string.
    
    Args:
        text (str): The input text to embed.
        
    Returns:
        List[float]: The generated embedding vector as a list of floats.
        
    Raises:
        ValueError: If the input text is not a valid non-empty string.
    """
    # Safe validation of empty or invalid text
    if not isinstance(text, str) or not text.strip():
        logger.error("Invalid text input: Input must be a non-empty string.")
        raise ValueError("Input text must be a non-empty string.")
        
    try:
        model = get_text_model()
        # Generate the embedding. convert the numpy array to a native Python list.
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error during text embedding generation: {e}")
        raise
