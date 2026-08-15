import os
import logging
from typing import List, Optional
from PIL import Image
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CLIP-compatible model configuration
MODEL_NAME = "clip-ViT-B-32"

# Global cache for the image model
_IMAGE_MODEL: Optional[SentenceTransformer] = None

def get_image_model() -> SentenceTransformer:
    """
    Lazy loads and caches the CLIP SentenceTransformer image embedding model.
    """
    global _IMAGE_MODEL
    if _IMAGE_MODEL is None:
        logger.info(f"Loading image embedding model '{MODEL_NAME}'... (This might take a moment the first time)")
        try:
            _IMAGE_MODEL = SentenceTransformer(MODEL_NAME)
            logger.info("Image embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load image embedding model '{MODEL_NAME}': {e}")
            raise RuntimeError(f"Could not initialize image embedding model: {e}")
    return _IMAGE_MODEL

def generate_image_embedding(image_path: str) -> List[float]:
    """
    Generates a numerical embedding vector for a given image file path.
    
    Args:
        image_path (str): The absolute or relative path to the image file.
        
    Returns:
        List[float]: The generated image embedding vector as a list of floats.
        
    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is empty, invalid, or cannot be loaded as an image.
    """
    # 1. Path type and emptiness check
    if not isinstance(image_path, str) or not image_path.strip():
        logger.error("Invalid image path provided.")
        raise ValueError("Image path must be a non-empty string.")
        
    # 2. File existence check
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: '{image_path}'")
        raise FileNotFoundError(f"Image file not found at: '{image_path}'")
        
    # 3. Load image safely and generate embedding
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB mode to normalize color channels (e.g. handle RGBA transparency or Grayscale)
            rgb_img = img.convert("RGB")
            
            # Fetch the cached model
            model = get_image_model()
            
            # Generate and convert embedding to standard Python float list
            embedding = model.encode(rgb_img)
            return embedding.tolist()
            
    except Exception as e:
        logger.error(f"Failed to generate embedding for image '{image_path}': {e}")
        raise ValueError(f"Invalid image format or processing error: {e}")
