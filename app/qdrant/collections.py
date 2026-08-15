import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for Collection Names
TEXT_COLLECTION = "omnibrain_text"
IMAGE_COLLECTION = "omnibrain_images"

# Vector Dimensions matching our chosen models
TEXT_VECTOR_DIM = 384   # all-MiniLM-L6-v2 dimension
IMAGE_VECTOR_DIM = 512  # clip-ViT-B-32 dimension

def create_collection_if_not_exists(client: QdrantClient, name: str, vector_size: int):
    """
    Creates a collection with the given name and vector size if it doesn't already exist.
    Uses Cosine similarity for vector similarity matches.
    """
    try:
        # Check if the collection already exists in Qdrant
        collections = client.get_collections().collections
        exists = any(col.name == name for col in collections)
        
        if exists:
            logger.info(f"Collection '{name}' already exists. Skipping creation.")
            return
            
        logger.info(f"Creating collection '{name}' (Vector Size: {vector_size}, Metric: COSINE)...")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Collection '{name}' created successfully.")
        
    except Exception as e:
        logger.error(f"Failed to create collection '{name}': {e}")
        raise RuntimeError(f"Error during collection '{name}' creation: {e}")

def create_omnibrain_collections(client: QdrantClient):
    """
    Creates both the text ('omnibrain_text') and image ('omnibrain_images') collections.
    """
    logger.info("Starting OmniBrain collections initialization...")
    
    # Create the text vector space (384 dimensions)
    create_collection_if_not_exists(client, TEXT_COLLECTION, TEXT_VECTOR_DIM)
    
    # Create the image vector space (512 dimensions)
    create_collection_if_not_exists(client, IMAGE_COLLECTION, IMAGE_VECTOR_DIM)
    
    logger.info("OmniBrain collections initialization complete.")
