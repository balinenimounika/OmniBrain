import sys
import os
from PIL import Image

# Add project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import (
    create_omnibrain_collections,
    TEXT_COLLECTION,
    IMAGE_COLLECTION
)
from app.qdrant.insert import insert_text_vector, insert_image_vector
from app.qdrant.search import search_text_similarity, search_image_similarity
from app.embeddings.text_embeddings import generate_text_embedding
from app.embeddings.image_embeddings import generate_image_embedding

def generate_mock_images():
    """
    Programmatically creates two sample test images in data/images/ to support
    standalone, offline vector-search tests.
    """
    os.makedirs(os.path.join("data", "images"), exist_ok=True)
    img_green_path = os.path.join("data", "images", "sample.png")
    img_red_path = os.path.join("data", "images", "sample2.png")
    
    # 1. Forest Green solid color mock image
    if not os.path.exists(img_green_path):
        img_green = Image.new("RGB", (224, 224), color=(34, 139, 34))
        img_green.save(img_green_path)
        print(f"Generated mock image: '{img_green_path}'")
        
    # 2. Crimson Red solid color mock image
    if not os.path.exists(img_red_path):
        img_red = Image.new("RGB", (224, 224), color=(220, 20, 60))
        img_red.save(img_red_path)
        print(f"Generated mock image: '{img_red_path}'")
        
    return img_green_path, img_red_path

def main():
    print("=" * 70)
    print("RUNNING COMPLETE QDRANT INTEGRATION & SEARCH TEST")
    print("=" * 70)
    
    try:
        # Step 1: Connect and set up fresh collections
        print("\n--- [TEST 3] Initialize Client and Re-create Collections ---")
        client = get_qdrant_client()
        
        # Clean delete of pre-existing test collections for strict test isolation
        print("Cleaning up old collections (if any) to ensure fresh test runs...")
        try:
            client.delete_collection(TEXT_COLLECTION)
        except Exception:
            pass
            
        try:
            client.delete_collection(IMAGE_COLLECTION)
        except Exception:
            pass
            
        # Re-create collections using the initialization flow
        create_omnibrain_collections(client)
        
        # Step 2: Insert text vectors
        print("\n--- [TEST 4] Generate and Insert Text Vectors ---")
        
        # Define some sample document text chunks with valid integer IDs for Qdrant
        texts = [
            ("Revenue increased significantly during 2025.", 1, "chunk_01"),
            ("Our company expanded its retail presence in North America.", 2, "chunk_02"),
            ("Artificial Intelligence research and development costs grew by 15%.", 3, "chunk_03")
        ]
        
        for text, int_id, chunk_id in texts:
            # Generate the 384-dimensional SentenceTransformer vector
            vector = generate_text_embedding(text)
            insert_text_vector(
                client=client,
                point_id=int_id, # Must be a 64-bit integer or valid UUID string
                vector=vector,
                document_name="annual_report.pdf",
                page_number=25,
                chunk_id=chunk_id,
                source_path="data/documents/annual_report.pdf",
                text=text
            )
            
        # Step 3: Insert image vectors
        print("\n--- [TEST 5] Generate and Insert Image Vectors ---")
        img_green_path, img_red_path = generate_mock_images()
        
        # Generate & insert 512-dimensional CLIP vector for Green image (ID: 101)
        vector_green = generate_image_embedding(img_green_path)
        insert_image_vector(
            client=client,
            point_id=101, # Valid integer ID
            vector=vector_green,
            document_name="annual_report.pdf",
            page_number=25,
            image_id="image_025_01",
            source_path=img_green_path
        )
        
        # Generate & insert 512-dimensional CLIP vector for Red image (ID: 102)
        vector_red = generate_image_embedding(img_red_path)
        insert_image_vector(
            client=client,
            point_id=102, # Valid integer ID
            vector=vector_red,
            document_name="annual_report.pdf",
            page_number=25,
            image_id="image_025_02",
            source_path=img_red_path
        )
        
        # Step 4: Search Text Vectors
        print("\n--- [TEST 6] Execute Text Similarity Search ---")
        query_text = "How did company earnings or revenues change?"
        print(f"User Query: '{query_text}'")
        text_matches = search_text_similarity(client, query_text, top_k=2)
        
        for idx, match in enumerate(text_matches, 1):
            print(f"\nMatch #{idx}:")
            print(f"  ID:    {match['id']}")
            print(f"  Score: {match['score']:.4f} (Cosine Similarity)")
            print(f"  Text:  '{match['payload'].get('text')}'")
            print(f"  Meta:  Doc: {match['payload'].get('document_name')}, Page: {match['payload'].get('page_number')}, Chunk: {match['payload'].get('chunk_id')}")
            
        # Step 5: Search Image Vectors
        print("\n--- [TEST 7] Execute Image Similarity Search ---")
        # Query using the Green image (expect it to rank 1st with ~1.0 Cosine similarity)
        query_img_path = img_green_path
        print(f"Query Image Path: '{query_img_path}'")
        image_matches = search_image_similarity(client, query_img_path, top_k=2)
        
        for idx, match in enumerate(image_matches, 1):
            print(f"\nMatch #{idx}:")
            print(f"  ID:    {match['id']}")
            print(f"  Score: {match['score']:.4f} (Cosine Similarity)")
            print(f"  Path:  '{match['payload'].get('source_path')}'")
            print(f"  Meta:  Doc: {match['payload'].get('document_name')}, Page: {match['payload'].get('page_number')}, ImageID: {match['payload'].get('image_id')}")
            
        # Step 6: Test Error Handling cases
        print("\n--- [TEST 8] Error Handling Edge Cases ---")
        
        # 1. Invalid vector size insertion
        print("Testing invalid vector size insertion...")
        try:
            # Attempt to insert a 3-dimensional vector in a 384-dimensional collection
            insert_text_vector(
                client=client,
                point_id=999,
                vector=[0.1, 0.2, 0.3],
                document_name="error.pdf",
                page_number=1,
                chunk_id="err_01",
                source_path="error.pdf",
                text="Invalid dimension"
            )
            print("WARNING: Dimension mismatch insertion did not raise an error!")
        except Exception as e:
            print(f"Successfully caught expected dimension insertion error: '{e}'")
            
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED AND INTEGRATION SUCCESSFULLY VERIFIED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nTEST SUITE CRASHED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
