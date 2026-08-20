import os
import sys
import hashlib
from PIL import Image

# Setup sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION, create_omnibrain_collections
from app.qdrant.insert import insert_text_vector, insert_image_vector

def reset():
    try:
        client = get_qdrant_client()
        print("Clearing Qdrant points...")
        
        # Ensure collections exist first
        create_omnibrain_collections(client)
        
        # Purge text collection
        try:
            scroll_res = client.scroll(collection_name=TEXT_COLLECTION, limit=500)
            points = scroll_res[0]
            if points:
                ids = [p.id for p in points]
                client.delete(collection_name=TEXT_COLLECTION, points_selector=ids)
                print(f"Purged {len(ids)} text points.")
        except Exception as e:
            print(f"Text purge skipped: {e}")
            
        # Purge image collection
        try:
            scroll_res = client.scroll(collection_name=IMAGE_COLLECTION, limit=500)
            points = scroll_res[0]
            if points:
                ids = [p.id for p in points]
                client.delete(collection_name=IMAGE_COLLECTION, points_selector=ids)
                print(f"Purged {len(ids)} image points.")
        except Exception as e:
            print(f"Image purge skipped: {e}")
        
        # Seed text chunk
        print("Seeding text chunk chunk_025_01...")
        text = "The company's revenue increased by 25 percent in 2025 due to strong market growth."
        from app.embeddings.text_embeddings import generate_text_embedding
        vector = generate_text_embedding(text)
        insert_text_vector(
            client=client,
            point_id=1,
            vector=vector,
            document_name="annual_report.pdf",
            page_number=25,
            chunk_id="chunk_025_01",
            source_path="data/documents/annual_report.pdf",
            text=text,
            image_id="image_025_01"
        )
        
        # Seed image vectors
        print("Seeding images...")
        os.makedirs(os.path.join("data", "images"), exist_ok=True)
        img_green_path = os.path.join("data", "images", "sample.png")
        img_red_path = os.path.join("data", "images", "sample2.png")
        
        # Create Forest Green image
        if not os.path.exists(img_green_path):
            img_green = Image.new("RGB", (224, 224), color=(34, 139, 34))
            img_green.save(img_green_path)
            
        # Create Crimson Red image
        if not os.path.exists(img_red_path):
            img_red = Image.new("RGB", (224, 224), color=(220, 20, 60))
            img_red.save(img_red_path)
            
        from app.embeddings.image_embeddings import generate_image_embedding
        
        # Insert Forest Green image (ID: 101)
        vector_green = generate_image_embedding(img_green_path)
        insert_image_vector(
            client=client,
            point_id=101,
            vector=vector_green,
            document_name="annual_report.pdf",
            page_number=25,
            image_id="image_025_01",
            source_path="data/images/sample.png"
        )
        
        # Insert Crimson Red image (ID: 102)
        vector_red = generate_image_embedding(img_red_path)
        insert_image_vector(
            client=client,
            point_id=102,
            vector=vector_red,
            document_name="annual_report.pdf",
            page_number=25,
            image_id="image_025_02",
            source_path="data/images/sample2.png"
        )
        
        print("Qdrant database cleared and re-seeded successfully with clean metadata!")
        client.close()
    except Exception as e:
        print(f"Error resetting database: {e}")

if __name__ == "__main__":
    reset()
