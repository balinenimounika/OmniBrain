import os
import sys
from PIL import Image

# Setup sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION, create_omnibrain_collections
from app.qdrant.ingest import ingest_extracted_artifacts

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
        
        counts = ingest_extracted_artifacts(client)
        print(f"Qdrant database cleared and re-ingested from source artifacts: {counts}")
        client.close()
    except Exception as e:
        print(f"Error resetting database: {e}")

if __name__ == "__main__":
    reset()
