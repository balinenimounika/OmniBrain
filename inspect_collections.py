import os
import sys
from pathlib import Path

# Setup sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION

def inspect():
    client = get_qdrant_client()
    
    print("--- Inspecting IMAGE COLLECTION ---")
    try:
        results = client.scroll(collection_name=IMAGE_COLLECTION, limit=100)
        points = results[0]
        print(f"Total points found: {len(points)}")
        for p in points:
            payload = p.payload or {}
            print(f"Point ID: {p.id}")
            print(f"  image_id: {payload.get('image_id')}")
            print(f"  source_path: {payload.get('source_path')}")
            print(f"  document_name: {payload.get('document_name')}")
            print(f"  page_number: {payload.get('page_number')}")
            # Check if file exists
            path_val = payload.get('source_path', '')
            exists = os.path.exists(path_val) if path_val else False
            print(f"  File exists locally: {exists}")
            print("-" * 30)
    except Exception as e:
        print(f"Error reading image collection: {e}")
        
    print("\n--- Inspecting TEXT COLLECTION ---")
    try:
        results = client.scroll(collection_name=TEXT_COLLECTION, limit=100)
        points = results[0]
        print(f"Total points found: {len(points)}")
        for p in points:
            payload = p.payload or {}
            print(f"Point ID: {p.id}")
            print(f"  chunk_id: {payload.get('chunk_id')}")
            print(f"  image_id: {payload.get('image_id')}")
            print(f"  page_number: {payload.get('page_number')}")
            print(f"  document_name: {payload.get('document_name')}")
            print(f"  text snippet: {payload.get('text', '')[:60]}...")
            print("-" * 30)
    except Exception as e:
        print(f"Error reading text collection: {e}")
    client.close()

if __name__ == "__main__":
    inspect()
