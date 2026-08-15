import sys
import os

# Add the project root directory to the Python path to allow app imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings.text_embeddings import generate_text_embedding, MODEL_NAME

def main():
    print("=" * 60)
    print("RUNNING TEXT EMBEDDING TEST")
    print("=" * 60)
    
    sample_text = "Revenue increased significantly during 2025."
    
    try:
        # Generate the embedding
        embedding = generate_text_embedding(sample_text)
        
        # Display requested test details
        print(f"Original Text       : {sample_text}")
        print(f"Embedding Model Name: {MODEL_NAME}")
        print(f"Embedding Dimension : {len(embedding)}")
        print(f"Vector (first 5)    : {embedding[:5]}")
        print("-" * 60)
        
        # Verify invalid input handling works safely
        print("Testing invalid input handling...")
        try:
            generate_text_embedding("")
            print("WARNING: Empty string did not raise an error!")
        except ValueError as e:
            print(f"Successfully caught expected error for empty input: '{e}'")
            
        print("=" * 60)
        print("Text embedding test completed successfully!")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
