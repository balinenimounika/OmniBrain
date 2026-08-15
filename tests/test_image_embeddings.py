import sys
import os
from PIL import Image

# Add project root directory to the Python path to allow app imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings.image_embeddings import generate_image_embedding, MODEL_NAME

def create_sample_image(path: str):
    """
    Generates a mock image for testing purposes if it doesn't already exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Creating sample test image at: '{path}'")
        # Create a simple solid blue square
        img = Image.new("RGB", (224, 224), color=(34, 139, 34)) # Forest green
        img.save(path)
        print("Sample image created.")

def main():
    print("=" * 60)
    print("RUNNING IMAGE EMBEDDING TEST")
    print("=" * 60)
    
    sample_image_path = os.path.join("data", "images", "sample.png")
    
    try:
        # Create sample image to ensure zero external dependency setup
        create_sample_image(sample_image_path)
        
        # Generate the embedding
        embedding = generate_image_embedding(sample_image_path)
        
        # Display requested test details
        print(f"Image Path          : {sample_image_path}")
        print(f"Embedding Model Name: {MODEL_NAME}")
        print(f"Embedding Dimension : {len(embedding)}")
        print(f"Vector (first 5)    : {embedding[:5]}")
        print("-" * 60)
        
        # Test error handling: missing image file
        print("Testing missing file handling...")
        missing_path = os.path.join("data", "images", "does_not_exist.png")
        try:
            generate_image_embedding(missing_path)
            print("WARNING: Missing file did not raise FileNotFoundError!")
        except FileNotFoundError as e:
            print(f"Successfully caught expected error for missing file: '{e}'")
            
        # Test error handling: invalid file format
        print("Testing invalid file handling...")
        invalid_path = os.path.join("data", "images", "not_an_image.txt")
        with open(invalid_path, "w") as f:
            f.write("This text file is not a valid image format.")
            
        try:
            generate_image_embedding(invalid_path)
            print("WARNING: Invalid file did not raise ValueError!")
        except ValueError as e:
            print(f"Successfully caught expected error for invalid file: '{e}'")
        finally:
            # Clean up the temporary invalid text file
            if os.path.exists(invalid_path):
                os.remove(invalid_path)
                
        print("=" * 60)
        print("Image embedding test completed successfully!")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
