import os
from PIL import Image


IMAGE_DIR = r"C:\Users\Lenovo\Documents\OmniBrain\data\output\images"


def vision_agent(state):

    query = state["query"]

    print("\n[VISION AGENT]")
    print("Handling image/chart query...")

    # Find available images
    image_files = []

    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(filename)

    if not image_files:
        return {
            "response": "No images found in the document."
        }

    print("Available images:")

    for image in image_files:
        print(" -", image)

    # For now, use the first image
    image_path = os.path.join(
        IMAGE_DIR,
        image_files[0]
    )

    print("Selected image:", image_path)

    # Check whether image can actually be opened
    try:

        img = Image.open(image_path)

        print(
            "Image loaded successfully:",
            img.size
        )

    except Exception as e:

        return {
            "response": f"Could not open image: {e}"
        }

    return {
        "response": (
            f"Vision Agent found image: {image_files[0]}. "
            f"Image size: {img.size}. "
            f"Further VLM analysis can be performed on this image."
        )
    }