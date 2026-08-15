from pdf_parser import extract_text
from text_chunker import create_chunks
from image_extractor import extract_images


import os
import json

PDF_PATH = "../data/input/sample.pdf"

TEXT_OUTPUT = "../data/output/text"
CHUNKS_OUTPUT = "../data/output/chunks"
IMAGES_OUTPUT = "../data/output/images"


def main():

    print("Starting OmniBrain PDF ingestion...")

    # --------------------------------
    # 1. Extract text
    # --------------------------------

    print("\n[1/3] Extracting text...")

    pages = extract_text(PDF_PATH)

    os.makedirs(TEXT_OUTPUT, exist_ok=True)

    for page in pages:

        filename = (
            f"page_{page['page']:03d}.txt"
        )

        filepath = os.path.join(
            TEXT_OUTPUT,
            filename
        )

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(page["text"])

    print(f"Extracted text from {len(pages)} pages.")

    # --------------------------------
    # 2. Create chunks
    # --------------------------------

    print("\n[2/3] Creating text chunks...")

    chunks = create_chunks(
        pages,
        chunk_size=1000,
        overlap=200
    )

    os.makedirs(CHUNKS_OUTPUT, exist_ok=True)

    chunks_file = os.path.join(
        CHUNKS_OUTPUT,
        "chunks.json"
    )

    with open(
        chunks_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------
    # 3. Extract images
    # --------------------------------

    print("\n[3/3] Extracting images...")

    images = extract_images(
        PDF_PATH,
        IMAGES_OUTPUT
    )

    print(f"Extracted {len(images)} images.")

    # Save image metadata

    metadata_file = os.path.join(
        IMAGES_OUTPUT,
        "images.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            images,
            file,
            indent=4
        )

    print("\n--------------------------------")
    print("OmniBrain ingestion completed!")
    print("--------------------------------")


if __name__ == "__main__":
    main()