import fitz
import os


def extract_images(pdf_path, output_folder):
    """
    Extract all images from a PDF.

    Returns:
        list of extracted image metadata.
    """

    os.makedirs(output_folder, exist_ok=True)

    document = fitz.open(pdf_path)

    extracted_images = []

    for page_index in range(len(document)):

        page = document[page_index]

        images = page.get_images(full=True)

        for image_index, image in enumerate(images, start=1):

            xref = image[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_ext = image_data["ext"]

            image_id = f"image_{page_index + 1:03d}_{image_index:02d}"
            filename = f"{image_id}.{image_ext}"

            output_path = os.path.join(
                output_folder,
                filename
            )

            with open(output_path, "wb") as image_file:
                image_file.write(image_bytes)

            extracted_images.append({
                "page": page_index + 1,
                "image_id": image_id,
                "filename": filename,
                "path": output_path
            })

    document.close()

    return extracted_images