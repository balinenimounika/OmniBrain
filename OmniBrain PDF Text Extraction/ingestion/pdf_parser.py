import fitz


def extract_text(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns:
        list: A list containing page-wise text.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text.strip()
        })

    document.close()

    return pages