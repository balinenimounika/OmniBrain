def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters

    Returns:
        list of text chunks
    """

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def create_chunks(pages, chunk_size=1000, overlap=200):
    """
    Create chunks while preserving page information.
    """

    all_chunks = []

    chunk_id = 1

    for page in pages:

        page_chunks = chunk_text(
            page["text"],
            chunk_size,
            overlap
        )

        for chunk in page_chunks:

            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": chunk
            })

            chunk_id += 1

    return all_chunks