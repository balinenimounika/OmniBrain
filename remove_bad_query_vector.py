"""Remove the known accidental query-as-document vector safely."""

from app.qdrant.client import get_qdrant_client
from app.qdrant.cleanup import remove_text_content

BAD_QUERY = "What factors contributed to revenue growth in 2025?"


if __name__ == "__main__":
    client = get_qdrant_client()
    try:
        removed = remove_text_content(client, BAD_QUERY, document_name="annual_report.pdf")
        print(f"Removed {removed} accidental query vector(s).")
    finally:
        client.close()
