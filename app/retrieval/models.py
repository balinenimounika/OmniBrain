from pydantic import BaseModel
from typing import Optional, Union

class RetrievalResult(BaseModel):
    id: Union[str, int]
    modality: str  # "text" or "image"
    type: str = ""  # "text" or "image" for compatibility
    score: float
    document_name: str
    page_number: int
    chunk_id: Optional[str] = None
    image_id: Optional[str] = None
    source_path: str
    content: Optional[str] = None  # text content for text results
    image_path: Optional[str] = None  # image path/reference for image results

    def __init__(self, **data):
        super().__init__(**data)
        if not self.type:
            self.type = self.modality

