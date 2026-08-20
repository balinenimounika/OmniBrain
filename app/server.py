import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.retrieval.langgraph_integration import retrieval_node
from app.qdrant.client import get_qdrant_client

app = FastAPI(title="OmniBrain API Server")

# Serve the static data/images directory on /images
images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images")
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")

class RetrievalRequest(BaseModel):
    query: str
    route: Optional[str] = "text"
    top_k: Optional[int] = 3

@app.post("/retrieve")
def retrieve(req: RetrievalRequest):
    try:
        initial_state = {
            "user_query": req.query,
            "retrieval_mode": req.route,
            "top_k": req.top_k,
            "conversation_state": {},
            "retrieved_text": [],
            "retrieved_images": [],
            "similarity_scores": [],
            "document_name": [],
            "page_number": [],
            "chunk_id": [],
            "image_id": [],
            "source_path": [],
            "retrieval_status": "empty",
            "error_message": ""
        }
        
        updated_state = retrieval_node(initial_state, client=get_qdrant_client())
        
        # Translate source_path to FastAPI HTTP URL for frontend rendering
        # e.g., "data/images/image_025_01.png" -> "http://localhost:8000/images/image_025_01.png"
        host_url = "http://localhost:8000"
        
        def convert_path_to_url(path_val: str) -> str:
            if not path_val:
                return ""
            # If it is already an HTTP URL, return as is
            if path_val.startswith("http://") or path_val.startswith("https://"):
                return path_val
            filename = os.path.basename(path_val)
            return f"{host_url}/images/{filename}"
            
        # Update state results with HTTP URLs
        if "retrieved_images" in updated_state:
            for item in updated_state["retrieved_images"]:
                item["image_path"] = convert_path_to_url(item.get("source_path", ""))
                item["source_path"] = item["image_path"]
                
        if "retrieval_results" in updated_state:
            for item in updated_state["retrieval_results"]:
                if item.get("type") == "image":
                    item["source_path"] = convert_path_to_url(item.get("source_path", ""))
                    item["image_path"] = item["source_path"]
                elif item.get("image_id") and item.get("image_path"):
                    item["image_path"] = convert_path_to_url(item["image_path"])
                    
        # Update list source paths
        if "source_path" in updated_state:
            updated_state["source_path"] = [
                convert_path_to_url(p) if ("data/images" in p or "OIP" in p or p.endswith(".png") or p.endswith(".jpeg") or p.endswith(".jpg")) else p
                for p in updated_state["source_path"]
            ]
            
        return updated_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
