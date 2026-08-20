import os
import sys
import streamlit as st
import hashlib
from PIL import Image
from pathlib import Path
from typing import Optional

# Ensure the project root directory is in the Python path for importing 'app' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_image_path(stored_path: str) -> Optional[Path]:
    if not stored_path:
        return None
    try:
        normalized_path = str(stored_path).replace("\\", "/")
        p = Path(normalized_path)
        # Check if absolute and exists
        if p.is_absolute():
            if p.exists() and p.is_file():
                return p
        
        # Check relative to PROJECT_ROOT
        resolved = (PROJECT_ROOT / normalized_path).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
            
        # Fallback: check relative to data/images using just the file name
        resolved_filename = PROJECT_ROOT / "data" / "images" / p.name
        if resolved_filename.exists() and resolved_filename.is_file():
            return resolved_filename
            
        return None
    except Exception:
        return None


from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import (
    create_omnibrain_collections,
    TEXT_COLLECTION,
    IMAGE_COLLECTION
)
from app.qdrant.insert import insert_text_vector, insert_image_vector
from app.qdrant.search import search_text_similarity, search_image_similarity
from app.retrieval import retrieval_node

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="OmniBrain Dashboard - Multi-Modal RAG Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom visual CSS styling injection for premium appearance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #6c5ce7, #ff7675);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        border-color: #6c5ce7;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.1);
    }
    
    .score-badge {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: white;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .meta-tag {
        font-size: 0.75rem;
        color: #ccc;
        background: rgba(255, 255, 255, 0.06);
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DB Initialize -----------------
def seed_default_data_if_empty(client):
    try:
        # Check text collection count
        text_count = client.count(collection_name=TEXT_COLLECTION).count
        if text_count == 0:
            text = "The company's revenue increased by 25 percent in 2025 due to strong market growth."
            from app.embeddings.text_embeddings import generate_text_embedding
            vector = generate_text_embedding(text)
            insert_text_vector(
                client=client,
                point_id=1,
                vector=vector,
                document_name="annual_report.pdf",
                page_number=25,
                chunk_id="chunk_025_01",
                source_path="data/documents/annual_report.pdf",
                text=text,
                image_id="image_025_01"
            )
            
        # Check image collection count
        image_count = client.count(collection_name=IMAGE_COLLECTION).count
        if image_count == 0:
            os.makedirs(os.path.join("data", "images"), exist_ok=True)
            images_root = PROJECT_ROOT / "data" / "images"
            images_root.mkdir(parents=True, exist_ok=True)
            img_green_path = str(images_root / "sample.png")
            img_red_path = str(images_root / "sample2.png")
            
            # Create Forest Green image
            if not os.path.exists(img_green_path):
                img_green = Image.new("RGB", (224, 224), color=(34, 139, 34))
                img_green.save(img_green_path)
                
            # Create Crimson Red image
            if not os.path.exists(img_red_path):
                img_red = Image.new("RGB", (224, 224), color=(220, 20, 60))
                img_red.save(img_red_path)
                
            from app.embeddings.image_embeddings import generate_image_embedding
            
            # Insert Forest Green image (ID: 101)
            vector_green = generate_image_embedding(img_green_path)
            insert_image_vector(
                client=client,
                point_id=101,
                vector=vector_green,
                document_name="annual_report.pdf",
                page_number=25,
                image_id="image_025_01",
                source_path=img_green_path
            )
            
            # Insert Crimson Red image (ID: 102)
            vector_red = generate_image_embedding(img_red_path)
            insert_image_vector(
                client=client,
                point_id=102,
                vector=vector_red,
                document_name="annual_report.pdf",
                page_number=25,
                image_id="image_025_02",
                source_path=img_red_path
            )
    except Exception as e:
        print(f"Warning: Could not seed default database collections: {e}")

@st.cache_resource
def get_or_init_db():
    try:
        client = get_qdrant_client()
        create_omnibrain_collections(client)
        seed_default_data_if_empty(client)
        return client, None
    except Exception as e:
        return None, str(e)


client, db_err = get_or_init_db()

# ----------------- Sidebar Panel -----------------
st.sidebar.markdown("## 🧠 OmniBrain Backend")
st.sidebar.markdown("### System Connection")

if db_err:
    st.sidebar.error(f"🔴 Connection Failed:\n{db_err}")
else:
    st.sidebar.success("🟢 Connected to local Qdrant database")
    
    # Helper stats
    try:
        text_stats = client.count(collection_name=TEXT_COLLECTION).count
        image_stats = client.count(collection_name=IMAGE_COLLECTION).count
    except Exception:
        text_stats, image_stats = 0, 0
        
    st.sidebar.markdown("**Active Metrics**:")
    st.sidebar.write(f"- Text Items: **{text_stats}**")
    st.sidebar.write(f"- Image Items: **{image_stats}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Control Panel")
if st.sidebar.button("🧹 Clear & Reset Collections"):
    if client:
        try:
            client.delete_collection(TEXT_COLLECTION)
            client.delete_collection(IMAGE_COLLECTION)
            create_omnibrain_collections(client)
            st.sidebar.success("Database reset successfully.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to reset: {e}")

# ----------------- Dashboard Layout -----------------
st.markdown("<div class='main-header'>OmniBrain - Localhost RAG Client</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Multi-Modal Vector Database & Similarity Search Dashboard</div>", unsafe_allow_html=True)

tab_text, tab_image, tab_graph = st.tabs([
    "📝 Text Embeddings & Search",
    "🖼️ Image Embeddings & Search",
    "🤖 LangGraph Flow & Retrieval"
])

# ================= TAB 1: TEXT SPACE =================
with tab_text:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📥 Ingest Text Chunk")
        with st.form("text_ingest_form"):
            doc_name = st.text_input("Document Name Source", "annual_report.pdf")
            page_num = st.number_input("Page Number", min_value=1, value=25, step=1)
            chunk_id = st.text_input("Chunk Reference ID", "chunk_025_01")
            source_path = st.text_input("Filesystem Source Path", "data/documents/annual_report.pdf")
            image_id = st.text_input("Associated Image ID (Optional)", "")
            chunk_text = st.text_area("Content Block Text", placeholder="Type the text paragraph to embed...", height=120)
            
            btn_text_ingest = st.form_submit_button("Generate Vector & Store")
            
            if btn_text_ingest:
                if not chunk_text.strip():
                    st.error("❌ Operation failed: Text content cannot be empty.")
                elif not client:
                    st.error("❌ Operation failed: No Qdrant database connection.")
                else:
                    try:
                        with st.spinner("Creating embedding vector..."):
                            from app.embeddings.text_embeddings import generate_text_embedding
                            vector = generate_text_embedding(chunk_text)
                            
                        with st.spinner("Inserting into Qdrant..."):
                            # Compute a deterministic integer ID from chunk_id
                            deterministic_id = int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (10**9)
                            insert_text_vector(
                                client=client,
                                point_id=deterministic_id,
                                vector=vector,
                                document_name=doc_name,
                                page_number=page_num,
                                chunk_id=chunk_id,
                                source_path=source_path,
                                text=chunk_text,
                                image_id=image_id.strip() if image_id.strip() else None
                            )
                        st.success("✅ Text vector generated and stored successfully!")
                    except Exception as ex:
                        st.error(f"❌ Operation failed: {ex}")
                        
    with col2:
        st.subheader("🔍 Search Text Vectors")
        query_text = st.text_input("Enter Search Phrase", placeholder="Type a concept or statement (e.g., 'revenue expansion')...")
        top_k_text = st.slider("Max Results Limit", min_value=1, max_value=10, value=3)
        
        if query_text:
            if not client:
                st.error("Database connection unavailable.")
            else:
                try:
                    with st.spinner("Searching..."):
                        matches = search_text_similarity(client, query_text, top_k=top_k_text)
                        
                    if not matches:
                        st.info("No matching text vectors found.")
                    else:
                        st.markdown(f"##### Showing top {len(matches)} results:")
                        for item in matches:
                            payload = item["payload"]
                            st.markdown(f"""
                            <div class="card">
                                <span class="score-badge">Similarity Match Score: {item['score']:.4f}</span>
                                <p style="font-size:1.05rem; margin-top:0.5rem; margin-bottom:0.5rem;"><b>"{payload.get('text', '')}"</b></p>
                                <div class="meta-row">
                                    <span class="meta-tag"><b>ID:</b> {item['id']}</span>
                                    <span class="meta-tag"><b>Doc:</b> {payload.get('document_name')}</span>
                                    <span class="meta-tag"><b>Page:</b> {payload.get('page_number')}</span>
                                    <span class="meta-tag"><b>Chunk:</b> {payload.get('chunk_id')}</span>
                                    <span class="meta-tag"><b>Source:</b> {payload.get('source_path')}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Search failed: {ex}")

# ================= TAB 2: IMAGE SPACE =================
with tab_image:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📥 Upload & Ingest Image")
        with st.form("image_ingest_form", clear_on_submit=True):
            img_file = st.file_uploader("Upload Image File (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
            doc_name_img = st.text_input("Associated Document Name", "annual_report.pdf")
            page_num_img = st.number_input("Source Page Number", min_value=1, value=25, step=1)
            image_id = st.text_input("Image ID Reference", "image_025_01")
            
            btn_img_ingest = st.form_submit_button("Generate CLIP Embedding & Store")
            
            if btn_img_ingest:
                if not img_file:
                    st.error("❌ Operation failed: Please upload an image file.")
                elif not client:
                    st.error("❌ Operation failed: No database connection.")
                else:
                    try:
                        # Ensure target folder exists
                        images_root = PROJECT_ROOT / "data" / "images"
                        images_root.mkdir(parents=True, exist_ok=True)
                        save_path = str(images_root / img_file.name)
                        
                        # Save file to filesystem to obtain filepath for core functions
                        with open(save_path, "wb") as f:
                            f.write(img_file.getbuffer())
                            
                        with st.spinner("Creating CLIP vector..."):
                            from app.embeddings.image_embeddings import generate_image_embedding
                            vector = generate_image_embedding(save_path)
                            
                        with st.spinner("Inserting into Qdrant..."):
                            # Compute a deterministic integer ID from image_id
                            deterministic_id = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (10**9)
                            save_relative_path = f"data/images/{img_file.name}"
                            insert_image_vector(
                                client=client,
                                point_id=deterministic_id,
                                vector=vector,
                                document_name=doc_name_img,
                                page_number=page_num_img,
                                image_id=image_id,
                                source_path=save_relative_path
                            )
                        st.success("✅ Image CLIP embedding generated and stored successfully!")
                    except Exception as ex:
                        st.error(f"❌ Operation failed: {ex}")
                        
    with col2:
        st.subheader("🔍 Search Image Vectors")
        query_img = st.file_uploader("Upload Query Image to Find Matches", type=["png", "jpg", "jpeg"])
        top_k_img = st.slider("Matches Limit", min_value=1, max_value=10, value=3)
        
        if query_img:
            if not client:
                st.error("Database connection unavailable.")
            else:
                try:
                    # Save query image locally to pass to generator
                    images_root = PROJECT_ROOT / "data" / "images"
                    images_root.mkdir(parents=True, exist_ok=True)
                    query_path = str(images_root / f"query_{query_img.name}")
                    with open(query_path, "wb") as f:
                        f.write(query_img.getbuffer())
                        
                    with st.spinner("Searching image vector space..."):
                        matches = search_image_similarity(client, query_path, top_k=top_k_img)
                        
                    # Clean up query file
                    if os.path.exists(query_path):
                        os.remove(query_path)
                        
                    if not matches:
                        st.info("No matching image vectors found.")
                    else:
                        st.markdown(f"##### Showing top {len(matches)} matches:")
                        for item in matches:
                            payload = item["payload"]
                            img_path = payload.get("source_path", "")
                            
                            st.markdown(f"""
                            <div class="card">
                                <span class="score-badge">Similarity Match Score: {item['score']:.4f}</span>
                                <div style="display:flex; gap:1.5rem; align-items:center;">
                            """, unsafe_allow_html=True)
                            
                            # Render image if file can be resolved or is a web URL
                            if isinstance(img_path, str) and (img_path.startswith("http://") or img_path.startswith("https://")):
                                st.image(img_path, width=200)
                            else:
                                resolved_img = resolve_image_path(img_path)
                                if resolved_img:
                                    try:
                                        pil_img = Image.open(resolved_img)
                                        # Limit sizing
                                        st.image(pil_img, width=200)
                                    except Exception:
                                        st.caption("Image not found")
                                else:
                                    st.caption("Image not found")
                                
                            st.markdown(f"""
                                    <div>
                                        <div class="meta-row">
                                            <span class="meta-tag"><b>ID:</b> {item['id']}</span>
                                            <span class="meta-tag"><b>Doc:</b> {payload.get('document_name')}</span>
                                            <span class="meta-tag"><b>Page:</b> {payload.get('page_number')}</span>
                                            <span class="meta-tag"><b>Image ID:</b> {payload.get('image_id')}</span>
                                            <span class="meta-tag"><b>Path:</b> {payload.get('source_path')}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Search failed: {ex}")

# ================= TAB 3: LANGGRAPH FLOW & RETRIEVAL =================
with tab_graph:
    st.subheader("🤖 LangGraph Node & Retrieval Integration Flow")
    st.markdown("""
    This screen demonstrates the complete multi-modal retrieval integration flow:
    `User Query` ➔ `Supervisor Router` ➔ `Retrieval Node` ➔ `Qdrant DB` ➔ `LangGraph State`
    """)
    
    col_input, col_flow = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("📥 Input Parameters")
        
        # Display Qdrant Connection Status dynamically
        if client:
            st.markdown("🔒 **Qdrant DB Status:** <span style='color:green; font-weight:bold;'>🟢 Connected (Local Disk)</span>", unsafe_allow_html=True)
        else:
            st.markdown("🔒 **Qdrant DB Status:** <span style='color:red; font-weight:bold;'>🔴 Disconnected</span>", unsafe_allow_html=True)
            
        with st.form("langgraph_form"):
            user_query = st.text_input("User Query / Prompt", placeholder="e.g., 'What was the company's revenue growth in 2025?'...")
            
            # Simulator Route Decision
            route_option = st.selectbox(
                "Supervisor Routing Mode",
                ["Auto Router", "Force Text Retrieval (text)", "Force CLIP Image Retrieval (image)", "Force Multimodal Retrieval (multimodal)"]
            )
            
            top_k_graph = st.slider("Top-k Results Limit", min_value=1, max_value=10, value=3)
            
            btn_run_graph = st.form_submit_button("🚀 Run LangGraph Flow Node")
            
    with col_flow:
        st.subheader("📊 Execution Flow & LangGraph State")
        
        if btn_run_graph:
            if not user_query.strip():
                st.error("❌ Operation failed: Please enter a query.")
            elif not client:
                st.error("❌ Operation failed: Database connection unavailable.")
            else:
                try:
                    # 1. Simulate Supervisor router decision
                    if route_option == "Auto Router":
                        query_lower = user_query.lower()
                        if any(w in query_lower for w in ["image", "photo", "picture", "chart", "draw", "show", "diagram"]):
                            route = "image"
                            route_desc = "Auto-routed to CLIP Image retrieval based on keywords."
                        elif any(w in query_lower for w in ["both", "all", "multimodal", "mix", "hybrid"]):
                            route = "multimodal"
                            route_desc = "Auto-routed to Multimodal retrieval based on keywords."
                        else:
                            route = "text"
                            route_desc = "Auto-routed to Text similarity retrieval."
                    else:
                        if "text" in route_option:
                            route = "text"
                        elif "image" in route_option:
                            route = "image"
                        else:
                            route = "multimodal"
                        route_desc = f"Forced route: '{route}' search."
                        
                    # Display route selection & status parameters
                    st.info(f"🔄 **Supervisor Node Decided Route:** `{route}`\n*{route_desc}*")
                    
                    # 2. Invoke the retrieval node with LangGraph State definition
                    with st.spinner("Invoking LangGraph Retrieval Node..."):
                        initial_state = {
                            "user_query": user_query,
                            "retrieval_mode": route,
                            "top_k": top_k_graph,
                            "conversation_state": {"session_id": "session_demo_999"},
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
                        
                        # Run the node in this Streamlit process so every tab shares
                        # the cached embedded Qdrant client.
                        updated_state = retrieval_node(initial_state, client=client)
                        
                    # 3. Check status & render metrics
                    ret_status = updated_state.get("retrieval_status", "empty")
                    err_msg = updated_state.get("error_message", "")
                    
                    if ret_status == "error":
                        st.error(f"❌ Operation failed: {err_msg}")
                    elif ret_status == "empty":
                        st.warning("⚠️ **LangGraph Retrieval Node Complete! (STATUS: NO RESULTS)**")
                    else:
                        st.success("✅ LangGraph retrieval completed successfully!")
                        
                    # Render structured outputs
                    st.markdown("### 📥 Retrieved Structured Results")
                    
                    # Combine results list for displaying
                    text_results = updated_state.get("retrieved_text", [])
                    image_results = updated_state.get("retrieved_images", [])
                    total_results = text_results + image_results
                    
                    # Sort combined list by score desc
                    total_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                    
                    if not total_results:
                        st.warning("No results retrieved from the vector database.")
                    else:
                        st.write(f"Found **{len(total_results)}** structured matches:")
                        for idx, item in enumerate(total_results, 1):
                            modality = item.get("type", "text")
                            score = item.get("score", 0.0)
                            doc_name = item.get("document_name", "Unknown")
                            page_num = item.get("page_number", -1)
                            source_path = item.get("source_path", "")
                            
                            st.markdown(f"""
                            <div class="card">
                                <span class="score-badge">{modality.upper()} | Similarity: {score:.4f}</span>
                                <div style="margin-top: 0.5rem; font-size: 0.95rem;">
                                    <b>Doc:</b> {doc_name} | <b>Page:</b> {page_num} | <b>Source:</b> {source_path}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if modality == "text":
                                chunk_id = item.get("chunk_id", "N/A")
                                text_content = item.get("content", "")
                                st.markdown(f"""
                                    <div style="margin-top: 0.5rem; padding-left: 0.5rem; border-left: 3px solid #6c5ce7; font-style: italic;">
                                        <b>Chunk ID:</b> {chunk_id}<br/>
                                        "{text_content}"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                image_id = item.get("image_id", "N/A")
                                st.markdown(f"<b>Image ID:</b> {image_id}</div>", unsafe_allow_html=True)
                                # Render image if file can be resolved or is a web URL
                                if isinstance(source_path, str) and (source_path.startswith("http://") or source_path.startswith("https://")):
                                    st.image(source_path, width=250)
                                else:
                                    resolved_source = resolve_image_path(source_path)
                                    if resolved_source:
                                        try:
                                            pil_img = Image.open(resolved_source)
                                            st.image(pil_img, width=250)
                                        except Exception:
                                            st.caption("Image not found")
                                    else:
                                        st.caption("Image not found")
                                    
                    # Display final State dict
                    st.markdown("### 🧠 Resulting LangGraph State")
                    st.json(updated_state)
                    
                except Exception as ex:
                    st.error(f"❌ Operation failed: {ex}")


