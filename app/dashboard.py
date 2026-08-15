import os
import sys
import streamlit as st
import hashlib
from PIL import Image

# Ensure the project root directory is in the Python path for importing 'app' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import (
    create_omnibrain_collections,
    TEXT_COLLECTION,
    IMAGE_COLLECTION
)
from app.qdrant.insert import insert_text_vector, insert_image_vector
from app.qdrant.search import search_text_similarity, search_image_similarity

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
@st.cache_resource
def get_or_init_db():
    try:
        client = get_qdrant_client()
        create_omnibrain_collections(client)
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

tab_text, tab_image = st.tabs(["📝 Text Embeddings & Search", "🖼️ Image Embeddings & Search"])

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
            chunk_text = st.text_area("Content Block Text", placeholder="Type the text paragraph to embed...", height=120)
            
            btn_text_ingest = st.form_submit_button("Generate Vector & Store")
            
            if btn_text_ingest:
                if not chunk_text.strip():
                    st.error("Text content cannot be empty.")
                elif not client:
                    st.error("No Qdrant database connection.")
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
                                text=chunk_text
                            )
                        st.success(f"Success! Vector stored in Qdrant with deterministic ID: {deterministic_id}")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Failed to ingest: {ex}")
                        
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
                    st.error("Please upload an image file.")
                elif not client:
                    st.error("No database connection.")
                else:
                    try:
                        # Ensure target folder exists
                        os.makedirs(os.path.join("data", "images"), exist_ok=True)
                        save_path = os.path.join("data", "images", img_file.name)
                        
                        # Save file to filesystem to obtain filepath for core functions
                        with open(save_path, "wb") as f:
                            f.write(img_file.getbuffer())
                            
                        with st.spinner("Creating CLIP vector..."):
                            from app.embeddings.image_embeddings import generate_image_embedding
                            vector = generate_image_embedding(save_path)
                            
                        with st.spinner("Inserting into Qdrant..."):
                            # Compute a deterministic integer ID from image_id
                            deterministic_id = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (10**9)
                            insert_image_vector(
                                client=client,
                                point_id=deterministic_id,
                                vector=vector,
                                document_name=doc_name_img,
                                page_number=page_num_img,
                                image_id=image_id,
                                source_path=save_path
                            )
                        st.success(f"Success! Saved image to '{save_path}' and stored vector with ID: {deterministic_id}")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Failed to ingest image: {ex}")
                        
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
                    os.makedirs(os.path.join("data", "images"), exist_ok=True)
                    query_path = os.path.join("data", "images", f"query_{query_img.name}")
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
                            
                            # Render image if file exists
                            if os.path.exists(img_path):
                                try:
                                    pil_img = Image.open(img_path)
                                    # Limit sizing
                                    st.image(pil_img, width=200)
                                except Exception:
                                    st.warning(f"Could not load image file: {img_path}")
                            else:
                                st.error(f"Image not found at path: {img_path}")
                                
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
