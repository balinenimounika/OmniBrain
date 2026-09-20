import os
import sys
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import Optional
from uuid import uuid4

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
from app.qdrant.insert import (
    insert_text_vector,
    insert_image_vector,
    next_chunk_id,
    next_image_id,
)
from app.qdrant.search import search_text_similarity, search_image_similarity
from app.qdrant.ingest import looks_like_user_query
from app.retrieval import retrieval_node
from app.guardrails.guardrails import (
    get_guardrail_response,
    should_answer_question,
    validate_query_scope,
    check_query_relevance,
)


def safe_upload_name(filename: str) -> str:
    """Keep uploaded files inside data/images and discard path components."""
    name = Path(str(filename or "")).name
    return name if name not in {"", ".", ".."} else f"upload_{uuid4().hex}.bin"

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
    
    .block-container {
        padding-top: 2.2rem !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #6c5ce7, #ff7675);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        line-height: 1.35;
        padding-top: 0.6rem;
        padding-bottom: 0.2rem;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        display: block;
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
    
    .pipeline-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 1.25rem;
        gap: 0.35rem;
        flex-wrap: wrap;
    }
    
    .stepper-step {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.65rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .stepper-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        font-size: 0.7rem;
        font-weight: 700;
    }
    
    .step-done {
        background: rgba(46, 213, 115, 0.12);
        border: 1px solid rgba(46, 213, 115, 0.35);
        color: #2ed573;
    }
    .step-done .stepper-num {
        background: #2ed573;
        color: #121212;
    }
    
    .step-active {
        background: rgba(108, 92, 231, 0.15);
        border: 1px solid rgba(108, 92, 231, 0.4);
        color: #a29bfe;
    }
    .step-active .stepper-num {
        background: #6c5ce7;
        color: #ffffff;
    }
    
    .step-blocked {
        background: rgba(255, 71, 87, 0.12);
        border: 1px solid rgba(255, 71, 87, 0.35);
        color: #ff4757;
    }
    .step-blocked .stepper-num {
        background: #ff4757;
        color: #ffffff;
    }
    
    .step-bypassed {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #888888;
    }
    .step-bypassed .stepper-num {
        background: rgba(255, 255, 255, 0.15);
        color: #888888;
    }
    
    .stepper-arrow {
        color: #555555;
        font-size: 0.85rem;
        user-select: none;
    }
    
    .pipeline-stage-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #6c5ce7;
    }
    .stage-passed {
        border-left-color: #2ed573;
        background: rgba(46, 213, 115, 0.04);
    }
    .stage-blocked {
        border-left-color: #ff4757;
        background: rgba(255, 71, 87, 0.04);
    }
    .stage-bypassed {
        border-left-color: #636e72;
        background: rgba(255, 255, 255, 0.02);
    }
    .stage-routed {
        border-left-color: #0984e3;
        background: rgba(9, 132, 227, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DB Initialize -----------------
def seed_default_data_if_empty(client):
    try:
        text_count = client.count(collection_name=TEXT_COLLECTION).count
        image_count = client.count(collection_name=IMAGE_COLLECTION).count
        if text_count == 0 or image_count == 0:
            from app.qdrant.ingest import ingest_extracted_artifacts
            ingest_extracted_artifacts(client)
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
            st.caption("Chunk Reference ID is generated automatically per document and page.")
            source_path = st.text_input("Filesystem Source Path", "data/documents/annual_report.pdf")
            image_id = st.text_input("Associated Image ID (Optional)", "")
            chunk_text = st.text_area("Content Block Text", placeholder="Type the text paragraph to embed...", height=120)
            
            btn_text_ingest = st.form_submit_button("Generate Vector & Store")
            
            if btn_text_ingest:
                if not chunk_text.strip():
                    st.error("❌ Operation failed: Text content cannot be empty.")
                elif looks_like_user_query(chunk_text):
                    st.error("❌ Operation failed: Search questions cannot be stored as document content. Provide source-document text.")
                elif not client:
                    st.error("❌ Operation failed: No Qdrant database connection.")
                else:
                    try:
                        with st.spinner("Creating embedding vector..."):
                            from app.embeddings.text_embeddings import generate_text_embedding
                            vector = generate_text_embedding(chunk_text)
                            
                        with st.spinner("Inserting into Qdrant..."):
                            chunk_id = next_chunk_id(client, doc_name, page_num)
                            insert_text_vector(
                                client=client,
                                point_id=None,
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
                                <p style="font-size:1.05rem; margin-top:0.5rem; margin-bottom:0.5rem;"><b>"{payload.get('content', payload.get('text', ''))}"</b></p>
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
                        safe_name = safe_upload_name(img_file.name)
                        save_path = str(images_root / safe_name)
                        
                        # Save file to filesystem to obtain filepath for core functions
                        with open(save_path, "wb") as f:
                            f.write(img_file.getbuffer())
                            
                        with st.spinner("Creating CLIP vector..."):
                            from app.embeddings.image_embeddings import generate_image_embedding
                            vector = generate_image_embedding(save_path)
                            
                        with st.spinner("Inserting into Qdrant..."):
                            image_id = image_id.strip() or next_image_id(client, doc_name_img, page_num_img)
                            save_relative_path = str(Path("data") / "images" / safe_name)
                            insert_image_vector(
                                client=client,
                                point_id=None,
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
                    query_path = str(images_root / f"query_{safe_upload_name(query_img.name)}")
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
                            img_path = payload.get("image_path") or payload.get("source_path", "")
                            
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
            st.markdown("🛡️ **Guardrails Status:** <span style='color:green; font-weight:bold;'>ACTIVE</span>", unsafe_allow_html=True)
        else:
            st.markdown("🔒 **Qdrant DB Status:** <span style='color:red; font-weight:bold;'>🔴 Disconnected</span>", unsafe_allow_html=True)
            
        st.markdown("**💡 Quick Examples (Click to Populate):**")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if st.button("💰 Revenue Growth (Text)", key="btn_sample_1", use_container_width=True):
                st.session_state["graph_input_query"] = "What was the company's revenue growth in 2025?"
                st.rerun()
            if st.button("📊 Revenue Chart (Image)", key="btn_sample_2", use_container_width=True):
                st.session_state["graph_input_query"] = "Show me the revenue growth chart"
                st.rerun()
        with q_col2:
            if st.button("📈 Operating Performance", key="btn_sample_3", use_container_width=True):
                st.session_state["graph_input_query"] = "What financial performance information is mentioned in the annual report?"
                st.rerun()
            if st.button("🍵 Green Tea (Out-of-Scope)", key="btn_sample_4", use_container_width=True):
                st.session_state["graph_input_query"] = "What are the health benefits of drinking green tea?"
                st.rerun()

        with st.form("langgraph_form"):
            default_query = st.session_state.get("graph_input_query", "What was the company's revenue growth in 2025?")
            user_query = st.text_input("User Query / Prompt", value=default_query, placeholder="e.g., 'What was the company's revenue growth in 2025?'...")
            
            # Guardrails Enforcement Mode selector
            guardrail_mode = st.radio(
                "🛡️ Guardrail Enforcement Mode",
                ["Strict Mode (Enforce Document Scope via NeMo)", "Permissive Mode (Bypass Guardrails & Retrieve Always)"],
                index=0,
                help="Strict mode blocks queries unrelated to annual_report.pdf to prevent hallucinations. Permissive mode allows any query directly to vector retrieval."
            )

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
                            selected_route = "image"
                            route_desc = "Auto-routed to CLIP Image retrieval based on keywords."
                        elif any(w in query_lower for w in ["both", "all", "multimodal", "mix", "hybrid"]):
                            selected_route = "multimodal"
                            route_desc = "Auto-routed to Multimodal retrieval based on keywords."
                        else:
                            selected_route = "text"
                            route_desc = "Auto-routed to Text similarity retrieval."
                    else:
                        if "text" in route_option:
                            selected_route = "text"
                        elif "image" in route_option:
                            selected_route = "image"
                        else:
                            selected_route = "multimodal"
                        route_desc = f"Forced route: '{selected_route}' search."

                    # 2. Guardrail Check against Qdrant database embeddings
                    is_permissive = "Permissive" in guardrail_mode
                    if is_permissive:
                        query_allowed = True
                        query_message = "Guardrail check bypassed via Permissive Mode."
                        guardrail_score = 1.0
                    else:
                        query_allowed, query_message, guardrail_score = check_query_relevance(
                            user_query,
                            client=client,
                            route=selected_route
                        )

                    if not query_allowed:
                        # 1. Pipeline Stepper Header (BLOCKED)
                        st.markdown("""
                        <div class="pipeline-stepper">
                            <div class="stepper-step step-done">
                                <span class="stepper-num">1</span> Query
                            </div>
                            <span class="stepper-arrow">➔</span>
                            <div class="stepper-step step-blocked">
                                <span class="stepper-num">2</span> Guardrail
                            </div>
                            <span class="stepper-arrow">➔</span>
                            <div class="stepper-step step-bypassed">
                                <span class="stepper-num">3</span> Supervisor
                            </div>
                            <span class="stepper-arrow">➔</span>
                            <div class="stepper-step step-bypassed">
                                <span class="stepper-num">4</span> Retrieval
                            </div>
                            <span class="stepper-arrow">➔</span>
                            <div class="stepper-step step-blocked">
                                <span class="stepper-num">5</span> Output State
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Pipeline Stage Cards
                        st.markdown(f"""
                        <div class="pipeline-stage-card stage-passed">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b>Step 1 · User Query Input</b>
                                <span class="meta-tag" style="background:#2ed57322; color:#2ed573; border-color:#2ed57355;">RECEIVED</span>
                            </div>
                            <div style="margin-top:0.4rem; font-size:1.02rem; font-style:italic;">"{user_query}"</div>
                        </div>
                        
                        <div class="pipeline-stage-card stage-blocked">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b>Step 2 · NeMo Guardrails Scope Check</b>
                                <span class="meta-tag" style="background:rgba(255, 159, 67, 0.2); color:#ff9f43; border-color:rgba(255, 159, 67, 0.4);">OUT-OF-SCOPE (PROTECTED)</span>
                            </div>
                            <div style="margin-top:0.5rem; font-size:0.92rem;">
                                <b>Guardrail Policy:</b> {query_message}<br/>
                                <b>Relevance Score:</b> <code>{guardrail_score:.4f}</code> (Required Minimum: <code>0.35</code>)<br/>
                                <span style="color:#ff9f43; font-size:0.85rem;">🛡️ <b>Notice:</b> This is an active security policy, not a system failure. It prevents hallucinated responses to questions outside <code>annual_report.pdf</code>. To retrieve anyway, select <b>Permissive Mode</b> above or click a quick example.</span>
                            </div>
                        </div>

                        <div class="pipeline-stage-card stage-bypassed">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b>Step 3 · Supervisor Router</b>
                                <span class="meta-tag" style="background:#ffffff11; color:#888; border-color:#ffffff22;">BYPASSED</span>
                            </div>
                            <div style="margin-top:0.4rem; font-size:0.88rem; color:#aaa;">
                                Routing intent evaluated as <code>{selected_route}</code>, but execution was bypassed due to guardrail protection.
                            </div>
                        </div>

                        <div class="pipeline-stage-card stage-bypassed">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b>Step 4 · Qdrant Vector Retrieval</b>
                                <span class="meta-tag" style="background:#ffffff11; color:#888; border-color:#ffffff22;">SKIPPED</span>
                            </div>
                            <div style="margin-top:0.4rem; font-size:0.88rem; color:#aaa;">
                                <code>retrieval_status: "skipped"</code> | <code>retrieval_skipped: true</code><br/>
                                Zero queries executed against Qdrant database collections to prevent hallucinations.
                            </div>
                        </div>

                        <div class="pipeline-stage-card stage-blocked">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b>Step 5 · Resulting LangGraph State</b>
                                <span class="meta-tag" style="background:rgba(255, 159, 67, 0.2); color:#ff9f43; border-color:rgba(255, 159, 67, 0.4);">GUARDRAIL PROTECTED STATE</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        blocked_state = {
                            "user_query": user_query,
                            "selected_route": selected_route,
                            "guardrail_status": "BLOCKED",
                            "guardrail_score": round(float(guardrail_score), 4),
                            "retrieval_status": "skipped",
                            "retrieved_text_results": [],
                            "retrieved_image_results": [],
                            "retrieval_skipped": True,
                            "guardrail_message": query_message,
                            "guardrail_allowed": False,
                            "route": selected_route,
                            "retrieval_mode": selected_route,
                            "retrieved_text": [],
                            "retrieved_images": [],
                            "retrieval_results": [],
                            "similarity_scores": [],
                            "document_name": [],
                            "page_number": [],
                            "chunk_id": [],
                            "image_id": [],
                            "source_path": [],
                            "error_message": "",
                        }
                        st.markdown("### 🧠 Resulting LangGraph State")
                        st.json(blocked_state)
                        st.stop()

                    # 1. Pipeline Stepper Header (ALLOWED)
                    st.markdown("""
                    <div class="pipeline-stepper">
                        <div class="stepper-step step-done">
                            <span class="stepper-num">1</span> Query
                        </div>
                        <span class="stepper-arrow">➔</span>
                        <div class="stepper-step step-done">
                            <span class="stepper-num">2</span> Guardrail
                        </div>
                        <span class="stepper-arrow">➔</span>
                        <div class="stepper-step step-active">
                            <span class="stepper-num">3</span> Supervisor
                        </div>
                        <span class="stepper-arrow">➔</span>
                        <div class="stepper-step step-done">
                            <span class="stepper-num">4</span> Retrieval
                        </div>
                        <span class="stepper-arrow">➔</span>
                        <div class="stepper-step step-done">
                            <span class="stepper-num">5</span> Output State
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if is_permissive:
                        step2_badge = "BYPASSED (PERMISSIVE MODE)"
                        step2_style = "background:#0984e322; color:#74b9ff; border-color:#0984e355;"
                        step2_desc = "<b>Mode:</b> Permissive mode active. Guardrail scope check bypassed. Retrieval permitted for all questions."
                    else:
                        step2_badge = "ALLOWED (IN-SCOPE)"
                        step2_style = "background:#2ed57322; color:#2ed573; border-color:#2ed57355;"
                        step2_desc = f"<b>Relevance Score:</b> <code>{guardrail_score:.4f}</code> (Required Threshold: <code>0.35</code>)<br/><b>Policy Decision:</b> Query verified against document knowledge base. Downstream retrieval permitted."

                    # Stages 1, 2, 3 cards
                    st.markdown(f"""
                    <div class="pipeline-stage-card stage-passed">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>Step 1 · User Query Input</b>
                            <span class="meta-tag" style="background:#2ed57322; color:#2ed573; border-color:#2ed57355;">RECEIVED</span>
                        </div>
                        <div style="margin-top:0.4rem; font-size:1.02rem; font-style:italic;">"{user_query}"</div>
                    </div>

                    <div class="pipeline-stage-card stage-passed">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>Step 2 · NeMo Guardrails Scope Check</b>
                            <span class="meta-tag" style="{step2_style}">{step2_badge}</span>
                        </div>
                        <div style="margin-top:0.5rem; font-size:0.92rem;">
                            {step2_desc}
                        </div>
                    </div>

                    <div class="pipeline-stage-card stage-routed">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>Step 3 · Supervisor Router</b>
                            <span class="meta-tag" style="background:#0984e322; color:#74b9ff; border-color:#0984e355;">ROUTED ➔ {selected_route.upper()}</span>
                        </div>
                        <div style="margin-top:0.4rem; font-size:0.9rem;">
                            <b>Mode:</b> {route_option}<br/>
                            <b>Rationale:</b> {route_desc}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 3. Invoke the retrieval node with LangGraph State definition
                    with st.spinner("Invoking LangGraph Retrieval Node..."):
                        initial_state = {
                            "user_query": user_query,
                            "query": user_query,
                            "selected_route": selected_route,
                            "route": selected_route,
                            "retrieval_mode": selected_route,
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
                            "retrieval_status": "started",
                            "retrieval_skipped": False,
                            "guardrail_allowed": True,
                            "guardrail_status": "ALLOWED",
                            "guardrail_score": round(float(guardrail_score), 4),
                            "guardrail_message": None,
                            "error_message": ""
                        }
                        
                        # Run the node in this Streamlit process so every tab shares
                        # the cached embedded Qdrant client.
                        updated_state = retrieval_node(initial_state, client=client)
                        if not is_permissive:
                            updated_state = should_answer_question(updated_state, client=client)
                        else:
                            updated_state["guardrail_allowed"] = True
                            updated_state["guardrail_status"] = "ALLOWED"
                            updated_state["guardrail_message"] = None
                            updated_state["guardrail_score"] = 1.0
                            updated_state["guardrail_reason"] = "Permissive mode active"
                        
                        # Populate all debug state fields
                        updated_state["user_query"] = user_query
                        updated_state["selected_route"] = selected_route
                        updated_state["retrieval_skipped"] = False
                        updated_state["retrieved_text_results"] = updated_state.get("retrieved_text", [])
                        updated_state["retrieved_image_results"] = updated_state.get("retrieved_images", [])
                        if updated_state.get("guardrail_allowed"):
                            updated_state["guardrail_status"] = "ALLOWED"
                            updated_state["guardrail_message"] = None
                            if not updated_state.get("guardrail_score"):
                                updated_state["guardrail_score"] = round(float(guardrail_score), 4)
                        else:
                            updated_state["guardrail_status"] = "BLOCKED"
                        
                    # 3. Check status & render metrics
                    ret_status = updated_state.get("retrieval_status", "empty")
                    err_msg = updated_state.get("error_message", "")
                    
                    # Combine results list for displaying
                    text_results = updated_state.get("retrieved_text", [])
                    image_results = updated_state.get("retrieved_images", [])
                    total_results = text_results + image_results
                    total_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                    target_coll = f"omnibrain_{selected_route}" if selected_route != "multimodal" else "omnibrain_text + omnibrain_images"
                    st.markdown(f"""
                    <div class="pipeline-stage-card stage-passed">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>Step 4 · Qdrant Vector Retrieval</b>
                            <span class="meta-tag" style="background:#2ed57322; color:#2ed573; border-color:#2ed57355;">EXECUTED ({len(total_results)} MATCHES)</span>
                        </div>
                        <div style="margin-top:0.4rem; font-size:0.9rem;">
                            <b>Target Collection(s):</b> <code>{target_coll}</code><br/>
                            <b>Top-k Limit:</b> {top_k_graph} | <b>Status:</b> <code>{ret_status}</code>
                        </div>
                    </div>

                    <div class="pipeline-stage-card stage-passed">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>Step 5 · Resulting LangGraph State & Output</b>
                            <span class="meta-tag" style="background:#6c5ce722; color:#a29bfe; border-color:#6c5ce755;">COMPLETED</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    guardrail_message = get_guardrail_response(updated_state)
                    if guardrail_message and not is_permissive:
                        st.warning(f"🚫 Guardrails restricted this response: {guardrail_message}")
                    elif ret_status == "error":
                        st.error(f"❌ Operation failed: {err_msg}")
                    elif ret_status == "empty":
                        st.info("ℹ️ **Vector Retrieval Complete: 0 matches found in document.**\n\n*(Note: The database contains company financial reports from `annual_report.pdf`. Questions about nutrition/green tea have no matching records in this dataset.)*")
                    else:
                        st.success("✅ LangGraph retrieval completed successfully!")
                        
                    # Render structured outputs
                    st.markdown("### 📥 Retrieved Structured Results")
                    
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


