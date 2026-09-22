import streamlit as st
import requests

st.set_page_config(
    page_title="OmniBrain",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 OmniBrain")
st.caption("Multi-Modal RAG Assistant")

API_URL = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something about your documents...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": prompt},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get(
                "answer",
                "No answer received from backend."
            )

            sources = data.get("sources", [])
            images = data.get("images", [])

        else:
            answer = f"Backend error: {response.status_code}"
            sources = []
            images = []

    except requests.RequestException:
        answer = "⚠️ FastAPI backend is not running."
        sources = []
        images = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:
            st.markdown("### 📚 Sources")
            for source in sources:
                st.write(source)

        if images:
            st.markdown("### 🖼️ Images")
            for image in images:
                st.image(image)