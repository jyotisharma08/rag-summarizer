import streamlit as st
from rag_pipeline import RAGSummarizer

st.set_page_config(page_title="Smart RAG Summarizer", layout="wide")

st.title("📄✨ Smart RAG Notes Summarizer")

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGSummarizer()

rag = st.session_state.rag

# Sidebar settings
st.sidebar.header("⚙️ Settings")
min_len = st.sidebar.slider("Min Length", 30, 100, 50)
max_len = st.sidebar.slider("Max Length", 100, 300, 120)

# File upload
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    if st.button("🚀 Generate Summary"):
        with st.spinner("Processing... ⏳"):
            try:
                text = rag.extract_text(uploaded_file)
                rag.build_index(text)
                summary = rag.summarize(max_len=max_len, min_len=min_len)

                st.subheader("📌 Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Error: {str(e)}")
