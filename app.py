"""
app.py
------
Streamlit front-end for the AI Document Summarization System.
Run with:  streamlit run app.py
"""

import hashlib

import streamlit as st

from pdf_extractor import PDFExtractionError, extract_text_from_pdf
from summarizer import MODEL_NAME, check_ollama_status, generate_summary

st.set_page_config(
    page_title="AI Document Summarization System",
    page_icon="📄",
    layout="wide",
)

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("System Information")
    st.markdown(
        f"""
**AI Model:** {MODEL_NAME}  
**LLM:** Ollama  
**PDF Library:** pypdf  
**Frontend:** Streamlit
"""
    )
    st.divider()
    running, model_ok, status_msg = check_ollama_status()
    if running and model_ok:
        st.success("🟢 " + status_msg)
    elif running:
        st.warning("🟡 " + status_msg + f"\n\nRun: `ollama pull {MODEL_NAME}`")
    else:
        st.error("🔴 " + status_msg + " A fallback summary will be used.")

# --------------------------------------------------------------------- Main
st.title("📄 AI Document Summarization System")
st.write(
    "Upload a PDF document and generate a concise AI-powered summary "
    "using a locally running Ollama model."
)

st.subheader("📁 Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is None:
    st.info("Please upload a PDF file to begin.")
    st.stop()

# Extract only once per uploaded file (Streamlit reruns the script on every click).
file_bytes = uploaded_file.getvalue()
file_id = hashlib.md5(file_bytes).hexdigest()

if st.session_state.get("file_id") != file_id:
    try:
        with st.spinner("Extracting text from PDF..."):
            text, total_pages = extract_text_from_pdf(file_bytes)
    except PDFExtractionError as exc:
        st.session_state.pop("file_id", None)
        st.error(f"❌ {exc}")
        st.stop()
    except Exception:
        st.session_state.pop("file_id", None)
        st.error("❌ An unexpected error occurred while reading the PDF.")
        st.stop()

    st.session_state.update(
        file_id=file_id,
        text=text,
        total_pages=total_pages,
        result=None,
    )

text = st.session_state["text"]
total_pages = st.session_state["total_pages"]

st.write(f"**Filename:** {uploaded_file.name}")
st.write(f"**Number of pages:** {total_pages}")
st.success("✅ Text extracted successfully.")

with st.expander("📄 View Extracted Document Text"):
    st.text_area("Extracted text", text, height=400, label_visibility="collapsed")

if st.button("🚀 Generate Summary", type="primary"):
    with st.spinner("🤖 Generating AI summary..."):
        st.session_state["result"] = generate_summary(uploaded_file.name, total_pages, text)

result = st.session_state.get("result")
if result is not None:
    if result.used_fallback:
        st.warning(f"⚠️ {result.warning}")
        st.info("A fallback summary was generated automatically from the extracted text.")
    elif result.warning:
        st.warning(f"⚠️ {result.warning}")

    st.subheader("📋 FINAL DOCUMENT SUMMARY")
    st.markdown(result.summary)

    st.download_button(
        label="📥 Download Summary",
        data=result.summary,
        file_name="document_summary.txt",
        mime="text/plain",
    )
