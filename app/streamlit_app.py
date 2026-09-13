import streamlit as st
import base64

from functions import (
    get_pdf_text,
    create_vectorstore_from_texts,
    query_document
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main background ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }


    /* ---------- Remove top padding ---------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ---------- Main title ---------- */

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #111827;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }


    /* ---------- Cards ---------- */

    .card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }


    /* ---------- Section headings ---------- */

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }


    /* ---------- API status ---------- */

    .api-status {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }


    /* ---------- Upload box ---------- */

    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 2px dashed #c7d2fe;
        border-radius: 14px;
        padding: 0.8rem;
    }


    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        background: #4f46e5;
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #4338ca;
        transform: translateY(-1px);
    }


    /* ---------- Success messages ---------- */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ---------- Dataframe ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }


    /* ---------- PDF preview ---------- */

    .pdf-container {
        background: #f1f5f9;
        border-radius: 16px;
        padding: 10px;
        border: 1px solid #e2e8f0;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        margin-top: 3rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Research Paper RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload a research paper and let Gemini extract
        meaningful insights using Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TWO COLUMNS
# ============================================================

col1, col2 = st.columns(
    [0.42, 0.58],
    gap="large"
)


# ============================================================
# LEFT SIDE
# ============================================================

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🔑 Gemini Configuration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="api-status">
            ✓ Gemini API connected
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📄 Upload Research Paper</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    if uploaded_file is not None:

        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            with st.spinner(
                "🔄 Reading and indexing your research paper..."
            ):

                try:

                    documents = get_pdf_text(
                        uploaded_file
                    )

                    st.session_state.vector_store = (
                        create_vectorstore_from_texts(
                            documents,
                            file_name=uploaded_file.name
                        )
                    )

                    st.session_state.uploaded_file_name = (
                        uploaded_file.name
                    )

                    st.success(
                        "✓ Research paper processed successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Error processing PDF: {e}"
                    )


    # ========================================================
    # GENERATE RESULTS
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">✨ Extract Information</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Generate structured information about the paper:"
    )

    if st.button(
        "🚀 Generate Research Paper Table"
    ):

        if st.session_state.vector_store is None:

            st.warning(
                "Please upload a research paper first."
            )

        else:

            with st.spinner(
                "🧠 Gemini is analyzing the research paper..."
            ):

                try:

                    answer = query_document(
                        vectorstore=(
                            st.session_state.vector_store
                        ),
                        query="""
                        Give me the title, summary,
                        publication date, and authors
                        of the research paper.
                        """
                    )

                    st.session_state.answer = answer

                except Exception as e:

                    st.error(
                        f"Error generating answer: {e}"
                    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT SIDE - PDF
# ============================================================

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">👀 Paper Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.info(
            "📄 Upload a research paper to preview it here."
        )

    else:

        bytes_data = uploaded_file.getvalue()

        base64_pdf = base64.b64encode(
            bytes_data
        ).decode("utf-8")

        pdf_display = f"""
        <div class="pdf-container">
            <iframe
                src="data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="750"
                style="
                    border: none;
                    border-radius: 12px;
                    background: white;
                "
            ></iframe>
        </div>
        """

        st.markdown(
            pdf_display,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# RESULTS
# ============================================================

if "answer" in st.session_state:

    st.divider()

    st.markdown(
        "## 📊 Research Paper Information"
    )

    st.dataframe(
        st.session_state.answer,
        use_container_width=True,
        height=350
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Streamlit • LangChain • Chroma • Gemini
    </div>
    """,
    unsafe_allow_html=True
)