import streamlit as st
import time, hashlib
from state_manager import initialize_session_state

# Function to inject CSS
def inject_custom_css():
    st.markdown("""
    <style>
        .main-header {
            color: #2c3e50;
            text-align: center;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.3rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        .sidebar-toggle-btn {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 999;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .sidebar-toggle-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.2rem;
            border-radius: 15px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .processing-step {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 14px 18px;
            margin: 10px 0;
            border-radius: 8px;
            font-family: 'SF Pro Display', -apple-system, sans-serif;
            font-size: 0.95rem;
            font-weight: 500;
            color: #2c3e50;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }
        .step-complete {
            background: #e3f2fd !important;
            border-left-color: #1976d2 !important;
            color: #0d47a1 !important;
            font-weight: 600 !important;
        }
        .step-error {
            background: #fff3e0 !important;
            border-left-color: #f57c00 !important;
            color: #e65100 !important;
            font-weight: 600 !important;
        }
        .step-processing {
            background: #f3e5f5 !important;
            border-left-color: #9c27b0 !important;
            color: #4a148c !important;
            font-weight: 500 !important;
        }
        .model-info {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            font-size: 0.9rem;
            color: #1565c0;
        }
        .metrics-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 20px;
            background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
            border-radius: 12px;
            margin: 15px 0;
            animation: pulse 2s infinite;
            border: 1px solid #b3e5fc;
            color: #0277bd;
            font-weight: 500;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
            animation: typing 1.4s infinite ease-in-out;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        .dot:nth-child(3) { animation-delay: 0s; }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.3; }
            40% { transform: scale(1.2); opacity: 1; }
        }
        .stChatInput > div {
            position: fixed !important;
            bottom: 10px !important;
            left: 20px !important;
            right: 20px !important;
            z-index: 100 !important;
            background: white !important;
            border: 2px solid #e0e0e0 !important;
            border-radius: 25px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
            padding: 5px !important;
        }
        .main .block-container {
            padding-bottom: 120px !important;
        }
        .css-1rs6os.edgvbvh3 { display: none; }
        .css-1d391kg {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #28a745; }
        .status-processing { background-color: #ffc107; }
        .status-offline { background-color: #dc3545; }
        .debug-stats {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.85rem;
            color: #856404;
        }
        .step-icon {
            font-size: 1.1em;
            margin-right: 8px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown('<h1 class="main-header">⚖️ DemiDoc Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Legal Document AI Assistant</p>', unsafe_allow_html=True)



def render_document_section(processor):
    initialize_session_state()  # Make sure all session keys exist

    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader("📁 Choose PDF Document", type=["pdf"])
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
        if (not st.session_state.document_uploaded or 
            st.session_state.current_document_hash != file_hash or 
            st.button("🔄 Reprocess Document")):

            with st.spinner("🔍 Processing document..."):
                extraction_result = processor.extract_pdf_with_metadata(uploaded_file.getvalue())
                if extraction_result["success"]:
                    chunks = processor.create_enhanced_chunks(
                        extraction_result["text"], 
                        extraction_result["metadata"]
                    )
                    if chunks and processor.ensure_embeddings():
                        texts = [c["text"] for c in chunks]
                        from langchain_community.vectorstores import FAISS
                        st.session_state.vector_store = FAISS.from_texts(texts, embedding=processor.embeddings)
                        st.session_state.document_uploaded = True
                        st.session_state.document_metadata = extraction_result["metadata"]
                        st.session_state.current_document_hash = file_hash
                        st.session_state.messages = []
                        st.success("✅ Document processed successfully!")
                        st.balloons()
                else:
                    st.error("❌ Failed to process document")



def render_chat(processor):
    initialize_session_state()

    st.markdown('<div class="chat-header">💬 AI Legal Assistant</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your legal document..."):
        if not st.session_state.document_uploaded:
            st.error("📤 Please upload a PDF document first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                thinking = st.empty()
                thinking.markdown("🤔 Thinking...")

                docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in docs])
                analysis = processor.analyze_document_enhanced(context, st.session_state.document_metadata)
                context_result = processor.understand_context_enhanced(analysis, prompt, context)

                response_placeholder = st.empty()
                text = ""
                for chunk in processor.generate_enhanced_explanation(analysis, context_result, prompt):
                    text += chunk
                    response_placeholder.markdown(text + "▌")

                response_placeholder.markdown(text)
                st.session_state.messages.append({"role": "assistant", "content": text})
                thinking.empty()



def render_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 16px; font-weight: bold;'>
            Your AI Assistant for Legal Document Clarity and Confidence
        </div>
        """,
        unsafe_allow_html=True
    )

