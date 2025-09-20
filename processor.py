import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2, os, logging, time, hashlib
from io import BytesIO
from datetime import datetime
from typing import Generator, List, Dict, Any 
from dotenv import load_dotenv
from state_manager import initialize_session_state

load_dotenv()

class EnhancedLegalDocumentProcessor:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        # self.api_key = st.secrets["GOOGLE_API_KEY"]

        if not self.api_key:
            st.error("🔑 Google API key required. Set GOOGLE_API_KEY in environment.")
            st.stop()

        genai.configure(api_key=self.api_key)
        self.embeddings = None
        self.available_models = self._check_available_models()

        # Fixed: Better tracking with detailed stats
        self.processing_stats = {
            "total_queries": 0, 
            "successful_responses": 0, 
            "blocked_responses": 0,
            "failed_responses": 0,
            "partial_responses": 0  # New: for responses with some content but issues
        }

        # Optimized safety settings
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]

    def _check_available_models(self) -> List[str]:
        """Enhanced model checking with caching"""
        try:
            if hasattr(st.session_state, 'cached_models'):
                return st.session_state.cached_models

            models = genai.list_models()
            available = [model.name for model in models if 'generateContent' in model.supported_generation_methods]

            st.session_state.cached_models = available
            logging.info(f"Found {len(available)} available models")
            return available
        except Exception as e:
            logging.error(f"Model check failed: {e}")
            return ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]

    def _get_optimal_model(self, task_type: str) -> str:
        """Intelligent model selection based on task"""
        model_strategies = {
            "analysis": {
                "preferred": ["gemini-1.5-flash-latest", "gemini-2.5-flash"],
                "fallback": ["gemini-1.5-flash", "gemini-1.5-pro"]
            },
            "understanding": {
                "preferred": ["gemini-2.0-flash-lite", "gemini-2.0-flash"],
                "fallback": ["gemini-1.5-flash", "gemini-2.5-flash"]
            },
            "explanation": {
                "preferred": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"],
                "fallback": ["gemini-1.5-flash", "gemini-1.5-pro"]
            }
        }

        strategy = model_strategies.get(task_type, model_strategies["analysis"])

        for model in strategy["preferred"]:
            if f"models/{model}" in self.available_models:
                return model

        for model in strategy["fallback"]:
            if f"models/{model}" in self.available_models:
                logging.info(f"Using fallback {model} for {task_type}")
                return model

        return self.available_models[0].replace("models/", "") if self.available_models else "gemini-1.5-flash"

    def ensure_embeddings(self) -> bool:
        """Optimized embedding loading with progress indication"""
        if self.embeddings is None:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🔄 Initializing embeddings model...")
                progress_bar.progress(25)

                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

                progress_bar.progress(100)
                status_text.text("✅ Embeddings ready!")
                time.sleep(1)

                progress_bar.empty()
                status_text.empty()
                return True

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                logging.error(f"Embedding initialization failed: {e}")
                st.error("⚠️ Failed to load embeddings. Some features may be limited.")
                return False
        return True

    def extract_pdf_with_metadata(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Enhanced PDF extraction with metadata"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

            pages_text = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_text.append({"page": page_num + 1, "text": text})
                except Exception as e:
                    logging.warning(f"Page {page_num + 1} extraction failed: {e}")

            full_text = "\n".join([p["text"] for p in pages_text])

            metadata = {
                "total_pages": len(pdf_reader.pages),
                "pages_with_text": len(pages_text),
                "total_characters": len(full_text),
                "word_count": len(full_text.split()) if full_text else 0,
                "extraction_time": datetime.now().isoformat(),
                "document_hash": hashlib.md5(pdf_bytes).hexdigest()[:8]
            }

            return {
                "text": full_text,
                "pages": pages_text,
                "metadata": metadata,
                "success": len(pages_text) > 0
            }

        except Exception as e:
            logging.exception("PDF extraction failed")
            return {"text": "", "pages": [], "metadata": {}, "success": False, "error": str(e)}

    def create_enhanced_chunks(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Smart text chunking with metadata preservation"""
        if not text.strip():
            return []

        doc_size = len(text)
        if doc_size < 5000:
            chunk_size = 800
        elif doc_size < 20000:
            chunk_size = 1000
        else:
            chunk_size = 1200

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

        raw_chunks = splitter.split_text(text)

        enhanced_chunks = []
        for i, chunk in enumerate(raw_chunks):
            enhanced_chunks.append({
                "text": chunk,
                "chunk_id": i,
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "document_hash": metadata.get("document_hash", "unknown")
            })

        logging.info(f"Created {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks

    def _safe_generate_content(self, model, prompt: str, stream: bool = False):
        """Enhanced content generation with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if stream:
                    response = model.generate_content(
                        prompt,
                        stream=True,
                        safety_settings=self.safety_settings,
                        generation_config={
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 2048,
                        }
                    )
                else:
                    response = model.generate_content(
                        prompt,
                        safety_settings=self.safety_settings,
                        generation_config={
                            "temperature": 0.6,
                            "top_p": 0.8,
                            "max_output_tokens": 1024,
                        }
                    )
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Content generation failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    # Layer implementations remain the same but with better error handling
    def analyze_document_enhanced(self, context: str, metadata: Dict) -> Dict[str, Any]:
        """Enhanced document analysis with structured output"""
        model_name = self._get_optimal_model("analysis")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        You are a legal document analysis expert. Analyze this document systematically.

        DOCUMENT METADATA:
        - Pages: {metadata.get('total_pages', 'Unknown')}
        - Word Count: {metadata.get('word_count', 'Unknown')}
        - Document ID: {metadata.get('document_hash', 'Unknown')}

        DOCUMENT CONTENT (First 6000 chars):
        {context[:6000]}

        Please provide structured analysis:
        1. DOCUMENT TYPE: [Contract/Agreement/Policy/Legal Notice/Other]
        2. PRIMARY PURPOSE: [Brief description]
        3. KEY SECTIONS: [List main sections found]
        4. IMPORTANT TERMS: [Key legal terms or concepts]
        5. COMPLEXITY LEVEL: [Simple/Moderate/Complex]

        Be concise and factual. Focus on document structure and legal elements.
        """

        try:
            response = self._safe_generate_content(model, prompt)
            if response and response.text:
                return {
                    "analysis": response.text,
                    "model_used": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
            else:
                return {
                    "analysis": "Document structure analysis completed with limited details.",
                    "model_used": model_name,
                    "success": False,
                    "reason": "No response content"
                }
        except Exception as e:
            logging.exception("Enhanced document analysis failed")
            return {
                "analysis": "Unable to complete detailed document analysis.",
                "success": False,
                "error": str(e)
            }

    def understand_context_enhanced(self, analysis_result: Dict, question: str, context: str) -> Dict[str, Any]:
        """Enhanced context understanding with relevance scoring"""
        model_name = self._get_optimal_model("understanding")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        You are a legal context analysis specialist. Extract the most relevant information for the user's question.

        DOCUMENT ANALYSIS:
        {analysis_result.get('analysis', '')[:3000]}

        USER QUESTION: {question}

        RELEVANT DOCUMENT SECTIONS:
        {context[:4000]}

        Please provide:
        1. RELEVANCE SCORE: [High/Medium/Low] - How well can this document answer the question?
        2. KEY INFORMATION: Extract the most pertinent facts/clauses
        3. MISSING ELEMENTS: What information might be needed but not found?
        4. CONFIDENCE LEVEL: [High/Medium/Low] in the available information

        Be precise and highlight the most relevant details for answering the user's question.
        """

        try:
            response = self._safe_generate_content(model, prompt)
            if response and response.text:
                return {
                    "understanding": response.text,
                    "model_used": model_name,
                    "context_length": len(context),
                    "success": True
                }
            else:
                return {
                    "understanding": "Context analysis completed with limited insights.",
                    "model_used": model_name,
                    "success": False
                }
        except Exception as e:
            logging.exception("Context understanding failed")
            return {
                "understanding": "Context partially analyzed.",
                "success": False,
                "error": str(e)
            }

    # FIXED: Enhanced Response Generation with Proper Statistics Tracking
    def generate_enhanced_explanation(self, analysis_result: Dict, context_result: Dict, question: str) -> Generator[str, None, None]:
        """Enhanced explanation generation with FIXED statistics tracking"""
        model_name = self._get_optimal_model("explanation")
        model = genai.GenerativeModel(model_name)

        # Track query start
        self.processing_stats["total_queries"] += 1

        prompt = f"""
        You are a helpful legal document assistant. Provide a clear, structured response to the user's question.

        DOCUMENT ANALYSIS:
        {analysis_result.get('analysis', '')[:2000]}

        CONTEXT UNDERSTANDING:
        {context_result.get('understanding', '')[:2000]}

        USER QUESTION: {question}

        RESPONSE GUIDELINES:
        - Structure your answer with clear headings
        - Use bullet points for lists or multiple items  
        - Explain legal terms in simple language
        - Be specific and cite relevant document sections when possible
        - If information is limited, clearly state what you can/cannot determine
        - Maintain a helpful, professional tone

        Provide a comprehensive yet accessible answer.
        """

        try:
            response_stream = self._safe_generate_content(model, prompt, stream=True)

            if response_stream is None:
                self.processing_stats["failed_responses"] += 1
                yield "⚠️ Unable to generate response due to technical issues. Please try rephrasing your question."
                return

            response_text = ""
            chunk_count = 0
            safety_blocked = False
            recitation_warning = False

            for chunk in response_stream:
                try:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]

                        # Check finish reasons
                        if hasattr(candidate, 'finish_reason'):
                            if candidate.finish_reason == 1:  # SAFETY
                                safety_blocked = True
                                self.processing_stats["blocked_responses"] += 1
                                yield "\n\n🔒 **Response filtered for safety.** Please try asking about general document topics or structure."
                                return
                            elif candidate.finish_reason == 3:  # RECITATION
                                recitation_warning = True
                                # Continue but note the recitation warning

                        # Extract text content
                        text_content = ""
                        if hasattr(chunk, 'text') and chunk.text:
                            text_content = chunk.text
                        elif hasattr(candidate, 'content') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_content += part.text

                        if text_content:
                            response_text += text_content
                            yield text_content
                            chunk_count += 1

                        time.sleep(0.03)  # Smooth streaming

                except Exception as chunk_error:
                    logging.warning(f"Chunk processing error: {chunk_error}")
                    continue

            # FIXED: Proper success tracking
            if response_text.strip():
                if len(response_text.strip()) > 50:  # Meaningful response threshold
                    self.processing_stats["successful_responses"] += 1
                    logging.info(f"✅ SUCCESSFUL response with {chunk_count} chunks, {len(response_text)} chars")
                else:
                    self.processing_stats["partial_responses"] += 1
                    logging.info(f"⚠️ PARTIAL response with {len(response_text)} chars")

                # Add recitation warning if needed
                if recitation_warning:
                    yield "\n\n📋 **Note:** Response may reference copyrighted content. Information provided for analysis purposes only."
            else:
                self.processing_stats["failed_responses"] += 1
                yield "\n\nℹ️ **Limited Response Available.** The document was processed but detailed analysis couldn't be completed. Try asking about specific sections or general document topics."

        except Exception as e:
            logging.exception("Enhanced explanation generation failed")
            self.processing_stats["failed_responses"] += 1
            yield f"❌ **Generation Error:** Unable to create response. Please try a different question or check your document format."

@st.cache_resource
def get_enhanced_processor():
    return EnhancedLegalDocumentProcessor()

