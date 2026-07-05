import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from datetime import datetime
import traceback
import sys
from typing import List, Optional, Dict, Any
import os

load_dotenv()

# ------------------------
# Logging Configuration
# ------------------------


def setup_logger():
    """Configure logging with both file and console handlers"""
    # Create logger
    logger = logging.getLogger("YouTubeRAG")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler (detailed logging)
    log_filename = f'youtube_rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler (info and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_filename}")

    return logger, log_filename


# Initialize logger
logger, LOG_FILENAME = setup_logger()

# ------------------------
# Language Configuration
# ------------------------

LANGUAGE_MAP = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Chinese (中文)": "zh",
    "Japanese (日本語)": "ja",
    "Korean (한국어)": "ko",
    "Arabic (العربية)": "ar",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Malayalam (മലയാളം)": "ml",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Urdu (اردو)": "ur",
}

# ------------------------
# Custom Exceptions
# ------------------------


class YouTubeRAGException(Exception):
    """Base exception for YouTube RAG application"""

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, str]:
        return {
            "error": self.message,
            "error_code": self.error_code,
            "timestamp": datetime.now().isoformat(),
        }


class VideoIDExtractionError(YouTubeRAGException):
    """Exception raised when video ID extraction fails"""

    def __init__(self, url: str, message: str = "Failed to extract video ID from URL"):
        self.url = url
        super().__init__(f"{message}: {url}", "INVALID_URL")


class TranscriptFetchError(YouTubeRAGException):
    """Exception raised when transcript fetching fails"""

    def __init__(self, video_id: str, message: str = "Failed to fetch transcript"):
        self.video_id = video_id
        super().__init__(f"{message} for video {video_id}", "TRANSCRIPT_ERROR")


class EmbeddingError(YouTubeRAGException):
    """Exception raised when embedding creation fails"""

    def __init__(self, message: str = "Failed to create embeddings"):
        super().__init__(message, "EMBEDDING_ERROR")


class QueryError(YouTubeRAGException):
    """Exception raised when querying fails"""

    def __init__(self, question: str, message: str = "Failed to process query"):
        self.question = question
        super().__init__(f"{message}: {question[:100]}", "QUERY_ERROR")


class TranslationError(YouTubeRAGException):
    """Exception raised when translation fails"""

    def __init__(self, message: str = "Failed to translate text"):
        super().__init__(message, "TRANSLATION_ERROR")


# ------------------------
# Logger Mixin
# ------------------------


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self):
        """Get logger for the class"""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
        }
        if context:
            error_context.update(context)

        self.logger.error(f"Error in {self.__class__.__name__}: {error_context}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)


# ------------------------
# Translation Service
# ------------------------


class TranslationService(LoggerMixin):
    """Handles text translation using OpenAI"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.output_parser = StrOutputParser()

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text"""
        try:
            self.log_info("Detecting language...")

            prompt = PromptTemplate(
                template="""Detect the language of the following text. Return ONLY the language name (e.g., "English", "Hindi", "Spanish", etc.):

Text: {text}

Language:""",
                input_variables=["text"],
            )

            chain = prompt | self.llm | self.output_parser
            language = chain.invoke({"text": text[:500]})

            self.log_info(f"Detected language: {language}")
            return language.strip()

        except Exception as e:
            self.log_error(e, {"text_length": len(text)})
            return "Unknown"

    def translate_text(
        self, text: str, target_language: str, source_language: str = None
    ) -> str:
        """Translate text to target language"""
        try:
            if not source_language:
                source_language = self.detect_language(text)

            self.log_info(f"Translating from {source_language} to {target_language}")

            # For large texts, translate in chunks
            if len(text) > 4000:
                return self._translate_large_text(
                    text, target_language, source_language
                )
            else:
                return self._translate_chunk(text, target_language, source_language)

        except Exception as e:
            self.log_error(
                e, {"target_language": target_language, "text_length": len(text)}
            )
            raise TranslationError(
                f"Failed to translate text to {target_language}: {str(e)}"
            )

    def _translate_chunk(
        self, text: str, target_language: str, source_language: str
    ) -> str:
        """Translate a single chunk of text"""
        prompt = PromptTemplate(
            template="""Translate the following text from {source_language} to {target_language}. 
Maintain the original meaning, tone, and formatting. Return ONLY the translated text:

Original Text ({source_language}):
{text}

Translated Text ({target_language}):""",
            input_variables=["text", "source_language", "target_language"],
        )

        chain = prompt | self.llm | self.output_parser
        translated = chain.invoke(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            }
        )

        return translated

    def _translate_large_text(
        self, text: str, target_language: str, source_language: str
    ) -> str:
        """Translate large text by splitting into chunks"""
        self.log_info(
            f"Large text detected ({len(text)} chars), splitting into chunks..."
        )

        # Split into chunks of ~3000 characters (to stay within token limits)
        chunk_size = 3000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            self.log_info(f"Translating chunk {i}/{len(chunks)}...")
            translated_chunk = self._translate_chunk(
                chunk, target_language, source_language
            )
            translated_chunks.append(translated_chunk)

        return " ".join(translated_chunks)


# ------------------------
# Processor Classes
# ------------------------


class YouTubeURLParser(LoggerMixin):
    """Parser for YouTube URLs"""

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        context = {"url": url}

        try:
            self.log_info(f"Parsing YouTube URL: {url}")

            if not url:
                raise VideoIDExtractionError(url, "URL is empty")

            video_id = None

            # Method 1: youtube.com/watch?v=VIDEO_ID
            if "v=" in url:
                video_id = url.split("v=")[-1].split("&")[0]
                self.log_debug(f"Extracted video ID using 'v=' pattern: {video_id}")

            # Method 2: youtu.be/VIDEO_ID
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[-1].split("?")[0]
                self.log_debug(
                    f"Extracted video ID using 'youtu.be/' pattern: {video_id}"
                )

            # Validate extracted ID
            if not video_id or len(video_id) != 11:
                raise VideoIDExtractionError(
                    url, f"Invalid video ID format: {video_id}"
                )

            self.log_info(f"Successfully extracted video ID: {video_id}")
            return video_id

        except VideoIDExtractionError:
            raise
        except Exception as e:
            self.log_error(e, context)
            raise VideoIDExtractionError(url, f"Unexpected error: {str(e)}")


class TranscriptFetcher(LoggerMixin):
    """Fetches transcripts from YouTube videos"""

    def __init__(self, languages: List[str] = None):
        self.languages = languages or ["en", "hi"]

    def fetch_transcript(self, video_id: str) -> tuple:
        """Fetch transcript for a YouTube video and return text with detected language"""
        context = {"video_id": video_id, "languages": self.languages}

        try:
            self.log_info(f"Fetching transcript for video {video_id}")

            # Get transcript list
            transcript_list = YouTubeTranscriptApi().list(video_id)
            self.log_debug(
                f"Available transcripts: {[t.language_code for t in transcript_list]}"
            )

            # Find transcript in preferred languages
            try:
                transcript = transcript_list.find_transcript(self.languages)
                self.log_info(f"Found transcript in {transcript.language_code}")
            except Exception as e:
                self.log_warning(
                    f"Could not find transcript in {self.languages}, fetching first available"
                )
                available_langs = [t.language_code for t in transcript_list]
                transcript = transcript_list.find_transcript(available_langs)

            # Fetch and combine transcript
            transcript_parts = transcript.fetch()
            full_transcript = " ".join(chunk.text for chunk in transcript_parts)

            # Get language name from language code
            language_code = transcript.language_code
            language_name = self._get_language_name(language_code)

            self.log_info(
                f"Successfully fetched transcript. Language: {language_name} ({language_code}), Length: {len(full_transcript)} characters"
            )

            return full_transcript, language_name

        except TranscriptFetchError:
            raise
        except Exception as e:
            self.log_error(e, context)
            raise TranscriptFetchError(
                video_id, f"Failed to fetch transcript: {str(e)}"
            )

    def _get_language_name(self, language_code: str) -> str:
        """Convert language code to language name"""
        for name, code in LANGUAGE_MAP.items():
            if code == language_code:
                return name.split(" ")[0]  # Remove emoji/script part
        return language_code.upper()


class TextChunker(LoggerMixin):
    """Chunks text into smaller pieces for processing"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(self, text: str) -> List[Any]:
        """Split text into chunks"""
        context = {"text_length": len(text), "chunk_size": self.chunk_size}

        try:
            self.log_info(f"Chunking text of length {len(text)} characters")
            docs = self.splitter.create_documents([text])
            self.log_info(f"Created {len(docs)} chunks")
            return docs

        except Exception as e:
            self.log_error(e, context)
            raise YouTubeRAGException(
                f"Failed to chunk text: {str(e)}", "CHUNKING_ERROR"
            )


class VectorStoreBuilder(LoggerMixin):
    """Builds and manages vector store"""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def build_vectorstore(self, docs: List[Any]) -> FAISS:
        """Build vector store from documents"""
        context = {"num_documents": len(docs), "embedding_model": self.embedding_model}

        try:
            self.log_info(f"Building vector store from {len(docs)} documents")
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.log_info("Vector store created successfully")
            return vectorstore

        except Exception as e:
            self.log_error(e, context)
            raise EmbeddingError(f"Failed to build vector store: {str(e)}")


class QuestionAnswerer(LoggerMixin):
    """Handles question answering using RAG"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.output_parser = StrOutputParser()
        self.translation_service = TranslationService()

    def answer_question(
        self,
        vectorstore: FAISS,
        question: str,
        response_language: str = "English",
        k: int = 4,
    ) -> str:
        """Answer a question using RAG with language support"""
        context = {
            "question": question,
            "model": self.model,
            "k": k,
            "response_language": response_language,
        }

        try:
            self.log_info(
                f"Processing question in {response_language}: {question[:100]}..."
            )

            if vectorstore is None:
                raise QueryError(question, "Vector store is not initialized")

            # Translate question to English for better retrieval if needed
            search_question = question
            if response_language != "English":
                self.log_info(f"Translating question to English for better search...")
                search_question = self.translation_service.translate_text(
                    question, "English", response_language
                )

            # Set up retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )

            # Create prompt template
            prompt = PromptTemplate(
                template="""You are a helpful assistant. Answer the question based ONLY on the provided transcript context. 
If the context is insufficient, just say you don't know.

IMPORTANT: Provide your answer in {response_language}.

Context:
{context}

Question:
{question}

Answer in {response_language}:""",
                input_variables=["context", "question", "response_language"],
            )

            # Retrieve relevant documents
            self.log_debug("Retrieving relevant documents...")
            docs = retriever.invoke(search_question)
            self.log_debug(f"Retrieved {len(docs)} documents")

            if not docs:
                return "I couldn't find relevant information in the video transcript to answer your question."

            # Combine context
            context_text = "\n\n".join(doc.page_content for doc in docs)
            self.log_debug(f"Context length: {len(context_text)} characters")

            # Create chain and get response
            self.log_debug("Invoking LLM chain...")
            chain = prompt | self.llm | self.output_parser
            response = chain.invoke(
                {
                    "context": context_text,
                    "question": question,
                    "response_language": response_language,
                }
            )

            self.log_info(
                f"Successfully generated response. Length: {len(response)} characters"
            )
            return response

        except Exception as e:
            self.log_error(e, context)
            raise QueryError(question, f"Failed to answer question: {str(e)}")


class YouTubeRAGPipeline(LoggerMixin):
    """Main pipeline orchestrator"""

    def __init__(self):
        self.log_info("Initializing YouTubeRAGPipeline")
        self.url_parser = YouTubeURLParser()
        self.transcript_fetcher = TranscriptFetcher()
        self.text_chunker = TextChunker()
        self.vector_store_builder = VectorStoreBuilder()
        self.qa_system = QuestionAnswerer()
        self.translation_service = TranslationService()

    def process_video(self, url: str, target_language: str = "English") -> tuple:
        """Complete pipeline to process a YouTube video with language support"""
        pipeline_context = {
            "url": url,
            "target_language": target_language,
            "pipeline_start": datetime.now().isoformat(),
        }

        try:
            self.log_info(
                f"Starting video processing pipeline for: {url} (Target language: {target_language})"
            )

            # Step 1: Parse URL
            self.log_info("Step 1/6: Parsing URL")
            video_id = self.url_parser.extract_video_id(url)

            # Step 2: Fetch transcript with language detection
            self.log_info("Step 2/6: Fetching transcript")
            transcript, detected_language = self.transcript_fetcher.fetch_transcript(
                video_id
            )

            # Step 3: Translate transcript if needed
            self.log_info("Step 3/6: Processing language")
            if (
                target_language != "Auto-detect"
                and detected_language != target_language
            ):
                self.log_info(
                    f"Translating transcript from {detected_language} to {target_language}"
                )
                transcript = self.translation_service.translate_text(
                    transcript, target_language, detected_language
                )
            else:
                target_language = detected_language
                self.log_info(f"Using original language: {detected_language}")

            # Step 4: Chunk text
            self.log_info("Step 4/6: Chunking text")
            docs = self.text_chunker.chunk_text(transcript)

            # Step 5: Build vector store
            self.log_info("Step 5/6: Building vector store")
            vectorstore = self.vector_store_builder.build_vectorstore(docs)

            # Step 6: Generate summary
            self.log_info("Step 6/6: Generating summary")
            summary = self.qa_system.answer_question(
                vectorstore,
                "Provide a comprehensive summary of this video. Include the main topic, key points discussed, and important conclusions.",
                response_language=target_language,
            )

            self.log_info(f"Pipeline completed successfully for video {video_id}")
            return vectorstore, summary, detected_language, target_language

        except YouTubeRAGException as e:
            self.log_error(e, pipeline_context)
            raise
        except Exception as e:
            self.log_error(e, pipeline_context)
            raise YouTubeRAGException(f"Pipeline failed unexpectedly: {str(e)}")


# ------------------------
# Error Handler Decorator
# ------------------------


def handle_streamlit_errors(func):
    """Decorator to handle errors in Streamlit functions"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except YouTubeRAGException as e:
            error_dict = e.to_dict()
            logger.error(f"Application error: {error_dict}")
            st.error(f"❌ {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            st.error(f"❌ An unexpected error occurred: {str(e)}")

    return wrapper


# ------------------------
# Streamlit UI
# ------------------------


# Initialize pipeline
@st.cache_resource
def get_pipeline():
    """Initialize and cache the YouTube RAG pipeline"""
    logger.info("Initializing YouTubeRAGPipeline")
    return YouTubeRAGPipeline()


# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY in your .env file."
    )
    st.stop()

try:
    pipeline = get_pipeline()
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    st.error(f"Failed to initialize application: {str(e)}")
    st.stop()

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="centered",
)

st.title("🎥 YouTube Video Q&A (RAG)")
st.write(
    "Paste a YouTube URL to automatically summarize and ask questions in any language!"
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_processed_url" not in st.session_state:
    st.session_state.last_processed_url = ""
if "detected_language" not in st.session_state:
    st.session_state.detected_language = ""
if "target_language" not in st.session_state:
    st.session_state.target_language = "English"
if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

# YouTube URL Input
video_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
    key="url_input",
)

# Language Selection
col1, col2 = st.columns(2)
with col1:
    target_language = st.selectbox(
        "🌐 Translate transcript to:",
        ["Auto-detect"] + list(LANGUAGE_MAP.keys()),
        key="target_language_select",
        help="Select the language to translate the transcript. 'Auto-detect' will keep the original language.",
    )
with col2:
    response_language = st.selectbox(
        "💬 Response language:",
        list(LANGUAGE_MAP.keys()),
        index=0,  # Default to English
        key="response_language_select",
        help="Select the language for answers and summaries.",
    )


# Auto-process when URL changes
@handle_streamlit_errors
def process_video_url(video_url, target_language, response_language):
    """Process video URL and update session state"""
    if video_url and video_url != st.session_state.last_processed_url:
        if not st.session_state.processing:
            st.session_state.processing = True

            # Create placeholders for progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Process video
                status_text.text("🔍 Starting video processing...")
                progress_bar.progress(10)

                logger.info(f"Processing video URL: {video_url}")

                # Step by step with progress updates
                status_text.text("🔍 Fetching transcript...")
                progress_bar.progress(25)

                status_text.text("🌐 Detecting language...")
                progress_bar.progress(40)

                status_text.text("📝 Processing transcript...")
                progress_bar.progress(55)

                status_text.text("🧠 Building AI knowledge base...")
                progress_bar.progress(70)

                status_text.text("📊 Generating summary...")
                progress_bar.progress(85)

                vectorstore, summary, detected_language, final_language = (
                    pipeline.process_video(video_url, target_language)
                )

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Store results in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.summary = summary
                st.session_state.video_processed = True
                st.session_state.video_url = video_url
                st.session_state.last_processed_url = video_url
                st.session_state.detected_language = detected_language
                st.session_state.target_language = final_language
                st.session_state.response_language = response_language
                st.session_state.processing = False

                # Clear progress indicators
                status_text.empty()
                progress_bar.empty()

                logger.info("Video processing completed successfully")
                st.rerun()

            except Exception as e:
                st.session_state.processing = False
                status_text.empty()
                progress_bar.empty()
                logger.error(f"Failed to process video: {str(e)}")
                raise


if video_url:
    process_video_url(video_url, target_language, response_language)

# Display Summary if video is processed
if st.session_state.video_processed and st.session_state.summary:
    st.divider()

    # Language Information
    st.info(
        f"📹 Original language: **{st.session_state.detected_language}** | Processed in: **{st.session_state.target_language}** | Responses in: **{st.session_state.response_language}**"
    )

    # Summary Section
    with st.expander("📝 Video Summary", expanded=True):
        st.write(st.session_state.summary)

        # Quick action buttons for the summary
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Regenerate Summary", key="regen_summary"):
                with st.spinner("Regenerating summary..."):
                    try:
                        logger.info("Regenerating summary")
                        if st.session_state.vectorstore is None:
                            st.error(
                                "Vector store not found. Please process the video again."
                            )
                        else:
                            st.session_state.summary = pipeline.qa_system.answer_question(
                                st.session_state.vectorstore,
                                "Provide a comprehensive summary of this video. Include the main topic, key points discussed, and important conclusions.",
                                response_language=st.session_state.response_language,
                            )
                            logger.info("Summary regenerated successfully")
                    except Exception as e:
                        logger.error(f"Failed to regenerate summary: {str(e)}")
                        st.error(f"Failed to regenerate summary: {str(e)}")
                st.rerun()

        with col2:
            if st.button("🎯 Key Takeaways", key="key_takeaways"):
                with st.spinner("Extracting key takeaways..."):
                    try:
                        logger.info("Extracting key takeaways")
                        if st.session_state.vectorstore is None:
                            st.error(
                                "Vector store not found. Please process the video again."
                            )
                        else:
                            takeaways = pipeline.qa_system.answer_question(
                                st.session_state.vectorstore,
                                "List the top 5 key takeaways from this video in bullet points.",
                                response_language=st.session_state.response_language,
                            )
                            st.info(f"### 🎯 Key Takeaways\n{takeaways}")
                    except Exception as e:
                        logger.error(f"Failed to extract takeaways: {str(e)}")
                        st.error(f"Failed to extract takeaways: {str(e)}")

        with col3:
            if st.button("⏱️ Timeline Summary", key="timeline"):
                with st.spinner("Creating timeline..."):
                    try:
                        logger.info("Creating timeline summary")
                        if st.session_state.vectorstore is None:
                            st.error(
                                "Vector store not found. Please process the video again."
                            )
                        else:
                            timeline = pipeline.qa_system.answer_question(
                                st.session_state.vectorstore,
                                "Create a chronological timeline of the main events or topics discussed in this video.",
                                response_language=st.session_state.response_language,
                            )
                            st.info(f"### ⏱️ Timeline\n{timeline}")
                    except Exception as e:
                        logger.error(f"Failed to create timeline: {str(e)}")
                        st.error(f"Failed to create timeline: {str(e)}")

    # Question & Prompt Section
    st.divider()
    st.header("💬 Ask Questions or Write a Prompt")
    st.write("Ask anything about the video or provide a custom prompt for analysis!")

    # Input method selection
    input_type = st.radio(
        "Choose input type:",
        ["❓ Specific Question", "📝 Custom Prompt"],
        horizontal=True,
        key="input_type",
    )

    if input_type == "❓ Specific Question":
        user_input = st.text_area(
            "Your Question",
            placeholder="What is the main argument presented in this video?",
            height=100,
            key="question_area",
            help=f"Ask your question in any language. The response will be in {st.session_state.response_language}.",
        )

        if st.button("🔍 Get Answer", key="answer_btn", type="primary"):
            if not user_input:
                st.warning("Please enter a question.")
            elif st.session_state.vectorstore is None:
                st.error("Please process a video first before asking questions.")
            else:
                with st.spinner("Analyzing video content..."):
                    try:
                        logger.info(f"Answering question: {user_input[:100]}")
                        answer = pipeline.qa_system.answer_question(
                            st.session_state.vectorstore,
                            user_input,
                            response_language=st.session_state.response_language,
                        )
                        st.markdown("### 💡 Answer")
                        st.write(answer)
                        logger.info("Question answered successfully")
                    except Exception as e:
                        logger.error(f"Failed to get answer: {str(e)}")
                        st.error(f"Failed to get answer: {str(e)}")

    else:  # Custom Prompt
        # Prompt templates
        st.caption("Quick prompt templates:")
        prompt_templates = {
            "Detailed Analysis": "Analyze this video in depth. Discuss the main arguments, supporting evidence, and overall effectiveness of the presentation.",
            "Technical Explanation": "Explain the technical concepts discussed in this video in simple terms.",
            "Counter Arguments": "What are potential counter-arguments to the main points presented in this video?",
            "Expert Review": "Review this content as an expert in the field. What's accurate, what's missing, and what could be improved?",
            "Custom": "",
        }

        selected_template = st.selectbox(
            "Choose a prompt template (optional):",
            list(prompt_templates.keys()),
            key="prompt_template",
        )

        custom_prompt = st.text_area(
            "Your Custom Prompt",
            value=prompt_templates[selected_template],
            placeholder="Write your custom prompt here...",
            height=120,
            key="prompt_area",
            help=f"Write your prompt in any language. The response will be in {st.session_state.response_language}.",
        )

        if st.button("✨ Generate Response", key="prompt_btn", type="primary"):
            if not custom_prompt:
                st.warning("Please enter a prompt.")
            elif st.session_state.vectorstore is None:
                st.error("Please process a video first before using prompts.")
            else:
                with st.spinner("Processing your prompt..."):
                    try:
                        logger.info(f"Processing custom prompt: {custom_prompt[:100]}")
                        response = pipeline.qa_system.answer_question(
                            st.session_state.vectorstore,
                            custom_prompt,
                            response_language=st.session_state.response_language,
                        )
                        st.markdown("### 🎯 Response")
                        st.write(response)
                        logger.info("Prompt processed successfully")
                    except Exception as e:
                        logger.error(f"Failed to process prompt: {str(e)}")
                        st.error(f"Failed to process prompt: {str(e)}")

    # Additional Features Section
    st.divider()
    st.header("🔧 Additional Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📚 Key Topics", key="topics"):
            with st.spinner("Identifying key topics..."):
                try:
                    logger.info("Identifying key topics")
                    if st.session_state.vectorstore is None:
                        st.error(
                            "Vector store not found. Please process the video again."
                        )
                    else:
                        topics = pipeline.qa_system.answer_question(
                            st.session_state.vectorstore,
                            "List the main topics and subtopics discussed in this video. Group them hierarchically.",
                            response_language=st.session_state.response_language,
                        )
                        st.info(f"### 📚 Key Topics\n{topics}")
                except Exception as e:
                    logger.error(f"Failed to identify topics: {str(e)}")
                    st.error(f"Failed to identify topics: {str(e)}")

    with col2:
        if st.button("💭 Quotes", key="quotes"):
            with st.spinner("Extracting important quotes..."):
                try:
                    logger.info("Extracting quotes")
                    if st.session_state.vectorstore is None:
                        st.error(
                            "Vector store not found. Please process the video again."
                        )
                    else:
                        quotes = pipeline.qa_system.answer_question(
                            st.session_state.vectorstore,
                            "Extract 5-7 notable quotes or statements from this video that capture its essence.",
                            response_language=st.session_state.response_language,
                        )
                        st.info(f"### 💭 Notable Quotes\n{quotes}")
                except Exception as e:
                    logger.error(f"Failed to extract quotes: {str(e)}")
                    st.error(f"Failed to extract quotes: {str(e)}")

    with col3:
        if st.button("🗑️ Process New Video", key="clear_btn"):
            logger.info("Clearing session state for new video")
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

elif st.session_state.video_processed and not st.session_state.summary:
    st.info("Processing video... Please wait.")

elif not st.session_state.video_processed and not video_url:
    st.info("👆 Enter a YouTube URL above to get started!")

# Footer
st.divider()
st.caption(
    "Built with Streamlit, LangChain, and OpenAI • Supports multiple languages • Automatically processes YouTube videos on URL input"
)
