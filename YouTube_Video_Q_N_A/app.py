import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ------------------------
# Helper Functions
# ------------------------


def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")


def get_youtube_transcript(video_id: str, languages=["en", "hi"]) -> str:
    transcript_list = (
        YouTubeTranscriptApi().list(video_id).find_transcript(languages).fetch()
    )
    return " ".join(chunk.text for chunk in transcript_list)


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.create_documents([text])


def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)


def ask_question(vectorstore, question: str) -> str:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    )

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": question})


# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="centered",
)

st.title("🎥 YouTube Video Q&A (RAG)")
st.write("Ask questions or summarize any YouTube video using RAG.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_url" not in st.session_state:
    st.session_state.video_url = ""

# Section 1: Video Processing
st.header("1️⃣ Process Video")
video_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
    key="url_input",
)

process_button = st.button("Process Video", key="process_btn")

if process_button:
    if not video_url:
        st.error("Please enter a YouTube URL.")
        st.stop()

    try:
        video_id = extract_video_id(video_url)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Fetching transcript..."):
        transcript = get_youtube_transcript(video_id)

    with st.spinner("Chunking & embedding transcript..."):
        docs = chunk_text(transcript)
        st.session_state.vectorstore = build_vectorstore(docs)
        st.session_state.video_processed = True
        st.session_state.video_url = video_url

    st.success("✅ Video processed successfully! You can now ask questions.")

# Section 2: Question Asking (Only shown after video is processed)
if st.session_state.video_processed:
    st.header("2️⃣ Ask Questions")

    # Display the processed video URL
    st.info(
        f"📺 Processed Video: {st.session_state.video_url[:60]}..."
        if len(st.session_state.video_url) > 60
        else f"📺 Processed Video: {st.session_state.video_url}"
    )

    question = st.text_input(
        "Your Question",
        placeholder="What is the main topic of this video?",
        key="question_input",
    )

    if question:
        if st.button("Ask Question", key="ask_btn"):
            with st.spinner("Thinking..."):
                answer = ask_question(
                    st.session_state.vectorstore,
                    question,
                )

            st.markdown("### 💡 Answer")
            st.write(answer)

# Section 3: Quick Actions (Only shown after video is processed)
if st.session_state.video_processed:
    st.divider()
    st.markdown("### ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Summarize Video", key="summary_btn"):
            with st.spinner("Summarizing..."):
                summary = ask_question(
                    st.session_state.vectorstore,
                    "Summarize the entire video in a concise way.",
                )

            st.markdown("### 📝 Summary")
            st.write(summary)

    with col2:
        if st.button("🗑️ Clear & Process New Video", key="clear_btn"):
            # Clear session state
            for key in ["vectorstore", "video_processed", "video_url"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
