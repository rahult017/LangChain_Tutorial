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
st.write("Paste a YouTube URL to automatically summarize and ask questions!")

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

# YouTube URL Input
video_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
    key="url_input",
)

# Auto-process when URL changes
if video_url and video_url != st.session_state.last_processed_url:
    if not st.session_state.processing:
        st.session_state.processing = True

        try:
            video_id = extract_video_id(video_url)
        except ValueError as e:
            st.error(str(e))
            st.session_state.processing = False
            st.stop()

        # Create placeholders for progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Fetch transcript
        status_text.text("🔍 Fetching transcript...")
        progress_bar.progress(20)

        try:
            transcript = get_youtube_transcript(video_id)
            progress_bar.progress(40)
        except Exception as e:
            st.error(f"Failed to fetch transcript: {str(e)}")
            st.session_state.processing = False
            st.stop()

        # Step 2: Chunking
        status_text.text("📝 Chunking transcript...")
        docs = chunk_text(transcript)
        progress_bar.progress(60)

        # Step 3: Building embeddings
        status_text.text("🧠 Building embeddings...")
        st.session_state.vectorstore = build_vectorstore(docs)
        progress_bar.progress(80)

        # Step 4: Generate initial summary
        status_text.text("📊 Generating summary...")
        summary = ask_question(
            st.session_state.vectorstore,
            "Provide a comprehensive summary of this video. Include the main topic, key points discussed, and important conclusions.",
        )
        progress_bar.progress(100)

        # Store results in session state
        st.session_state.summary = summary
        st.session_state.video_processed = True
        st.session_state.video_url = video_url
        st.session_state.last_processed_url = video_url
        st.session_state.processing = False

        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()

        st.rerun()

# Display Summary if video is processed
if st.session_state.video_processed and st.session_state.summary:
    st.divider()

    # Summary Section
    with st.expander("📝 Video Summary", expanded=True):
        st.write(st.session_state.summary)

        # Quick action buttons for the summary
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Regenerate Summary", key="regen_summary"):
                with st.spinner("Regenerating summary..."):
                    st.session_state.summary = ask_question(
                        st.session_state.vectorstore,
                        "Provide a comprehensive summary of this video. Include the main topic, key points discussed, and important conclusions.",
                    )
                st.rerun()

        with col2:
            if st.button("🎯 Key Takeaways", key="key_takeaways"):
                with st.spinner("Extracting key takeaways..."):
                    takeaways = ask_question(
                        st.session_state.vectorstore,
                        "List the top 5 key takeaways from this video in bullet points.",
                    )
                st.info(f"### 🎯 Key Takeaways\n{takeaways}")

        with col3:
            if st.button("⏱️ Timeline Summary", key="timeline"):
                with st.spinner("Creating timeline..."):
                    timeline = ask_question(
                        st.session_state.vectorstore,
                        "Create a chronological timeline of the main events or topics discussed in this video.",
                    )
                st.info(f"### ⏱️ Timeline\n{timeline}")

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
        )

        if st.button("🔍 Get Answer", key="answer_btn", type="primary") and user_input:
            with st.spinner("Analyzing video content..."):
                answer = ask_question(st.session_state.vectorstore, user_input)

            st.markdown("### 💡 Answer")
            st.write(answer)

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
        )

        if (
            st.button("✨ Generate Response", key="prompt_btn", type="primary")
            and custom_prompt
        ):
            with st.spinner("Processing your prompt..."):
                response = ask_question(st.session_state.vectorstore, custom_prompt)

            st.markdown("### 🎯 Response")
            st.write(response)

    # Additional Features Section
    st.divider()
    st.header("🔧 Additional Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📚 Key Topics", key="topics"):
            with st.spinner("Identifying key topics..."):
                topics = ask_question(
                    st.session_state.vectorstore,
                    "List the main topics and subtopics discussed in this video. Group them hierarchically.",
                )
            st.info(f"### 📚 Key Topics\n{topics}")

    with col2:
        if st.button("💭 Quotes", key="quotes"):
            with st.spinner("Extracting important quotes..."):
                quotes = ask_question(
                    st.session_state.vectorstore,
                    "Extract 5-7 notable quotes or statements from this video that capture its essence.",
                )
            st.info(f"### 💭 Notable Quotes\n{quotes}")

    with col3:
        if st.button("🗑️ Process New Video", key="clear_btn"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

elif st.session_state.video_processed and not st.session_state.summary:
    # Handle case where video is processed but no summary (shouldn't normally happen)
    st.info("Processing video... Please wait.")

elif not st.session_state.video_processed and video_url:
    st.info("👆 Press Enter or click outside the input field to process the video.")

# Footer
st.divider()
st.caption(
    "Built with Streamlit, LangChain, and OpenAI • Automatically processes YouTube videos on URL input"
)
