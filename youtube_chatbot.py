from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def get_youtube_transcript_text(
    video_id: str,
    languages: list = ["en"],
    preserve_formatting: bool = False,
):
    try:
        Transcript_list_output = (
            YouTubeTranscriptApi().list(video_id).find_transcript(languages).fetch()
        )
        # Flatten it to plain text
        transcript = " ".join(chunk.text for chunk in Transcript_list_output)
        # transcript = [
        #     {s
        #         "text": chunk.text,
        #         "start": chunk.start,
        #         "duration": chunk.duration,
        #         }
        #         for chunk in Transcript_list_output
        # ]

        return transcript
    except Exception as e:
        return e


def covert_text_to_chunk(
    transcript: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents([transcript])
    return chunks


def convert_chunks_to_embeddings(chunks, model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_ids = vector_store.index_to_docstore_id
    vector_store.get_by_ids(["2436bdb8-3f5f-49c6-8915-0c654c888700"])
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    query = retriever.invoke("What is deepmind")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
        input_variables=["context", "question"],
    )
    question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)
    return answer


if __name__ == "__main__":
    video_id = "Gfr50f6ZBvo"
    languages = ["en"]
    result = get_youtube_transcript_text(
        video_id=video_id,
        languages=languages,
        preserve_formatting=True,
    )
    chunk = covert_text_to_chunk(transcript=result)
    embeddings = convert_chunks_to_embeddings(chunks=chunk)
    # print(embeddings)


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chunk = covert_text_to_chunk(transcript=result)
vector_store = FAISS.from_documents(chunk, embeddings)
vector_ids = vector_store.index_to_docstore_id
vector_store.get_by_ids(["2436bdb8-3f5f-49c6-8915-0c654c888700"])
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)
parser = StrOutputParser()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)
main_chain = parallel_chain | prompt | llm | parser
result = main_chain.invoke("Can you summarize the video")
print(result)
