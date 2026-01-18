from langchain_huggingface import HuggingFaceEmbeddings

embedded = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text = "Delhi is the capital of India"
result_text = embedded.embed_query(text)
print(str(result_text))

documents =[
    "Delhi is the capital of India",
    "kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
result = embedded.embed_documents(documents)
print(str(result))