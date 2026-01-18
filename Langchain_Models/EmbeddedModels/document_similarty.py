import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=300
)
document = [
    "sachin tendulkar is known as the god of cricket for his legendary career",
    "virat kohli is admired for his aggressive batting and strong leadership",
    "rohit sharma is famous for his elegant stroke play and multiple double centuries",
    "jasprit bumrah is one of the best fast bowlers in modern cricket",
    "ms dhoni is respected for his calm captaincy and finishing abilities",
    "rahul dravid is known for his solid defense and dedication to the team"
]
query = input("Ask question about Cricketer:-\n")

doc_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding],doc_embedding)[0]

index , score = sorted(list(enumerate(score)),key=lambda x:x[1])[-1]
print(document[index])
print("Score similarity is :",score)