from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
llm_obj = OpenAI(model="gpt-3.5-turbo-instruct")
output_resukt = llm_obj.invoke("who is the current prime minster of india.")
print(output_resukt)