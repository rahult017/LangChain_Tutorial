from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pathlib import Path
import os

BASE_DIR = Path(os.getenv("FAST_BASE_DIR", Path.cwd()))
BASE_DIR.mkdir(parents=True, exist_ok=True)

file_path = BASE_DIR / "chat_history.txt"
file_path.touch(exist_ok=True)

# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("User:"):
            chat_history.append(
                HumanMessage(content=line.replace("User:", "").strip())
            )
        elif line.startswith("Assistant:"):
            chat_history.append(
                AIMessage(content=line.replace("Assistant:", "").strip())
            )
print(prompt)