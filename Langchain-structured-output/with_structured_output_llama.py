from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.0,
)

# -------- Schema --------
class Review(BaseModel):
    key_themes: list[str]
    summary: str
    sentiment: Literal["pos", "neg"]
    pros: Optional[list[str]] = None
    cons: Optional[list[str]] = None
    name: Optional[str] = None

parser = PydanticOutputParser(pydantic_object=Review)

prompt = PromptTemplate(
    template="""
Extract structured information from the review below.
Return ONLY valid JSON.

{format_instructions}

Review:
{review}
""",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = prompt | llm | StrOutputParser() | parser

result = chain.invoke({
    "review": """I recently upgraded to the Samsung Galaxy S24 Ultra...
Review by Nitish Singh"""
})

print(result)
