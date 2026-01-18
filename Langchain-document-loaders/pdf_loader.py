from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import os

BASE_DIR = Path(os.getenv("FAST_BASE_DIR", Path.cwd()))
file_path = BASE_DIR / 'dl-curriculum.pdf'

loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)