from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path
import os

BASE_DIR = Path(os.getenv("FAST_BASE_DIR", Path.cwd()))
file_path = BASE_DIR / 'books'

loader = DirectoryLoader(
    path=file_path,
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)