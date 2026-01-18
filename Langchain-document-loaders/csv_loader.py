from langchain_community.document_loaders import CSVLoader
from pathlib import Path
import os

BASE_DIR = Path(os.getenv("FAST_BASE_DIR", Path.cwd()))
file_path = BASE_DIR / 'Social_Network_Ads.csv'

loader = CSVLoader(file_path=file_path)

docs = loader.load()

print(len(docs))
print(docs[1])