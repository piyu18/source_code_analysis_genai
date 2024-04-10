from src.helper import *
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

documents = load_repos('repo/')
text_chunks = text_splitter(documents)
embeddings = load_embedding_model()

# store vector in chromadb
vectordb = Chroma.from_documents(text_chunks, embeddings,persist_directory='./db')
vectordb.persist()
