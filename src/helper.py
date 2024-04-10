import os
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


# clone any git repositories
def repo_clone(repo_url):
    os.makedirs('repos',exist_ok=True)
    repo_path = 'repo/'
    Repo.clone_from(repo_url, to_path=repo_path)


# loading repositories as documents
def load_repos(repo_path):
    loader = GenericLoader.from_filesystem(repo_path+'app',
                       glob = '**/*',
                       suffixes = ['.py'],
                       parser=LanguageParser(language=Language.PYTHON,parser_threshold=500))
    
    documents = loader.load()
    return documents

# Creating chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                  chunk_size=2000,
                                                                  chunk_overlap=500)
    
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks

# load embedding model

def load_embedding_model():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings