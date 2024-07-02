from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS, chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings()

def create_embeddings(documents):

    # embeddings = OllamaEmbeddings(model="gemma:2b-instruct-q4_0")
    knowledge_base = FAISS.from_documents(documents, embeddings)
    # knowledge_base = chroma.from_documents(documents, embeddings)

    return knowledge_base

def save_embeddings(knowledge_base, file_path):
    # save to disk
    # chroma.from_documents(knowledge_base, embeddings, persist_directory=file_path)
    knowledge_base.save_local(file_path)

def load_embeddings(file_path, ):
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    # return chroma(persist_directory=file_path, embedding_function=embeddings)
