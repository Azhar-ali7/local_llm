import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO

def save_uploadedfile(uploadedfile):
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")
    file_path = os.path.join("tempDir", uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def process_files(files):
    documents = []
    for file in files:
        file_path = save_uploadedfile(file)
        loader = UnstructuredFileLoader(file_path)
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
