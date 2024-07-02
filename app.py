import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO

def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("tempDir", uploadedfile.name)

def process_pdf(files):
    documents = []
    for file in files:
        file_path = save_uploadedfile(file)
        loader = UnstructuredFileLoader(file_path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    llm = Ollama(model="gemma:2b-instruct-q4_0", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        retriever=knowledge_base.as_retriever(),
        llm=llm
    )
    return qa_chain

def ask_question(qa_chain, question):
    response = qa_chain.invoke({"query": question})
    return response['result']

# Streamlit interface
st.title("PDF Chatbot")

if not os.path.exists("tempDir"):
    os.makedirs("tempDir")

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.write("Processing files...")
    qa_chain = process_pdf(uploaded_files)
    st.write("Files processed. You can now ask questions.")
    
    question = st.text_input("Ask a question:")
    if question:
        response = ask_question(qa_chain, question)
        st.write(response)

# Clean up the temporary files after processing
import shutil
shutil.rmtree("tempDir")
