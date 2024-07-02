import streamlit as st
from streamlit_chat import message
import os
from data_ingestion import process_files, split_documents
from embeddings import create_embeddings, save_embeddings, load_embeddings
from question_answer import create_qa_chain, ask_question

st.title("PDF Chatbot")

# Initialize session state
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = process_files(uploaded_files)
    text_chunks = split_documents(documents)
    
    embeddings_path = "tempDir/embeddings.faiss"
    if os.path.exists(embeddings_path):
        st.session_state.knowledge_base = load_embeddings(embeddings_path)
    else:
        st.session_state.knowledge_base = create_embeddings(text_chunks)
        save_embeddings(st.session_state.knowledge_base, embeddings_path)
    
    st.write("Files processed. You can now ask questions.")

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    if chat["role"] == "user":
        message(chat["content"], is_user=True, key=f"user_{i}")
    else:
        message(chat["content"], key=f"assistant_{i}")

# Chat input form at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    prompt = st.text_input("Ask me anything about the uploaded documents:")
    submit_button = st.form_submit_button("Send")

if submit_button and prompt:
    if st.session_state.knowledge_base:
        qa_chain = create_qa_chain(st.session_state.knowledge_base)
        response = ask_question(qa_chain, prompt)
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun the app to display the new messages
        st.rerun()
    else:
        st.write("Please upload PDF files first.")
