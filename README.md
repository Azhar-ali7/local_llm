# Local LLM

This project is a PDF chatbot that allows users to upload multiple PDF files and interact with them through a chatbot interface. The application is built using Streamlit, LangChain, and FAISS for creating a Retrieval-Augmented Generation (RAG) system.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Details](#details)
  - [Ollama](#Gemma)
  - [Vector Store](#vector-store)
  - [Embeddings](#embeddings)

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Azhar-ali7/local_llm.git
    ```

2. **Create a virtual environment**

    ```bash
    conda create -n your_virtualenv python==3.10
    conda activate your_virtualenv 
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```
    #### requirements.txt is created using pip freeze > requirements.txt

## Usage

1. **Prepare your PDF files**

    Place your PDF files in the `Data` directory.

2. **Run the Streamlit app**

    ```bash
    streamlit run streamlit.py
    ```

3. **Interact with the Chatbot**

    - Upload your PDF files using the interface.
    - Ask questions and receive responses based on the content of the uploaded PDFs.

## Project Structure

- **Data/**: Directory containing the PDF files to be processed.
- **data_ingestion.py**: Script for ingesting and processing PDF files.
- **embeddings.py**: Script for creating embeddings from the processed text.
- **question_answer.py**: Script for handling the question-answer functionality.
- **streamlit.py**: Streamlit app script for the chatbot interface.
- **README.md**: This README file.
- **requirements.txt**: File containing the list of dependencies.
- **research.ipynb**: Jupyter notebook for research and experimentation.
- **tempDir/**: Temporary directory for processing files. (it will be created by the app)

## Methods

### Data Ingestion

- **data_ingestion.py**: Contains methods to load and preprocess PDF files, split text into manageable chunks, and prepare the data for embeddings.

### Embeddings

- **embeddings.py**: Contains methods to generate embeddings for the text chunks using HuggingFaceEmbeddings and store them in a FAISS vector store.

### Question Answer

- **question_answer.py**: Contains methods to create a retrieval-based QA chain using the embeddings and LangChain’s RetrievalQA functionality.

### Streamlit App

- **streamlit.py**: The main script for running the Streamlit app. It handles the file upload, displays the chat interface, processes the input questions, and displays the responses.

## Details

### Gemma

Gemma2b is an LLM model used in this project. It provides the language model for generating responses to the questions based on the content of the PDF files. (It used because of the computational limitations of my system)

### Vector Store

The vector store is implemented using FAISS (Facebook AI Similarity Search), which is an efficient library for similarity search and clustering of dense vectors. It allows for fast retrieval of the most relevant text chunks based on the user's query.

### Embeddings

Embeddings are generated using HuggingFaceEmbeddings. This method converts text into high-dimensional vectors that capture semantic meaning, enabling effective similarity search and retrieval operations.

---

This project is designed to be a starting point for building more complex and interactive chatbot systems using document-based knowledge.
