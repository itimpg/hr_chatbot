# HR Chatbot using LangChain and Streamlit

This project implements a simple HR chatbot that leverages LangChain, Streamlit, and Ollama to answer HR-related questions. The chatbot retrieves relevant information from a vector database (FAISS) and provides concise answers using a pre-trained LLM model.

## Features
- **Question-Answering**: Answers questions related to HR policies, benefits, and more.
- **Chat History**: Maintains user chat history for a personalized experience.
- **Contextual Responses**: Uses context from a vector store to provide relevant answers.
- **Local Model Integration**: Uses Ollama's LLM to process and respond to queries.
- **Streamlit Interface**: Simple and interactive UI for chatting with the HR assistant.

## Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Create a .env file with your environment variables, including the base URL for Ollama.
4. Prepare PDF files in ./rag-dataset
5. Run the script to generate Vector Database
    ```bash
    python update_vector_store.py

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
2. Enter your user ID and start asking questions related to HR.

## Technology Stack
- **LangChain**: Framework for building LLM applications.
- **Streamlit**: Web framework for creating interactive applications.
- **Ollama**: LLM service for natural language processing.relevant answers.
- **FAISS**: Vector store for fast similarity searches.
- **SQLite**: Database for storing chat history.