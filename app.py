import os
import streamlit as st
import warnings

from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')
load_dotenv()
base_url = "http://localhost:11434"
model = 'llama3.2'

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
db_name = r".\hr_info"
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context and history to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {input} 
Context: {context} 
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(prompt)

llm = ChatOllama(model=model, base_url=base_url)

def format_docs(docs):
    return "\n\n".join([f"{doc.metadata.get('source', '')}:\n{doc.page_content}" for doc in docs])

rag_chain = (
    RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: format_docs(retriever.invoke(x["input"])),
        "history": lambda x: "\n".join([msg.content for msg in x["history"]])  # Use only the content attribute
    })
    | prompt_template
    | llm
    | StrOutputParser()
)

st.title('HR Chatbot')
st.write('This is a simple HR Chatbot using LLMs. You can ask questions related to HR policies, benefits, and more.')
 
user_id = st.text_input("Enter your user ID", value="Manao Employee")

def get_session_history(session_id):
    return SQLChatMessageHistory(
        session_id,
        connection_string="sqlite:///chat_history.db"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()
 
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
runnable_with_history = RunnableWithMessageHistory(
    rag_chain, 
    get_session_history, 
    input_messages_key = 'input',
    history_messages_key = 'history')

def chat_with_llm(session_id, user_input):
    for chunk in runnable_with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk

prompt = st.chat_input("Ask your question here...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.write_stream(chat_with_llm(user_id, prompt))
        st.session_state.chat_history.append({"role": "assistant", "content": response})