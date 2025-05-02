import streamlit as st

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory 
from langchain_core.output_parsers import StrOutputParser
import uuid

load_dotenv('./.env')

st.title('HR Chatbot')
st.write('This is a simple HR Chatbot using LLMs. You can ask questions related to HR policies, benefits, and more.')

base_url = "http://localhost:11434"
model = 'llama3.2'

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
 
### LLM Setup
llm = ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")

messages = [
    system,
    MessagesPlaceholder(variable_name="history"),
    human
]

promptTemplate = ChatPromptTemplate(messages=messages)

chain = promptTemplate | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    chain, get_session_history, 
    input_messages_key = 'input',
    history_messages_key = 'history')

def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream(
        { "input": input }, 
        config = { 'configurable' : { "session_id": session_id }}
    ):
        yield output

prompt = st.chat_input("Ask your question here...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("User"):
        st.markdown(prompt)
        
    with st.chat_message("Assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))
         
    st.session_state.chat_history.append({"role": "assistant", "content": response})