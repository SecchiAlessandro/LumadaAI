import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from LumadaSapiens import LumadaAI

os.environ['USER_AGENT'] = 'chatbot'


load_dotenv()

# def get_response(user_query):
#     return "I don't know"

# app config
st.set_page_config(page_title="Chat with LumadaAI", page_icon="🤖")
st.title("Chat with LumadaAI")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am LumadaAI. How can I help you?"),
    ]

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:


    user_query = st.text_input("You: ", "")
    print("chatbot", user_query)


    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        print("chatbot", user_query)
        # Process the input message
        #response = get_response(user_query)
        response = LumadaAI(user_query)
        AIresponse = str(response)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=AIresponse))
        
    
        

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)