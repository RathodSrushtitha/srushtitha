import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import google.generativeai as genai
import os
from groq import Groq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv()
import streamlit as st
from typing import Generator
from groq import Groq

# Set the page configuration
st.set_page_config(
    page_title="InsightPDF",  # Title of the web page with smiley
    layout="centered"  # Layout of the page
)

# CSS to hide the footer and GitHub logo
hide_streamlit_style = """    
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

client = Groq(
    api_key=os.environ['GROQ_API_KEY'],
)

retriever_prompt = (
    "You are a chatbot assistant, here to answer questions about pdf. Use the chat history and the user’s latest question to give clear, helpful, and polite answers based on the conversation. If the question is not related to pdf, respond politely and let the user know that you can only help with pdf-related topics. (Do not answer unrelated questions in any case.)."
)





 
 
 
# Streamlit application layout
st.title("InsightPDF")
st.markdown("<hr style='border: 1px solid black; margin-top: -10px;'>", unsafe_allow_html=True) 



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_option" not in st.session_state:
    st.session_state.current_option = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = "llama-3.1-70b-versatile"

max_tokens_range =  8000

# Dropdown menu for selecting input type
option = st.sidebar.selectbox("Select Input Type", ("Please select an option","PDF"))


if "uploaded_file" not in st.session_state:
     st.session_state.uploaded_file = None
# Initialize session state for the file uploader
if option=="Please select an option":
   

    st.markdown("""
 

This system enables users to upload PDFs and receive answers based on the content. It uses advanced AI to process files and provide interactive, accurate responses to your queries.

Features:
PDF Upload: Extracts and analyzes information from any uploaded PDF file.
Interactive Chat: Ask questions about the uploaded files and get precise, context-aware answers.
Easily reset the system with the 'Clear' button to start fresh
""")


 
 
 



# Clear chat history if input type has changed
if option != st.session_state.current_option:
    st.session_state.messages = []
    st.session_state.current_option = option


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store chat history


# Define helper functions
def create_embeddings_from_pdf(file_path):
    """Load a PDF, split it into chunks, and create embeddings. Returns the vector store."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def answer_question_from_vectorstore(vectorstore, question):
    # """Answer a question using the provided vector store."""
    # prompt = ChatPromptTemplate.from_template(""" 
    #     You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    #     Question: {question} 
    #     Context: {context} 
    #     Answer:
    # """)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    llm = ChatGroq(temperature=0.1, model_name="llama-3.1-70b-versatile", max_tokens=8000)

    contextualize_q_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),


        ]
)

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    
    system_prompt = (
        "You are an intelligent assistant designed to answer questions specifically about the content of uploaded PDFs. Use the chat history and the user’s latest question to provide clear, accurate, and polite responses based solely on the information contained within the uploaded document. If the question is unrelated to the PDF content, politely inform the user that you can only assist with questions based on the uploaded file. Refrain from addressing unrelated queries under any circumstances..)"
        "\n\n"
        "{context}"
)
  
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    chat_history = []
    message= rag_chain.invoke({"input": question, "chat_history":  st.session_state.chat_history})
    return message["answer"]



 


# Conditional display based on selection
if option == "PDF":
        if option=="PDF":
            if "transcription" in st.session_state:
                st.session_state.pop("transcription")
            if "sample_file" in st.session_state:
                st.session_state.pop("sample_file")

            
        # PDF file uploader
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is None:
            st.sidebar.warning("Please upload a PDF file to proceed.")
        else:
            st.sidebar.info("Please click the '❌' to remove the file before clicking 'Clear' to reset the chat.") 

        if uploaded_file and "vectorstore" not in st.session_state:
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            with st.spinner("Please wait, we are processing your report..."):
                # Create embeddings from the PDF
                st.session_state.vectorstore = create_embeddings_from_pdf("temp_pdf.pdf")
                st.success("PDF processed! You can now ask questions.")


        if "vectorstore" in st.session_state:
                
            # question = st.text_input("Ask a question about the PDF:")
                if question:= st.chat_input("Ask a question about the PDF:"):
                    with st.chat_message("user"):
                        st.markdown(question)
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "user", "content": question})

                    ai_msg = answer_question_from_vectorstore(st.session_state.vectorstore, question)
                    with st.chat_message("assistant"):
                        st.write(ai_msg)
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_msg})
            
 

 

# if option!="Please select an option":
#     st.sidebar.info("Please click the '❌' to remove the file before clicking 'Clear' to reset the chat.")

if option!="Please select an option":
    if st.sidebar.button("Clear"):
                st.session_state.clear() 
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
          