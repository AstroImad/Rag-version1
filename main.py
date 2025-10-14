import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import os
import numpy as np

# Set your OpenAI API key
#llm = ChatOpenAI(model = "gpt-3.5-turbo", api_key = "API KEY", temperature = 0)

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    text = loader.load()
    return text

# Create text splitter
# Text or document (change dependending on the loader output)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Load embedding model
#from langchain_openai import OpenAIEmbeddings  #when using OpenAI
from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings(chunks):
    #embedding_function = OpenAIEmbeddings()  #when using OpenAI
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedding_model.embed_documents(chunks)
    return vectors


# Setting up FAISS vector database
from langchain.vectorstores import Pinecone

def create_faiss_index(vectors, chunks):
    vectorstore = Pinecone.from_documents(chunks, vectors)
    return vectorstore


# Retrieval function
def retrieve_docs(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwags = {"k":3}
    )
    return retriever 

# #Prompt template to add persona to the bot
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_template(
    """You are a senior digital marketer.
       You are concise, practical, and give campaign-level suggestions: target audience, messaging, ad formats, copy variants, testing plan, and expected KPIs. 
       Use the retrieved past campaign data as evidence and suggest the best approach for the new campaign.

       Context: {context}
       Question: {question}
    """)

prompt = ChatPromptTemplate.from_messages(template = template, input_variables = ["context", "question"])


# Chaining all
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = retrieve_docs(vectorstore)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    |prompt
    | llm
    | StrOutputParser()
)
