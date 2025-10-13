import os
import sys
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Create text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        text_length = len(text)
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
from langchain.vectorstores import FAISS

def create_faiss_index(vectors, chunks):
    vectorstore = FAISS.from_documents(chunks, vectors)
    return vectorstore

# Retrieval function
def retrieve_docs(vectorstore, query, k=3):
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwags = {"k":3}
    )

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

# Example knowledge base (replace with your own documents)
documents = [
    "Python is a popular programming language.",
    "RAG stands for Retrieval-Augmented Generation.",
    "OpenAI provides powerful language models.",
    "Sentence Transformers are used for embeddings."
]

# Precompute document embeddings
doc_embeddings = embedder.encode(documents)

def retrieve_relevant_docs(query: str, k: int = 2) -> List[str]:
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [documents[i] for i in top_k_idx]

def generate_answer(query: str, context: List[str]) -> str:
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n" + "\n".join(context) + "\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2,
    )
    return response['choices'][0]['message']['content'].strip()

def chat():
    print("RAG LLM Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        context = retrieve_relevant_docs(user_input)
        answer = generate_answer(user_input, context)
        print("Bot:", answer)

if __name__ == "__main__":
    if not openai.api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    chat()