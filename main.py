import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import os
import numpy as np
from langchain_community.llms import HuggingFaceHub
#from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeLangChain
import time

load_dotenv()

# Environment variable setup
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable is not set.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")


# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def setup_pinecone():
    index_name = "marketing-campaigns"
    dimension = 384  # For all-MiniLM-L6-v2 embeddings
    
    if index_name in pc.list_indexes().names():
        print(f"✅ Index '{index_name}' exists")
        return pc.Index(index_name)
    
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    while not pc.describe_index(index_name).status.ready:
        time.sleep(1)
    
    print("✅ Index ready!")
    return pc.Index(index_name)

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


# Debug print (temporary)
print("🔑 Pinecone key loaded:", os.getenv("PINECONE_API_KEY"))


# Set your OpenAI API key
#llm = ChatOpenAI(model = "gpt-4.0-mini", api_key = "API KEY", temperature = 0)
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",  # or "meta-llama/Llama-2-7b-chat-hf" #mistralai/Mistral-7B-Instruct-v0.2
    task = "text2text-generation",
    pipeline_kwargs={"max_new_tokens": 512},
)


# PDF loader
from langchain_community.document_loaders import PyPDFLoader
pdf_path = "/home/imad/Rag-version1/data-test/ESSB Showroom Poster.pdf"

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# Create text splitter
# Text or document (change dependending on the loader output)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(documents, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# embedding and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = PineconeLangChain.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name="marketing-campaigns"
    )
    return vectorstore


# Retrieve function
def retrieve_docs(vectorstore, k=3):
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":k}
    )
    return retriever


# #Prompt template to add persona to the bot
from langchain_core.prompts import ChatPromptTemplate

template = """
       You are a senior digital marketer.
       You are concise, practical, and give campaign-level suggestions: target audience, messaging, ad formats, copy variants, testing plan, and expected KPIs. 
       Use the retrieved past campaign data as evidence and suggest the best approach for the new campaign.

       Context: {context}
       Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)


# Chaining all
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(vectorstore):
    retriever = retrieve_docs(vectorstore)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        |prompt
        | llm
        | StrOutputParser()
)
    return rag_chain


def response(rag_chain, question):
    """
    Invokes the RAG chain with a specific question and returns the answer.
    """
    print(f"❓ Invoking chain with question: {question}")
    answer = rag_chain.invoke(question)
    return answer

# In your main execution block
if __name__ == "__main__":
    PDF_FILE_PATH = "path/to/your/marketing_report.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at '{pdf_path}'")
    else:
        print("✅ Starting the RAG pipeline...")
        
        print("Step 1: Setting up Pinecone index...")
        setup_pinecone()
        
        print("\nStep 2: Loading PDF document...")
        docs = load_pdf(pdf_path)
        
        print("\nStep 3: Chunking documents...")
        chunks = chunk_text(docs)
        
        print(f"\nStep 4: Creating vector store (this involves embedding)...")
        # THIS IS A LIKELY PLACE FOR IT TO HANG OR CRASH
        vectorstore = create_vectorstore(chunks)
        
        print("\nStep 5: Creating the RAG chain...")
        rag_chain = create_rag_chain(vectorstore)
        
        print("\n✅ RAG chain is ready. You can now ask questions.")
        print("   (Type 'quit', 'exit', or 'q' to end the session)")

        while True:
            # Ask the user to input a question
            question = input("\n❓ Your Question: ")

            # Check if the user wants to quit
            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Exiting program. Goodbye!")
                break

            # If not quitting, invoke the chain and get an answer
            print("🧠 Thinking...")
            answer = rag_chain.invoke(question)
            
            print("\n💡 Answer:")
            print(answer)



