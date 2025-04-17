from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import chromadb
import os

def build_chain():
    # Load logs from file
    with open("data/live_feed.txt", "r") as f:
        logs = f.read()

    # Split into chunks
    docs = [Document(page_content=logs)]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Force Chroma to use DuckDB instead of SQLite
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        client_settings=client_settings
    )

    # Create QA chain
    llm = ChatOpenAI(model="gpt-4-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return qa
