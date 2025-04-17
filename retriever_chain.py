from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

def build_chain():
    # Load latest logs
    with open("data/live_feed.txt", "r") as f:
        logs = f.read()

    # Split & embed
    docs = [Document(page_content=logs)]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")

    llm = ChatOpenAI(model="gpt-4-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa
