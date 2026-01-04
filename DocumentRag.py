from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature = 0)

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
)

pdf_path = "Stock_Market_Performance_2024.pdf"

pdf_loader = PyPDFLoader(pdf_path)

pages = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

pages_split = text_splitter.split_documents(pages)

persist_dir = "C:\Users\Santan\PycharmProjects\PythonProject"

collection_name = "stock_market"

if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

vector_store = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name=collection_name,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs = {"k":6},
)





