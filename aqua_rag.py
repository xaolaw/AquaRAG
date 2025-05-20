from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

app = FastAPI(
    title="AquaRagChatApp",
    summary="Summary of api app",
    description="""
    AquaRagChatApp API helps you do awesome stuff. 🚀
    ## Items
    You can **read items**.
    ## Users
    You will be able to:
    * **Create users** (_not implemented_).
    * **Read users** (_not implemented_).
    """,
    version="1.0",
)

load_dotenv()
embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"], truncate="END")
instruct_llm = ChatNVIDIA(model=os.environ["LLM"])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Jesteś chatbotem bazującym na dokumentach prawa wodnego. "
               "Twoim zadaniem jest pomaganie użytkownikowi poprzez odpowiadanie na jego pytania."
               "Użytkownik zadał następujące pytanie: {user_input}\n\n"
               "Na podstawie jego zapytania udało się nam uzyskać następujące możliwe przydatne informacje: "
               "Dokumenty: \n{db_context}\n\n"
               " (Odpowiedz tylko na podstawie danych z dokumentu. Niech Twoja odpowiedź będzie konwersacyjna)"),
    ('user', '{user_input}')
])

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/generator")
def generator():
    pass


