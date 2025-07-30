from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes
import os

app = FastAPI(
    title="AquaRagChatApp",
    summary="Summary of api app",
    description="""
    AquaRagChatApp API helps you do awesome stuff. 🚀
    """,
    version="1.0",
)

load_dotenv()
embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"],  NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"])
instruct_llm = ChatNVIDIA(model=os.environ["LLM"],  NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Jesteś chatbotem bazującym na dokumentach prawa wodnego. "
               "Twoim zadaniem jest pomaganie użytkownikowi poprzez odpowiadanie na jego pytania."
               "Użytkownik zadał następujące pytanie: {user_input}\n\n"
               "Na podstawie jego zapytania udało się nam uzyskać następujące możliwe przydatne informacje: "
               "Dokumenty: \n{db_context}\n\n"
               "(Odpowiedz tylko na podstawie danych z dokumentu. Niech Twoja odpowiedź będzie konwersacyjna)"),
    ('user', '{user_input}')
])

add_routes(
    app,
    instruct_llm,
    path="/nvida",
)

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/generator")
def generator():
    pass

@app.get("/retriver")
def retriver():
    pass

