import os
from typing import Annotated, Sequence, TypedDict, List
import logging

#logging.basicConfig(level=#logging.INFO)

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()
collection_name = 'prawo_wodne'

def vectorize_user_query(query: str) -> List[float]:
    embedder = NVIDIAEmbeddings(model = os.environ["EMBEDDER"])
    return embedder.embed_query(query)

@tool
def retrive_data_from_db(user_query: str) -> str:
    """Using user query input returns data from vector database as a string of anserws from document"""

    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    response = client.query_points(
        collection_name=collection_name,
        query=vectorize_user_query(query=user_query),
        limit=5,
        with_payload=True,
    )
    #logging.info("Using tool!")
    if not response:
        return "Nie znalazłem dokumentów odpowiednich do zadanego pytania"
    return "\n".join([i.payload['content']['page_content'] for i in response.points])



class RagState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Annotated[Sequence[BaseMessage], add_messages]

model = ChatNVIDIA(model=os.environ["LLM"], temprature = 0)



GENERATE_PROMPT = (
    "Jesteś botem odpowiadającym na zapytania użytkownika na podstawie dokumentu prawa wodnego. Udało Ci się uzyskać następujące dokumenty z bazy danych: "
    "Document z bazy:\n{context}\n\n "
    "Z ich pomocą odpowiedz na nsatępujące pytanie {question}\n"
)

"""TODO: Dodanie prompta systemowego lepsze zarządzanie historią czatu wraz z miejscem na odp z tool"""
def generate_query_context(state: RagState) -> RagState:
    #logging.info("generating query context")
    response = model.bind_tools([retrive_data_from_db]).invoke(state['messages'])
    #logging.info(response)
    return {
        "messages": state["messages"] + [response],
        "context": response
    }

def generate_user_answer(state: RagState) -> RagState:
    #logging.info("generating user answer")
    prompt = GENERATE_PROMPT.format(question=state["messages"][0], context=state["messages"][-1])
    #logging.info(prompt)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

from langgraph.prebuilt import tools_condition

graph = StateGraph(RagState)

graph.add_node(generate_query_context)
graph.add_node("retrieve", ToolNode([retrive_data_from_db]))
graph.add_node(generate_user_answer)

graph.add_edge(START, "generate_query_context")
graph.add_conditional_edges(
    "generate_query_context",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)
graph.add_edge("retrieve", "generate_user_answer")
graph.add_edge("generate_user_answer", END)

agent = graph.compile()

input = {"messages": "jakie są ceny za wydobywanie z wód powierzchniowych, wtym zmorskich wód wewnętrznych wraz zwodami wewnętrznymi Zatoki Gdańskiej? Podaj artykuł"}
for chunk in agent.stream(
    input
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")